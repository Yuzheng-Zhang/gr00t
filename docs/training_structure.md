# GR00T N1.7 训练流程与结构：一步一步讲

> 这份文档按“一次训练从按下回车到存盘”的顺序展开。每一站只解决一个问题，
> 建议按顺序读；读完应该能回答清楚：
>
> - 一个 batch 从哪来？
> - 它经过模型的哪些部分？
> - 哪部分参数在更新？
> - loss 怎么算？
> - checkpoint 里有什么，为什么续训要求同样的卡数？
>
> 示例 run：`runs/ball-placement-baseline/train-20261112-145726`（已跑完 10,000 步）。
> 当前 `examples/realman/train.sh` 默认已指向 `cup-stacking-baseline`，流程完全一样，
> 只是 `RUN_ROOT` / `SOURCE_DATASET` 不同。
>
> 代码基于当前工作树（相对官方上游有本地改动，见第 14 节）。函数行号可能随改动漂移，
> 所以正文以“文件 + 函数名”为主。

---

## 目录

1. [先看一张总图](#1-先看一张总图)
2. [第 1 站：终端命令 → 8 个训练进程](#2-第-1-站终端命令--8-个训练进程)
3. [第 2 站：命令行 → 配置单](#3-第-2-站命令行--配置单)
4. [第 3 站：run() 的开工准备](#4-第-3-站run-的开工准备)
5. [第 4 站：装配模型、处理器、数据集](#5-第-4-站装配模型处理器数据集)
6. [第 5 站：数据的三层结构](#6-第-5-站数据的三层结构)
7. [第 6 站：一个样本的诞生](#7-第-6-站一个样本的诞生)
8. [第 7 站：collator 把样本拼成 batch](#8-第-7-站collator-把样本拼成-batch)
9. [第 8 站：前向，两段式](#9-第-8-站前向两段式)
10. [第 9 站：反向传播与一步更新](#10-第-9-站反向传播与一步更新)
11. [第 10 站：保存与断点续训](#11-第-10-站保存与断点续训)
12. [把整条链路缩成一张表](#12-把整条链路缩成一张表)
13. [实际 run 的数字](#13-实际-run-的数字)
14. [容易误解的点](#14-容易误解的点)
15. [想改什么，动哪里](#15-想改什么动哪里)
16. [附：文件地图](#16-附文件地图)
17. [附：术语表](#17-附术语表)
18. [自检问题](#18-自检问题)

---

## 1. 先看一张总图

```
┌──────────────────────────────────────────────────────────────┐
│ 1. 入口层                                                     │
│    train.sh → torchrun → launch_finetune.py                  │
│    职责：把命令行参数变成 Config，启动 8 个进程                │
└───────────────────────────┬──────────────────────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────────┐
│ 2. 配置层                                                     │
│    FinetuneConfig → Config(model / data / training)          │
│    职责：决定用哪个模型、哪份数据、怎么训练                    │
└───────────────────────────┬──────────────────────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────────┐
│ 3. 装配层                                                     │
│    Gr00tN1d7Pipeline.setup()                                 │
│    model + processor + dataset + collator                    │
│    职责：把配置变成可训练的对象                                │
└───────────────────────────┬──────────────────────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────────┐
│ 4. 执行层                                                     │
│    Gr00tTrainer + DeepSpeed ZeRO-2                           │
│    batch → forward → loss → backward → step → save           │
│    职责：真正跑循环、更新参数、存 checkpoint                   │
└──────────────────────────────────────────────────────────────┘
```

先记住四个角色的分工：

| 层 | 主对象 | 一句话职责 |
|---|---|---|
| 入口层 | `train.sh`、`launch_finetune.py` | 把“人要训练”翻译成“进程和参数” |
| 配置层 | `Config`（model/data/training） | 一张完整的实验配置单 |
| 装配层 | `Gr00tN1d7Pipeline` | 按配置单造出模型、处理器、数据集、collator |
| 执行层 | `Gr00tTrainer` + DeepSpeed | 反复取 batch、前向、反传、更新、存盘 |

> **这一节记住**：训练代码不是“一个脚本从头写到尾”，而是
> **入口 → 配置 → 装配 → 执行** 四层。后面每一站都在填这四层里的细节。

---

## 2. 第 1 站：终端命令 → 8 个训练进程

### 2.1 你敲的命令

```bash
bash examples/realman/train.sh train
```

`train.sh` 做五件事：

1. **算数据视图路径**：用任务指令文件的哈希 + `SKIP_FIRST_EPISODES` 生成
   `data-<hash>-skip<N>`，保证改了指令或跳过条数后不会覆盖旧视图。
2. **校验动作语义**：执行 `realman_config.py`，确认 `action_semantics.json`
   里每组关节的 `representation`、单位、raw action 类型自洽。
3. **准备训练视图**：调用 `prepare_baseline.py`，复制/过滤元数据。
   `videos/` 始终软链接源数据；`data/` 默认也软链接，
   过滤 episode 时会生成独立的过滤后目录（避免统计量把被排除的 episode 算进去）。
4. **选卡**：如果没显式设置 `CUDA_VISIBLE_DEVICES`，用
   `examples/realman/select_gpu.py` 选 8 张空闲卡；然后断言 `torch.cuda.device_count() == 8`。
5. **启动**：

```bash
python -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=8 --max_restarts=0 \
  gr00t/experiment/launch_finetune.py ...
```

### 2.2 torchrun 做了什么

它启动 **8 个独立的 Python 进程**，每个进程：

- 跑同一份 `launch_finetune.py`
- 拿到不同的 `LOCAL_RANK`（0..7）、`RANK`（0..7）
- 共享 `WORLD_SIZE=8`
- 各自绑定一张 GPU（`experiment.py` 里 `torch.cuda.set_device(local_rank)`）

所以“分布式”不是脚本里写了一个 8 卡循环，而是**同一份代码被复制成 8 份，
靠 rank 决定各自读哪些 shard、谁写日志、谁存 checkpoint**。

> **这一站记住**：`torchrun` 只负责起 8 个进程；真正的分布式逻辑在
> `experiment.run()` 和数据集的分片过滤里。

---

## 3. 第 2 站：命令行 → 配置单

### 3.1 入口文件

`gr00t/experiment/launch_finetune.py` 做四件事：

```python
ft_config = tyro.cli(FinetuneConfig)                  # 1. 解析 CLI
ft_config.embodiment_tag = EmbodimentTag.resolve(...) # 2. 解析 embodiment
load_modality_config(ft_config.modality_config_path)  # 3. 注册 modality
config = get_default_config().load_dict({...})        # 4. 组装 Config
run(config)
```

### 3.2 CLI 参数去哪了

`FinetuneConfig` 只是一层“扁平参数”，会被手工映射到完整的 `Config`。
常用的对应关系：

| CLI 参数 | Config 字段 | 作用 |
|---|---|---|
| `--base-model-path` | `training.start_from_checkpoint` | 从哪个 GR00T checkpoint 初始化 |
| `--backbone-model-path` | `model.model_name` | 本地 Cosmos/Qwen3-VL 目录 |
| `--dataset-path` | `data.datasets[0].dataset_paths` | LeRobot 数据根目录 |
| `--modality-config-path` | 触发全局注册 | 新 embodiment 的 modality 定义 |
| `--embodiment-tag` | `data.datasets[0].embodiment_tag` | `NEW_EMBODIMENT` |
| `--num-gpus` | `training.num_gpus` | 必须等于 `WORLD_SIZE` |
| `--global-batch-size` | `training.global_batch_size` | 累积前的总 batch |
| `--gradient-accumulation-steps` | `training.gradient_accumulation_steps` | 累积次数 |
| `--max-steps` | `training.max_steps` | 优化器步数 |
| `--save-steps` | `training.save_steps` | 存 checkpoint 频率 |
| `--data-loader` | `data.loader` | `stock` 或 `fast` |
| `--shard-size` | `data.shard_size` | 每个 shard 的目标步数 |
| `--episode-sampling-rate` | `data.episode_sampling_rate` | 时间步切分份数 |

### 3.3 modality 注册为什么是“导入即生效”

`realman_config.py` 里：

```python
def register_config():
    config = build_config()
    register_modality_config(config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)

if __name__ != "__main__":
    realman_config = register_config()   # 被 import 时自动执行
```

`launch_finetune.py` 用 `importlib.import_module(path.stem)` 导入它，
于是每个 rank 的全局 `MODALITY_CONFIGS` 里都有了 `new_embodiment`。
如果不导入，`config.validate()` 会报“没有为 embodiment tag 注册 modality”。

> **这一站记住**：CLI 只是入口；训练代码真正消费的是 `Config` 这棵树。
> modality 注册是“导入 Python 文件”的副作用，不是普通配置项。

---

## 4. 第 3 站：run() 的开工准备

入口：`gr00t/experiment/experiment.py` 的 `run(config)`。

它按顺序做这些事：

### 4.1 一致性检查

- `_assert_num_gpus_matches_world_size`：`config.training.num_gpus` 必须等于
  `WORLD_SIZE`。否则 `per_device_batch = global_batch / num_gpus` 会被悄悄放大。
- `global_batch_size % num_gpus == 0`：保证每卡 batch 是整数。
- `check_resume_compatibility`：`save_only_model=True` 不能和
  `resume_from_checkpoint=True` 同时用。
- `config.validate()`：modality 是否注册、mix ratio 是否为正、
  fp16/bf16 是否冲突、动作配置是否补全。

### 4.2 环境初始化

- `_init_distributed_process_group()`：`WORLD_SIZE>1` 时初始化 NCCL，
  并 `torch.cuda.set_device(local_rank)`。
- `set_seed(config.data.seed)`：默认 42。
- 非 rank0 的日志级别调高，避免 8 份重复日志。

### 4.3 输出目录与配置落盘

- 如果 `experiment_name is None`：直接把 `output_dir` 当实验目录，
  `experiment_name` 取目录 basename。
- 否则：`output_dir/experiment_name`。
- rank0 写：
  - `experiment_cfg/config.yaml`：完整 Config（safe YAML）
  - `experiment_cfg/conf.yaml`：OmegaConf 版本
  - `wandb_config.json`
- rank0 还会在装配完成后保存 `processor/` 和 `dataset_statistics.json`。

### 4.4 选 Pipeline

```python
pipeline = MODEL_REGISTRY.get(type(config.model))(config, save_cfg_dir)
pipeline.setup()
```

`type(config.model)` 是 `Gr00tN1d7Config`，注册表把它映射到
`Gr00tN1d7Pipeline`。这一层是“模型无关”的：以后换模型，只要注册新的
Config + Pipeline 即可，`experiment.run()` 不用改。

### 4.5 建 Trainer

- 8 卡且 `use_ddp=False` → 读 `configs/deepspeed/zero2_config.json`。
- `per_device_train_batch_size = global_batch_size // num_gpus = 1`。
- 组装 `TrainingArguments`：
  - `max_steps=10000`
  - `bf16=True`、`tf32=True`
  - `lr_scheduler_type="cosine"`
  - `warmup_ratio=0.05`
  - `save_steps=1000`
  - `eval_strategy="no"`
  - `remove_unused_columns=False`
- 建 `Gr00tTrainer`，挂三个 callback：
  - `LocalTrainingMonitor`：写 `training_metrics.jsonl`、`training_status.json`、`loss.png`
  - `CheckpointFormatCallback`：把 `experiment_cfg/`、`processor/` 复制进每个 checkpoint
  - `BestMetricCheckpointCallback`：只在配置了 eval 指标时启用（当前没有）
- 最后 `trainer.train(resume_from_checkpoint=...)`。

> **这一站记住**：`run()` 不改数据、不改模型结构，它只负责
> **环境、目录、配置、Trainer**。真正的“造零件”在下一站。

---

## 5. 第 4 站：装配模型、处理器、数据集

入口：`gr00t/model/gr00t_n1d7/setup.py` 的 `Gr00tN1d7Pipeline.setup()`。

它只做四件事：

```python
self.model = self._create_model()
self.train_dataset, self.eval_dataset = self._create_dataset(self.save_cfg_dir)
self.data_collator = self._create_collator()
```

### 5.1 造模型 `_create_model()`

```python
model, loading_info = AutoModel.from_pretrained(
    config.training.start_from_checkpoint,   # models/GR00T-N1.7-3B
    tune_llm=False,
    tune_visual=False,
    tune_projector=True,
    tune_diffusion_model=True,
    tune_vlln=True,
    ...
)
```

发生的事：

1. 从 `GR00T-N1.7-3B/config.json` 读出 `Gr00tN1d7Config`。
2. `AutoModel` 注册表按 `model_type="Gr00tN1d7"` 找到 `Gr00tN1d7` 类。
3. `Gr00tN1d7.__init__` 建两部分：
   - `Qwen3Backbone`：加载本地 `Cosmos-Reason2-2B`（Qwen3-VL）
   - `Gr00tN1d7ActionHead`：状态编码器、动作编码器、DiT、动作解码器
4. `Qwen3Backbone` 里：
   - `while len(layers) > 12: pop(-1)`，语言塔只保留前 12 层
   - `tune_llm=False` → 语言塔 `requires_grad_(False)`
   - `tune_visual=False` → 视觉塔 `requires_grad_(False)`
5. 从 GR00T checkpoint 加载权重；`missing/unexpected/mismatched` 都会报错，
   只有 `mask_token` 缺失允许重新初始化。
6. 打印参数量。

当前 run 的实际量级（按 checkpoint 张量统计）：

| 模块 | 参数量 | 是否训练 |
|---|---:|---|
| backbone 语言塔（12 层） | 1.12B | 冻结 |
| backbone 视觉塔 | 407M | 冻结 |
| backbone `lm_head` | 311M | 冻结 |
| DiT（16 blocks） | 1.09B | 训练 |
| `vl_self_attention` | 201M | 训练 |
| state/action encoder + decoder + vlln | 327M | 训练 |
| **合计** | **3.455B** | **可训练约 1.62B** |

### 5.2 造处理器 `_create_dataset()` 的前半段

```python
processor = AutoProcessor.from_pretrained(
    config.training.start_from_checkpoint,   # GR00T-N1.7-3B
    modality_configs=config.data.modality_configs,
    use_percentiles=True,
    state_dropout_prob=0.2,
    ...
)
```

处理器负责：

- 读 checkpoint 里的 `processor_config.json`
- 用新的 `modality_configs` 覆盖 `new_embodiment` 的输入输出定义
- 建 `StateActionProcessor`（归一化/反归一化、绝对/相对动作）
- 建图像增强 pipeline
- 建 collator

注意：processor 里还带着预训练时的 `statistics.json` 和 `embodiment_id.json`。
`new_embodiment` 的统计量是接下来从你的数据集现算的，不是从 checkpoint 继承的。

### 5.3 造数据集 `_create_dataset()` 的后半段

```python
dataset_factory = DatasetFactory(config)
train_dataset, eval_dataset = dataset_factory.build(processor)
```

`DatasetFactory.build()`：

1. 对每个 dataset path，rank0 执行：
   - `generate_stats(dataset_path)`：算 state/action 的 min/max/q01/q99/mean/std
   - `generate_rel_stats(...)`：算相对动作统计量（当前 ABSOLUTE 配置下不参与）
   - 其他 rank 在 barrier 等 rank0 写完
2. 实例化 `ShardedSingleStepDataset`（`stock`）或 `FastShardedSingleStepDataset`（`fast`）
3. 用每个数据集的长度 × `mix_ratio` 算采样权重
4. 包成 `ShardedMixtureDataset` / `FastShardedMixtureDataset`
5. 合并统计量，调用 `processor.set_statistics(..., override=True)`
6. 把统计量写进 `experiment_cfg/dataset_statistics.json`

`eval_dataset` 当前固定为 `None`，因为 `DatasetFactory` 断言
`eval_strategy == "no"`。

### 5.4 造 collator `_create_collator()`

直接返回 `processor.collator`，也就是 `Gr00tN1d7DataCollator`。

> **这一站记住**：model / processor / dataset / collator 是四个独立装配的零件。
> 训练循环只认 collator 产出的 dict，不关心前面数据是怎么读的。

---

## 6. 第 5 站：数据的三层结构

这是最容易看晕的地方。把它拆成三层：

```
第 1 层  LeRobotEpisodeLoader
         读 meta/info.json、data/*.parquet、videos/*
         产出：一条 episode 的 DataFrame（state/action/video/language 列）

第 2 层  ShardedSingleStepDataset
         把 episode 切成“单个时间步”样本
         再装进 shard（约 128 步一个）
         产出：一个 shard 的样本列表

第 3 层  ShardedMixtureDataset
         决定“这一轮按什么顺序取哪些 shard”
         按 rank/worker 分配
         shard 内再 shuffle
         产出：一次一个样本
```

### 6.1 第 1 层：EpisodeLoader

`gr00t/data/dataset/lerobot_episode_loader.py`：

- 从 `meta/info.json` 读 episode 数量、长度、fps
- 从 `data/**/*.parquet` 读低维数据
- 从 `videos/**/*` 解码视频
- 每列命名规则：`state.left_arm`、`action.left_arm`、
  `video.left_wrist`、`language.annotation.human.task_description`

`stock` loader 会把一条 episode 的所有帧都解码出来；
`fast` loader 只解码当前 shard 真正会读到的那几帧。

### 6.2 第 2 层：SingleStepDataset

`gr00t/data/dataset/sharded_single_step_dataset.py`：

**第一步：算有效长度**

```python
effective_length = original_length - action_horizon + 1
```

因为每个样本要取未来 40 步动作，最后 39 个时刻凑不满一个完整 action chunk，
所以直接不采样。

**第二步：切分时间步**

```python
num_splits = int(1 / episode_sampling_rate)   # 0.1 -> 10
step_indices = np.arange(effective_length)
rng.shuffle(step_indices)
for i in range(num_splits):
    split = step_indices[i::num_splits]       # 交错取 1/10
```

这一步很关键：`episode_sampling_rate=0.1` **不是只保留 10% 的 episode，
也不是只保留 10% 的时间步**，而是把每条 episode 的时间步交错分成 10 份，
最终所有时间步都会进入 shard。

**第三步：装 shard**

```python
num_shards = min(ceil(total_steps / shard_size), len(episode_splits))
```

然后贪心地把每个 split 分配到当前最短的 shard，保证各 shard 长度尽量均衡。

你的 ball-placement 数据：

```
50 条 episode
  → 合计 34,349 个有效时间步
  → 每条切成 10 份，共约 500 个 split
  → 装进 269 个 shard
  → 平均每个 shard 127.69 步，标准差 21.48
```

**shard 里存的是什么**

不是张量，而是 `(episode_index, step_indices)` 的列表。
真正取数据时才 `load_episode` + `extract_step_data` + `processor`。

### 6.3 第 3 层：MixtureDataset

`gr00t/data/dataset/sharded_mixture_dataset.py`：

**采样权重**

```python
weight = (dataset_length / total_length) * mix_ratio
```

多数据集时再按平均 shard 长度归一化，避免“shard 大的数据集”被过度采样。
`ds_weights_alpha` 非空时，改用 `len(dataset)^alpha` 作为权重。

**生成调度表**

```python
rng = np.random.default_rng(seed + epoch)
dataset_sampling_schedule = rng.choice(
    num_datasets, size=num_shards_per_epoch, p=normalized_weights
)
```

每个 “epoch” 生成 `num_shards_per_epoch=100000` 个 shard 抽样，
shard 用完后重新 shuffle 再循环。

**按 rank/worker 切分**

```python
if i % (world_size * num_workers) == rank * num_workers + worker_id:
    filtered_schedule.append(shard)
```

8 卡、`num_workers=0` 时，rank0 拿第 0,8,16... 个 shard，
rank1 拿第 1,9,17... 个，依此类推。

**后台预取 + shard 内 shuffle**

- 用一个后台线程提前加载下一个 shard
- 当前 shard 的样本顺序再用 `rng.shuffle` 打乱
- 所以数据顺序 = “shard 随机调度 + shard 内随机”

**“epoch” 在这里几乎没意义**

10,000 步 × 8 卡 = 80,000 个样本，约 625 个 shard；
而一个调度表有 100,000 个 shard 位置。训练根本走不完一个调度表，
所以实际效果是“在一个固定随机序列里采样一个子集”。

> **这一站记住**：数据不是按 episode 喂的，是按 shard 喂的。
> shard 是 I/O 和随机化的基本单位；`stock` 一次加载整个 shard，
> `fast` 则流式解码。

---

## 7. 第 6 站：一个样本的诞生

入口：`gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py`。

### 7.1 先按 delta_indices 取原始数据

`extract_step_data()`：

```python
for modality, config in modality_configs.items():
    indices = [step_index + delta for delta in config.delta_indices]
    for key in config.modality_keys:
        data = episode_data[f"{modality}.{key}"].iloc[indices]
```

对当前配置：

| modality | delta_indices | 取到什么 |
|---|---|---|
| video | `[0]` | 当前时刻 3 路 RGB |
| state | `[0]` | 当前时刻 4 组关节，共 58 维 |
| action | `0..39` | 未来 40 步，每步 58 维 |
| language | `[0]` | 任务文本 |

### 7.2 Processor 做归一化和拼装

`Gr00tN1d7Processor.__call__()`：

1. **state**：4 组分别按 q01/q99 min-max 归一化到 `[-1,1]`，clip 离群值；
   拼成 58 维；右侧补零到 `max_state_dim=132`。
2. **action**：同样逐组归一化；拼成 `40×58`；补零到 `40×132`；
   生成 `action_mask`：前 `40×58` 为 1，其余为 0。
3. **绝对/相对动作**：`action_configs.rep` 是 `ABSOLUTE`，
   所以只做归一化，不做“绝对 → 相对”转换。
4. **state dropout**：训练时以 `state_dropout_prob=0.2` 的概率把 state 置零。
5. **图像**：三路图分别做 albumentations：
   最短边缩到 256 → 随机裁剪约 90% → 再缩到 256。
6. **语言**：转小写、去标点（`formalize_language=True`）。
7. **VLM 输入**：把 3 张图和文本套进 Qwen3-VL 的 chat template，
   得到 `vlm_content = {text, images, conversation}`。
8. **embodiment_id**：`new_embodiment` 对应的整数索引。

单个样本（还没有 batch 维）大致是：

| 字段 | 形状 | 说明 |
|---|---|---|
| `state` | `[1, 132]` | 58 有效 + 74 padding |
| `action` | `[40, 132]` | 40 步 × (58 有效 + 74 padding) |
| `action_mask` | `[40, 132]` | 前 40×58 为 1 |
| `embodiment_id` | `[]` | 标量 |
| `vlm_content.text` | 字符串 | 套好 chat template 的文本 |
| `vlm_content.images` | 3 张图 | 三路相机 |

> **这一站记住**：一个样本 =
> 当前时刻的三路图 + 当前 58 维状态 + 未来 40 步 58 维动作 + 一句任务文本。
> 所有数值都在 processor 里完成归一化和 padding。

---

## 8. 第 7 站：collator 把样本拼成 batch

入口：`Gr00tN1d7DataCollator.__call__()`。

它做两件事：

### 8.1 文本和图像走 Qwen processor

```python
vlm_inputs = self.processor(
    text=text_list,
    images=image_inputs,
    return_tensors="pt",
    padding=True,
)
```

- `text_list` 长度 = batch size
- `image_inputs` 长度 = batch size × 3
- tokenizer 设置为 **左 padding**（Flash Attention 要求）
- 输出 `input_ids`、`attention_mask`、`pixel_values`、`image_grid_thw`

### 8.2 数值字段直接 stack

```python
batch[key] = torch.from_numpy(np.stack(values))
```

单卡 batch=1 时，模型看到：

| 字段 | 形状 |
|---|---|
| `state` | `[1, 1, 132]` |
| `action` | `[1, 40, 132]` |
| `action_mask` | `[1, 40, 132]` |
| `embodiment_id` | `[1]` |
| `input_ids` | `[1, T]` |
| `attention_mask` | `[1, T]` |
| `pixel_values` | `[N_patch, D]` |
| `image_grid_thw` | `[3, 3]` |

注意：**全局 batch=8 是 8 张卡各 1 个样本拼出来的**，
单张卡上的模型输入 batch 维仍然是 1。

> **这一站记住**：collator 只做 padding、tokenize 和 stack；
> 归一化、增强、动作拼接都在上一站已经做完了。

---

## 9. 第 8 站：前向，两段式

入口：`Gr00tN1d7.forward(inputs)`。

### 9.1 先把输入分成两包

```python
backbone_inputs, action_inputs = self.prepare_input(inputs)
```

- `backbone_inputs`：`input_ids`、`attention_mask`、`pixel_values`、`image_grid_thw`
- `action_inputs`：`state`、`action`、`action_mask`、`embodiment_id`

### 9.2 第一段：Backbone（冻结）

`Qwen3Backbone.forward()`：

```python
outputs = self.model(**vl_input, output_hidden_states=True)
hidden = outputs.hidden_states[-1]          # [B, T, 2048]
image_mask = input_ids == image_token_id
attention_mask = attention_mask == 1
```

- 语言塔只跑前 12 层
- 视觉塔完整
- 因为 `tune_llm=False`、`tune_visual=False`，这段**不产生梯度**
- 每步前向都会把冻结模块设成 `eval()`，避免 dropout 干扰

输出：

```
backbone_features:      [B, T, 2048]
backbone_attention_mask:[B, T]
image_mask:             [B, T]
```

### 9.3 第二段：Action Head（训练）

`Gr00tN1d7ActionHead.forward()`：

**① 处理 VL 特征**

```python
backbone_features = self.vlln(backbone_features)
backbone_features = self.vl_self_attention(backbone_features, ...)
```

- `vlln`：LayerNorm
- `vl_self_attention`：对 VL token 再做几层自注意力（约 201M 参数）

**② 编码 state**

```python
state = state.view(B, 1, -1)          # [B, 1, 132]
state_features = state_encoder(state, embodiment_id)   # [B, 1, 1536]
```

- `CategorySpecificMLP`：按 embodiment_id 选不同的 MLP 表
- 训练时还会再做一次 state dropout（0.2）

**③ Flow matching：给动作加噪**

```python
t = Beta(1.5, 1.0).sample() * 0.999
noise = torch.randn_like(action)
noisy_action = (1 - t) * noise + t * action
velocity = action - noise
```

直觉：

- `t=0`：纯噪声
- `t=1`：干净动作
- 模型不直接预测动作，而是预测**从噪声指向干净动作的速度**
- `Beta(1.5,1)` 再取 `1-x`，让训练更关注高噪声区间（t 较小）

**④ 编码加噪动作**

```python
action_features = action_encoder(noisy_action, t_discretized, embodiment_id)
```

- `MultiEmbodimentActionEncoder`：动作 MLP + 时间步 embedding
- 输出 `[B, 40, 1536]`

**⑤ 拼 state 和 action**

```python
sa_embs = torch.cat((state_features, action_features), dim=1)
# [B, 1 + 40, 1536]
```

**⑥ DiT**

```python
model_output = self.model(
    hidden_states=sa_embs,                 # [B, 41, 1536]
    encoder_hidden_states=vl_embeds,       # [B, T, 2048]
    timestep=t_discretized,
)
```

- 16 层 `AlternateVLDiT`
- 每层：self-attention（state+action 内部）+ cross-attention（看 VL 特征）
- 输出 `[B, 41, 1024]`

**⑦ 解码成速度**

```python
pred = self.action_decoder(model_output, embodiment_id)  # [B, 41, 132]
pred_actions = pred[:, -40:]                             # [B, 40, 132]
```

**⑧ 算 loss**

```python
action_loss = MSE(pred_actions, velocity) * action_mask
loss = action_loss.sum() / (action_mask.sum() + 1e-6)
```

- `action_mask` 只在前 `40×58` 为 1
- padding 的 74 个维度和超出 horizon 的部分不计损失
- 每个样本的分母是 `40 × 58 = 2320`

当前配置 `strict_action_padding_mask=None`，在 action head 里会被解析成 `False`。
这意味着 padding 维度仍然会进入 encoder/DiT，只是最后被 loss mask 掉。
如果改成 `True`，padding 会在 encoder 和 DiT 内部也被屏蔽，
但推理时必须用 PyTorch action head（TensorRT 路径无法复现该 mask）。

### 9.4 推理时怎么用

训练学的是速度场；推理时从纯噪声开始，做 4 次 Euler 积分：

```python
actions = noise
for t in range(4):
    v = model(actions, t)
    actions = actions + (1/4) * v
```

`num_inference_timesteps=4` 就是这里的 4 步。
训练时的 RTC（real-time chunking）逻辑在 `get_action_with_features()` 里，
训练阶段不走。

> **这一站记住**：骨干只负责把图和文字变成特征；
> 动作头负责把“特征 + 状态 + 噪声动作”变成速度场，再用速度场算 loss。

---

## 10. 第 9 站：反向传播与一步更新

### 10.1 一步 optimizer step 的时间线

```
for optimizer_step in 1..10000:
    for micro_step in 1..4:
        batch = next(dataloader)          # 全 8 卡共 8 个样本
        loss = model(batch)               # forward
        loss.backward()                   # 梯度累积
    deepspeed.step()                      # 梯度 all-reduce + optimizer + scheduler
    log every 10 steps
    save every 1000 steps
```

- 每个 micro-step：8 卡 × 每卡 1 样本 = **8 个样本**
- 累积 4 次：**32 个样本 / optimizer step**
- 10,000 步：**320,000 次样本展示**

### 10.2 DeepSpeed ZeRO-2 做了什么

`gr00t/configs/deepspeed/zero2_config.json`：

```json
{
  "zero_optimization": {
    "stage": 2,
    "overlap_comm": true,
    "reduce_scatter": true,
    "allgather_partitions": true,
    "contiguous_gradients": true
  },
  "bf16": {"enabled": "auto"},
  "train_batch_size": "auto",
  "gradient_accumulation_steps": "auto"
}
```

ZeRO-2 的分工：

- **模型参数**：每卡一份（不做参数切分）
- **梯度**：all-reduce 后按卡切分
- **优化器状态**：按卡切分（exp_avg、exp_avg_sq 等）

所以 checkpoint 里会出现 8 份 `bf16_zero_pp_rank_*_optim_states.pt`。
换卡数续训时，这些分片对不上，所以必须保持 8 卡。

### 10.3 优化器与调度器

| 项 | 值 |
|---|---|
| optimizer | `adamw_torch` |
| lr | `1e-4` |
| scheduler | cosine |
| warmup | 5%（前 500 步） |
| weight decay | `1e-5` |
| grad clip | `1.0` |
| precision | bf16 autocast + TF32 |

### 10.4 日志

- 每 10 步：`loss`、`learning_rate`、`grad_norm`
- `LocalTrainingMonitor` 额外写：
  - `training_metrics.jsonl`（逐行 JSON）
  - `training_status.json`（当前阶段/步数/最新 loss）
  - `loss.png`
  - `progress.html`（10 秒自动刷新）
- 没有 wandb 网络日志（`use_wandb=False`）

> **这一站记住**：一次 optimizer step = 4 次 micro-forward/backward；
> 每次 micro-step 全 8 卡共 8 个样本，累积后有效 batch 是 32。

---

## 11. 第 10 站：保存与断点续训

### 11.1 每个 checkpoint 里有什么

`checkpoint-10000/`：

```
model-00001-of-00002.safetensors      # 合并后的完整权重，可直接推理
model-00002-of-00002.safetensors
model.safetensors.index.json
config.json
processor_config.json
statistics.json
embodiment_id.json
global_step10000/
  mp_rank_00_model_states.pt
  bf16_zero_pp_rank_0_optim_states.pt
  ...
  bf16_zero_pp_rank_7_optim_states.pt
scheduler.pt
rng_state_0.pth ... rng_state_7.pth
trainer_state.json
training_args.bin
experiment_cfg/                        # CheckpointFormatCallback 复制进来
```

两类东西要分清：

- **推理用**：`model-*.safetensors` + `config.json` + `processor_config.json` + `statistics.json`
- **续训用**：`global_step*/` + `scheduler.pt` + `rng_state_*.pth` + `trainer_state.json`

### 11.2 续训流程

`Gr00tTrainer.train()`：

1. `resume_from_checkpoint=True` 时，用 `get_last_checkpoint(output_dir)` 找最新 checkpoint。
2. 找不到就**直接报错**，不会悄悄从头开始。
3. 先读 `trainer_state.json` 到 `self.state`，让 `get_train_dataloader()` 能拿到
   `global_step`。
4. 用 `seed + global_step` 重置数据集种子：
   ```python
   new_seed = train_dataset.seed + curr_global_step
   train_dataset.reset_seed(new_seed)
   ```
5. `super().train(resume_from_checkpoint=...)` 恢复模型/优化器/调度器/RNG。

你的 `resume_training.sh` 在启动前还做了断言：

- 存在 `checkpoint-*`
- `global_step < max_steps`
- `scheduler.pt` 存在
- 8 个 `*optim_states.pt`
- 8 个 `rng_state_*.pth`
- `processor_config.json` 里动作表示还是 `ABSOLUTE`

### 11.3 训练结束

`experiment.run()` 最后：

```python
trainer.save_model()
```

把最终的模型权重、config、processor 写到 `output_dir` 根目录。
此时根目录本身就是一个可被 `Gr00tPolicy.from_pretrained` 加载的模型目录。

> **这一站记住**：checkpoint 不只是模型权重，还有优化器/调度器/RNG；
> 所以续训必须同卡数，且不能和 `save_only_model=True` 混用。

---

## 12. 把整条链路缩成一张表

| 阶段 | 主对象 | 输入 | 输出 |
|---|---|---|---|
| 入口 | `train.sh` | 模式、环境变量 | 数据视图路径、8 个进程 |
| 配置 | `FinetuneConfig` → `Config` | CLI 参数 | model/data/training 三棵树 |
| 装配-模型 | `Gr00tN1d7Pipeline._create_model` | base checkpoint + 本地 backbone | 3.455B 参数的模型 |
| 装配-处理器 | `_create_dataset` 前半 | checkpoint processor + modality | 归一化/增强/collator |
| 装配-数据 | `DatasetFactory.build` | 数据路径 + shard 参数 | 269 个 shard 的混合数据集 |
| 装配-collator | `processor.collator` | 单样本 dict | batch 张量 |
| 执行-取数 | `ShardedMixtureDataset.__iter__` | shard 调度 | 一个时间步样本 |
| 执行-前向 | `Gr00tN1d7.forward` | batch | loss |
| 执行-反传 | `Gr00tTrainer` + DeepSpeed | loss | 更新后的参数 |
| 执行-存盘 | `Trainer.save_model` | 参数/优化器状态 | checkpoint + 最终模型 |

---

## 13. 实际 run 的数字

以 `runs/ball-placement-baseline/train-20261112-145726` 为例：

| 项 | 值 | 来源 |
|---|---|---|
| episode 数 | 50 | 数据集 |
| 有效时间步 | 34,349 | 训练日志 |
| shard 数 | 269 | 训练日志 |
| 平均 shard 长度 | 127.69 | 训练日志 |
| shard 长度标准差 | 21.48 | 训练日志 |
| GPU 数 | 8 | config |
| 每卡 batch | 1 | `global_batch_size / num_gpus` |
| 梯度累积 | 4 | config |
| 有效 batch | 32 | 8 × 4 |
| 训练步数 | 10,000 | config |
| 样本展示次数 | 320,000 | 10,000 × 32 |
| 约等于数据遍数 | 9.3 | 320,000 / 34,349 |
| 最终 loss | 1.0263 | `training_status.json` |
| 最终 grad norm | 0.12 | `training_metrics.jsonl` |
| 总参数 | 3.455B | checkpoint 张量统计 |
| 可训练参数 | 约 1.62B | 同上 |
| checkpoint 间隔 | 1,000 步 | config |
| checkpoint 保留 | 全部（resume 脚本 `save_total_limit=0`） | config |

> 注意：loss 是 flow-matching 的 MSE 速度损失，不是真机成功率。
> 训练 loss 下降只能说明模型在拟合数据中的动作分布，不能证明任务成功。

---

## 14. 容易误解的点

### 14.1 `episode_sampling_rate=0.1` 不是只用 10% 数据

它把每条 episode 的时间步交错切成 10 份，再装进 shard。
所有时间步最终都会进入训练视图。日志里的 34,349 就是全量有效时间步。

### 14.2 state dropout 被应用了两次

- **processor 层**：`state_dropout_prob=0.2`，直接把归一化后的 state 置零
- **action head 层**：`state_dropout_prob=0.2`，再把 state feature 置零

两者独立，实际“至少丢一次”的概率约为：

```
1 - (1 - 0.2) × (1 - 0.2) = 36%
```

如果发现模型对 state 利用不足，这是第一个该检查的参数。

### 14.3 `use_relative_action=True` 不等于用了相对动作

`launch_finetune.py` 硬编码了：

```python
config.model.use_relative_action = True
```

但 `realman_config.py` 里 `action_configs.rep = ABSOLUTE`。
`StateActionProcessor.apply_action()` 只在 `rep == RELATIVE` 时才做绝对→相对转换，
所以当前 run 实际仍是**绝对动作 + 逐组归一化**。

想切相对动作，需要同时改：

- `action_semantics.json` 的 `representation`
- `realman_config.py` 的 `ActionConfig.rep`
- 确认 `raw_action_kind`、单位、state_key 都匹配

### 14.4 没有验证集

`DatasetFactory.build()` 直接断言：

```python
assert config.training.eval_strategy == "no"
```

`eval_dataset` 固定为 `None`。所以当前训练没有 hold-out 验证，
也没有 best-checkpoint 选择。

### 14.5 `select_layer=12` 是截断，不是取第 12 层输出

`Qwen3Backbone.__init__`：

```python
while len(self.model.language_model.layers) > select_layer:
    self.model.language_model.layers.pop(-1)
```

它把第 12 层之后的语言层全部删掉。骨干实际前向只有 12 层，
但视觉塔是完整的。

### 14.6 `global_batch_size` 是累积前的

`global_batch_size=8` 指的是“一次 forward/backward 全卡合计 8 个样本”，
不是一次 optimizer step 的 8 个。乘上 `gradient_accumulation_steps=4` 后，
一次 optimizer step 才是 32 个样本。

### 14.7 `save_total_limit=0` 表示不轮转

HF Trainer 的 `_rotate_checkpoints()`：

```python
if save_total_limit is None or save_total_limit <= 0:
    return
```

所以 0 是“保留全部”。早期 `train.sh` 默认是 3，会删旧 checkpoint；
resume 脚本改成 0 后，后续 checkpoint 都留着。

### 14.8 续训必须同卡数

ZeRO-2 的优化器状态按 rank 切分，8 卡训练就有 8 份
`bf16_zero_pp_rank_*_optim_states.pt`。换成 4 卡或 1 卡都无法直接恢复。
`resume_training.sh` 里对此有显式断言。

### 14.9 `num_gpus` 必须等于 `WORLD_SIZE`

`experiment.py` 会检查：

```python
if int(os.environ["WORLD_SIZE"]) != config.training.num_gpus:
    raise ValueError(...)
```

否则 `per_device_batch_size = global_batch_size // num_gpus` 会被算错，
实际 batch 可能放大 8 倍。

### 14.10 `max_steps` 覆盖 epoch 概念

数据集是 `IterableDataset`，没有真正的 `len()` 和 epoch。
`TrainingArguments(max_steps=10000)` 直接决定循环次数；
`num_train_epochs` 不起作用。

### 14.11 `dataloader_num_workers=0` 意味着主进程取数

当前 run 是 `stock` loader + 0 worker：

- 每个 rank 的主进程同步加载整个 shard（约 128 步）
- 8 个 rank 各自加载自己的 shard
- 如果 I/O 是瓶颈，可以考虑：
  - `--data-loader fast`（流式、只解码需要的帧）
  - 增加 `dataloader_num_workers`（但 `num_shards_per_epoch` 必须至少是
    `world_size × num_workers`）

### 14.12 加新模态不是只改 config

当前 processor 和模型只认 `video/state/action/language`。
加触觉、力矩等新模态，需要同时改：

- modality config
- `extract_step_data` / episode loader（读取新列）
- processor（归一化、拼装、mask）
- action head / backbone 的输入拼接
- collator（stack 新字段）
- 推理端的 observation/action 解码

只改 YAML 或 `realman_config.py` 不够。

---

## 15. 想改什么，动哪里

| 目标 | 位置 |
|---|---|
| 换任务指令 | `examples/realman/task_instruction.txt` |
| 换数据源 | `train.sh` 的 `SOURCE_DATASET` / `RUN_ROOT` |
| 跳过前 N 条 episode | `train.sh` 的 `SKIP_FIRST_EPISODES` |
| 改相机/关节/动作语义 | `examples/realman/realman_config.py`、`action_semantics.json` |
| 改 action horizon | `realman_config.py` 的 `delta_indices` + `model.action_horizon` |
| 解冻语言/视觉 | `--tune-llm` / `--tune-visual` |
| 改 batch/lr/步数 | `train.sh` 的 `BATCH_SIZE`、`GRAD_ACCUM`、`MAX_STEPS`、`--learning-rate` |
| 改 shard/采样 | `--shard-size`、`--episode-sampling-rate`、`--data-loader` |
| 改图像增强 | `FinetuneConfig.random_rotation_angle` / `color_jitter_params` / `extra_augmentation_config` |
| 改模型容量 | `Gr00tN1d7Config.diffusion_model_cfg`（层数/宽度） |
| 改归一化方式 | `--use-percentiles` / `model.use_mean_std` |
| 加验证集 | 需要改 `DatasetFactory`（当前直接禁用） |
| 加新模态 | processor + model + collator + 推理端，见 14.12 |
| 改本地监控 | `gr00t/experiment/local_monitor.py` |
| 改断点续训策略 | `runs/*/resume_training.sh`、`queue_resume_training.py` |

---

## 16. 附：文件地图

| 文件 | 职责 |
|---|---|
| `examples/realman/train.sh` | RealMan 训练入口：准备数据、选卡、torchrun |
| `examples/realman/prepare_baseline.py` | 创建/过滤训练元数据视图，软链接视频 |
| `examples/realman/realman_config.py` | 注册 `NEW_EMBODIMENT` 的 modality |
| `examples/realman/action_semantics.json` | 动作/状态单位与表示的事实来源 |
| `examples/realman/select_gpu.py` | 选空闲 GPU |
| `gr00t/configs/finetune_config.py` | CLI 参数定义 |
| `gr00t/experiment/launch_finetune.py` | CLI → Config → run |
| `gr00t/experiment/experiment.py` | 训练主流程、Trainer 组装 |
| `gr00t/experiment/trainer.py` | 自定义 DataLoader、续训种子、loss 日志 |
| `gr00t/experiment/local_monitor.py` | 本地 loss/status/html 监控 |
| `gr00t/experiment/utils.py` | checkpoint 格式化、best metric callback |
| `gr00t/model/gr00t_n1d7/setup.py` | 模型/processor/dataset/collator 装配 |
| `gr00t/model/gr00t_n1d7/gr00t_n1d7.py` | 模型主干 + 动作头 + flow matching loss |
| `gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py` | 样本构造、归一化、图像增强、collator |
| `gr00t/model/modules/qwen3_backbone.py` | Qwen3-VL 骨干加载、截断、冻结、前向 |
| `gr00t/model/modules/dit.py` | DiT / AlternateVLDiT |
| `gr00t/model/modules/embodiment_conditioned_mlp.py` | 按 embodiment 选的 MLP |
| `gr00t/data/dataset/factory.py` | 建数据集、算统计量、混合权重 |
| `gr00t/data/dataset/lerobot_episode_loader.py` | 读 LeRobot episode |
| `gr00t/data/dataset/sharded_single_step_dataset.py` | 时间步切分、shard 打包 |
| `gr00t/data/dataset/sharded_mixture_dataset.py` | shard 调度、rank 分配、后台预取 |
| `gr00t/data/dataset/fast_sharded_dataset.py` | 流式/稀疏解码版本 |
| `gr00t/data/state_action/state_action_processor.py` | 归一化、绝对/相对动作转换 |
| `gr00t/data/stats.py` | 数据集统计量 |
| `gr00t/configs/deepspeed/zero2_config.json` | ZeRO-2 配置 |

---

## 17. 附：术语表

| 术语 | 含义 |
|---|---|
| VLA | Vision-Language-Action，看图 + 读文字 + 输出动作的模型 |
| embodiment | 机器人本体/形态；`NEW_EMBODIMENT` 表示新本体 |
| modality | 输入输出模态：video/state/action/language |
| shard | 一组约 128 个时间步样本的集合，是取数和随机化的单位 |
| micro-batch | 一次 forward/backward 的 batch；8 卡时全卡合计 8 |
| global batch | 梯度累积前的总 batch，当前为 8 |
| accumulated batch | 梯度累积后的有效 batch，当前为 32 |
| ZeRO-2 | DeepSpeed 的显存优化：切分梯度和优化器状态，不切分参数 |
| flow matching | 学一个从噪声指向干净动作的速度场，推理时积分 |
| DiT | Diffusion Transformer，动作头的去噪网络 |
| processor | 把原始样本变成模型输入（归一化/增强/tokenize） |
| collator | 把多个样本 pad/stack 成一个 batch |
| checkpoint | 模型权重 + 优化器/调度器/RNG + 训练状态的快照 |

---

## 18. 自检问题

如果能不看文档回答下面 8 个问题，说明结构已经清楚了。

1. `train.sh` 到 `launch_finetune.py` 之间，谁负责起 8 个进程？
2. `global_batch_size=8`、8 卡、累积 4 次，一次 optimizer step 用多少个样本？
3. `episode_sampling_rate=0.1` 到底做了什么？会不会丢数据？
4. 一个 shard 里存的是张量还是索引？什么时候真正解码视频？
5. 一个样本的 state/action 在进入模型前分别是什么形状？为什么是 132 维？
6. 哪部分参数在训练？backbone 的 12 层语言模型会更新吗？
7. flow matching 的 `t=0` 和 `t=1` 分别代表什么？模型预测的是什么？
8. 为什么 8 卡训练的 checkpoint 不能直接拿到 4 卡上续训？

<details>
<summary>参考答案</summary>

1. `torch.distributed.run`（torchrun）负责起 8 个进程，每个进程跑同一份
   `launch_finetune.py`，靠 `LOCAL_RANK` 区分。
2. 8 × 4 = 32 个样本。
3. 把每条 episode 的时间步交错切成 10 份再装 shard，所有时间步都会用到，
   不会丢掉 90% 的数据。
4. shard 里存 `(episode_index, step_indices)` 索引；`stock` loader 在
   `get_shard()` 时加载 episode 并解码视频，`fast` loader 只解码需要的帧。
5. state 是 `[1, 132]`（58 有效 + 74 padding），action 是 `[40, 132]`
   （40 步 × (58 + 74)）；132 是模型配置的 `max_state_dim/max_action_dim`。
6. 只有动作头训练：DiT、vl_self_attention、state/action encoder/decoder、
   vlln 等约 1.62B；backbone 的语言塔和视觉塔都冻结，不更新。
7. `t=0` 是纯噪声，`t=1` 是干净动作；模型预测速度场
   `v = action - noise`，推理时从噪声积分到动作。
8. ZeRO-2 的优化器状态按 rank 切分，8 卡有 8 份
   `bf16_zero_pp_rank_*_optim_states.pt`；卡数变了分片对不上。

</details>
# RealMan 原版 N1.7 无触觉基线

数据流：三路 RGB + 当前58维关节 + 任务文字 → 原版 GR00T N1.7 → 40×58 动作目标。
没有增加触觉编码器、触觉数值输入、额外控制器或验证集。

## 文件

- realman_config.py：四种模态及 NEW_EMBODIMENT 注册。
- action_semantics.json：从采集代码填写各组动作类型、状态/动作单位、训练表示。已按采集源码、厂商协议和数据比对填写：臂为degree，手为radian；采用ABSOLUTE，保留原始数值。参见COLLECTION_AUDIT.md。
- prepare_baseline.py：复制元数据到 runs/ball-placement-baseline/data，data/videos 用软链接共享原始文件；使用用户给定的取球放框指令。
- train.sh：prepare / smoke / train 三种入口。

源数据/视频不改写。训练副本的 info.json 不声明未使用的触觉字段，避免官方 generate_stats 对嵌套触觉数组报错。全50条演示保留；采样比例沿用官方0.1（每轮采样时间点，并非排除90%的演示）。

## 本地模型

为官方 FinetuneConfig 添加 backbone_model_path 和 transformers_local_files_only；官方启动器传递这两个设置。模型初始化支持从本地 config.json 判断 Qwen3-VL，并将本地骨干路径传给嵌套加载过程。原权重文件不修改，模型结构/损失不改。

## 运行顺序

1. `bash examples/realman/train.sh prepare` 只准备训练元数据。
2. 确认任务指令符合所有演示。动作配置已填写，真机部署协议仍需在采集机核对。
3. `bash examples/realman/train.sh smoke` 实际跑5个更新步并保存一次检查点。这是待执行检查，尚未宣称通过。
4. 短跑通过后 `bash examples/realman/train.sh train`。默认10000更新步、单卡前向batch2×累积16，有效batch32；这些是初始训练设置，并非TacForcing论文的复现实验参数，也不是最优参数结论。

所有训练输出和缓存指向/data。正式训练冻结视觉/语言骨干，沿用官方微调默认的投影/动作分支训练设置。不启动wandb网络日志。

## 已知待办

- 采集机外部发送端版本尚未与公开源逐字比对；真机部署前确认控制接口、平滑及关节顺序。
- 本地完整模型加载、GPU前向/反向、5步短跑和检查点保存待执行。
- 真机推理需使用同样的相机映射、单位、归一化及动作解码，不能把归一化输出直接发给关节。

## 任务语言入口

打开 `examples/realman/task_instruction.txt`，直接编辑任务描述并保存。准备脚本读取它，再写入训练视图的 `meta/tasks.parquet` 和 episode 元数据。`realman_config.py` 只指定语言字段映射。

`train.sh` 按指令文件的内容哈希创建 `runs/ball-placement-baseline/data-<哈希>`。修改指令后会自动创建新的元数据视图，不覆盖旧视图；视频和Parquet仍通过软链接共享。每次运行也会把指令文件复制到日志目录。

命令行 `prepare_baseline.py --task ...` 仍可显式覆盖默认文字。所有50条演示沿用同一指令；多任务数据应另外保留逐episode任务映射，不能用此单任务脚本统一覆盖。

## 单卡启动

默认自动选择没有计算进程、至少90%显存空闲且GPU利用率不超过10%的卡，优先可用显存最多的一张。用UUID设置CUDA_VISIBLE_DEVICES，确保Transformers只看到一张卡。如果没有合适的卡，在加载模型前退出。它是启动时的状态检查，不是GPU资源预约；共享服务器上后续仍可能有新进程进入。

已经显式设置单张CUDA_VISIBLE_DEVICES时保留该分配。空值或多卡列表会被拒绝，避免越过已有资源分配；需要自动选卡时先unset CUDA_VISIBLE_DEVICES。启动命令仍为 `bash examples/realman/train.sh smoke`。`prepare`模式不需要GPU。

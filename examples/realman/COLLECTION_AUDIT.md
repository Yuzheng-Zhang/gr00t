# 采集源码核对（2026-11-12）

用户确认：RealMan机械臂使用度，Sharpa灵巧手使用弧度。
原始58维顺序：左臂[0:7]、左手[7:29]、右臂[29:36]、右手[36:58]。
训练选择ABSOLUTE：预测记录的位置目标，按训练集统计量逐维归一化，不做单位换算或状态相减。

## 本地上传源码

采集包：/data/yuzheng/projects/gr00t/lerobot-realman-collection.tar.gz

- src/lerobot/scripts/lerobot_record_manual.py:95–137：动作来自teleop.get_action()，保存teleop_action_processor输出；passive_recording=true时采集循环不发送控制。
- src/lerobot/processor/factory.py：默认动作/观测处理器为IdentityProcessorStep。
- src/lerobot/teleoperators/rm_aloha_master/rm_aloha_master.py:173–187：串口关节位置整数除配置joint_scale=10000，没有差分。
- src/lerobot/common/realman/client.py:931–944：UDP反馈joint_position除joint_scale=1000。
- src/lerobot/teleoperators/double_rm_aloha_master/double_rm_aloha_master.py:162–205：按左臂、左手、右臂、右手拼接动作。
- src/lerobot/common/sharpa_manus.py:619–643：读取HandAction的左右Joint.position，原样取22个float。
- 同文件:1166–1185：手反馈优先调用get_joint_position_rad()；失败时回退get_states().angles，后者未在采集代码标注单位。数据量级与弧度一致。
- configs/hardware/realman_double_arm_realtime_push_config.yaml：三路RGB来自左臂wrist/top、右臂wrist。left_top仅由配置所属左臂产生名字。

## 外部协议旁证

- https://develop.realman-robotics.com/robot/json/udpConfig/ ：UDP关节位置精度0.001度，对应除1000。
- https://github.com/sharpa-robotics/sharpa-manus-sdk/blob/main/retargeting_alg_release_V4.0/retargeting_manus_demo_multiprocess.py#L639-L677 ：ZMQ发送filtered_angles原值；UDP硬件路径另做rad2deg，SDK硬件路径使用弧度。ZMQ附近旧注释称转换为degree，但实际代码未转换。
- https://github.com/sharpa-robotics/sharpa-manus-sdk/blob/main/retargeting_alg_release_V4.0/include/proto_hand/sender_demo.py#L86-L105 ：positions直接序列化。

公开main版本仅作旁证，尚未与采集机外部SDK版本逐字比较；不据此宣称真机部署适配已完成。

## 已执行验证

- 全4个Parquet、36299帧、58维state/action均为有限数。
- 同单位action与state的全量平均绝对差：左臂1.2269度、右臂0.4347度、左手0.00705弧度、右手0.01125弧度。它们是遥操作目标与实测反馈，不应完全相同；该比较不证明时序或所有关节语义完全正确。
- 临时训练元数据视图保留50条演示，数据/视频仅用软链接；原始数据未改。
- 官方generate_stats成功，避免扫描无触觉基线不使用的嵌套触觉字段。
- 使用训练模态配置成功加载第0和49条演示的首尾RGB帧，三路图像均480×640×3。
- GPU模型加载、前后向和5步短跑尚未执行。

## 真机接入

需要对照实际控制端恢复组内关节顺序，反归一化后分别按臂degree/手radian输出；若控制接口要求不同单位，只在对应组做转换。公开手部发送端区分原始目标和硬件平滑后的命令，部署也需匹配。以上训练脚本不会连接机器人。

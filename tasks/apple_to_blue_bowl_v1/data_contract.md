# 原始录制与 ACT 数据契约

## 原始数据

沿用 DEX 的 `dex-native-v2`：manifest.yaml、effective_config.yaml、events.jsonl、每源 MCAP 及原始 SampleObservation。保留原始 CDR、原始图像、源时间与主机观测时钟。训练副本与原始数据分开存放；不覆盖、重写或删除 MCAP。`dex-native-v2` 与 LeRobotDataset v3.0 是两套不同的版本号。

三路 RGB、四路实际关节状态和三路目标命令必须录制。头部 CameraFrame 是包装消息，像素位于 `image` 中；腕部消息直接是 sensor_msgs/Image。处理 encoding、step、行填充、BGR→RGB 与大小端信息后再转换。不要使用平台预览 JPEG 或按裸 byte 数组假设图像布局。

| 角色 | Topic | 类型 | 向量字段 |
| --- | --- | --- | --- |
| 全局 RGB | `/head/left/image_raw` | dex_collection_interfaces/CameraFrame | image |
| 左/右腕 RGB | `/wrist/{left,right}/color/image_raw` | sensor_msgs/Image | data |
| 左/右臂实际状态 | `/{left,right}_arm/joint_states` | sensor_msgs/JointState | position |
| 左/右手实际状态 | `/{left,right}_hand/state` | manus_wuji_interfaces/HandState | position |
| 双臂目标 | `/tianji/joint_command` | tianji_interfaces/JointCommand | positions |
| 左/右手目标 | `/{left,right}_hand/command` | manus_wuji_interfaces/HandCommand | position |

诊断、机器人状态、遥操作状态、相机参数、Tracker 与 MANUS 默认留存，首版不作为模型输入。深度、头部右眼、PICO 和头部运动流不默认选择。若需要增加，先重新检查吞吐与磁盘预算。

## 名称、单位和有效性

权威排列在 task.json：left_j1..7、right_j1..7，然后 l_ 和 r_ 的 THJ/FFJ/MFJ/RFJ/LFJ 各 0..3。每条消息按名称重排，拒绝缺失、重复、非有限值和错误侧。核对现场名称与 SDK 单位；训练副本使用 float32 弧度，不把归一化值写回原始数据。

state 使用实际位置；action 使用目标位置。手部命令在 Enable 过渡时还会经过 hold/ramp，因此这些片段排除在任务区间之外。目标命令被发布或录到，并不证明驱动接收并执行；结合机器人/手部使能状态、watchdog 和跟踪反馈检查。

HandState 的 header 是发布时间，不能用它代替 SDK 样本新鲜度。所有 online、position_valid 必须为 true，stale 必须为 false，age_ms 必须在预算内。保留 received_stamp、source_timestamp_us、source_clock、source_sequence；设备时钟 `device_unsynchronized` 不与 ROS 或主机 epoch 直接相减。真实 source 为 wuji_hand2，mock/dry-run/sim 不能进入实物示教集。

## 时间配对

以采集主机 SampleObservation 的 CLOCK_MONOTONIC 建立可复现的入站时间轴。该轴是接收观测，不等于曝光/执行时间；保留 CameraFrame 的捕获侧时间，测量图像链路与控制链路延迟后再决定最终配对。相同 ROS 域不自动同步不同主机的单调时钟。

30 Hz 的每个决策点选择当时已可用的最新 RGB 与反馈，限制年龄；禁止未来图像、跨长缺口插值或无限重复旧值。初始观测年龄预算 100 ms，三路图像间偏差、图像年龄和状态年龄的 p50/p95/max 都须出报告。高于一个 33.3 ms 帧周期的偏差应检查原因，不能仅靠降低导出 FPS 掩盖输入丢失。

目标动作按各执行组在决策点后控制区间内首个有效命令配对，保留实际命令时刻与偏差；初始偏差预算 50 ms，不能任意整体平移一帧。训练与在线采样必须采用一致的可用信息与控制时序。若根据实测延迟更改配对方法，应递增数据处理版本，并重新导出全部训练/验证数据。

只接受同一有效会话中的双臂命令；会话切换、过期 valid_until、HOLD、驱动异常和重新使能划分边界。手部命令没有 session_id，必须结合时间和硬件状态限制区间。cached initial_state 不是任务样本。每个 task segment 是原始 admission interval 内的独立区间，序列训练窗口不得跨 Episode 或跨无效区间。

这些是慢速抓放任务的初始数据质量预算，不是修改机器人安全阈值的理由；必须用五条试采数据验证，再扩到正式批次。

## LeRobot v3.0 副本

features：三路 `observation.images.{front,left_wrist,right_wrist}`、`observation.state` shape [54]、`action` shape [54]。统一 RGB 与 480×640，保持比例缩放并补边，保存补边/缩放配置。LeRobot 写入图像通常采用 HWC uint8，训练加载器输出 CHW tensor；用固定版本 API，而不是手工猜视频索引。

通过官方 LeRobotDataset.create/add_frame/save_episode/finalize 创建数据，保留任务文本和自动索引列。API、数据格式及训练代码固定到同一版本。sidecar 保存 raw ID→exported episode_index、原始 manifest hash、片段边界、重采样时间、每路匹配偏差、关节顺序、处理代码版本和输出统计。

成功/失败、grasp_hand、位置区域和审核人属于 Episode 标注。首版只用审核成功且逐样本质量合格的片段训练；原始失败数据继续留存。grasp_hand 不送入标准 ACT，不要求部署操作者选择手。验证集按实际抓取手分层，再检查位置区域覆盖，保持整 Episode 分离，统计量只由训练部分计算。

## 转换器后续验收

1. 读取一条真实 MCAP，原始消息与 SampleObservation 配对，不依赖不同流的消息序号相等。
2. 验证三个视角像素/颜色，手部逐关节有效性和全部关节映射。
3. 输出图像/反馈/目标同步回放和时差报告，人工检查闭手、抬起、释放。
4. 证明无效、过期、跨会话及跨 Episode 窗口被拒绝，并保存拒绝原因。
5. 用 LeRobot 正式加载器读回帧、索引与 action chunk 后，才启动 pilot 训练。

上述转换器验收需要真实样本；本次采集前检查不替代它。

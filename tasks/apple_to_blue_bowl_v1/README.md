# 红苹果放入蓝色碗：首批采集准备

本任务使用 Tianji/Marvin M6 双臂、Wuji Hand2 双手。机器人观察固定头部相机、两路腕部相机和双臂双手关节反馈，根据苹果位置选择左手或右手抓取，将苹果移到蓝色碗上方并释放。成功时苹果完全脱离手部并稳定留在碗中至少 2 秒，全程无人干预。

不规定左右手先后顺序，不指定每次必须使用哪只手。首版每次使用一只手完成抓放；手间交接属于后续任务。双臂双手均记录，并保留 54 维 state/action。`grasp_hand` 是事后标注，不是部署时人工输入，也不作为本版 ACT 的输入。

本目录准备了任务、采集与数据检查文件。它们不会自动连接、使能、回零、停止或移动机器人。`teleop.yaml` 只声明手部参与、头部不参与跟随，不会执行硬件 Enable。实机数据率、时钟延迟、关节单位和任务表现仍需在 Ubuntu 现场验证。

## 文件与职责

| 文件 | 用途 |
| --- | --- |
| `task.json` | 版本化任务契约、10 路必需流、54 维顺序、数据质量预算 |
| `site.example.json` | 现场实测模板，复制为被忽略的 `site.local.json` 后填写 |
| `collection.yaml` | DEX 采集 profile；三路 RGB、四路反馈、三路命令为 required |
| `teleop.yaml` | 已有 ROS launch 的参数 overlay：双手参与、头部固定 |
| `data_contract.md` | 时间、有效性、action、导出格式与后续转换器要求 |
| `train_pilot.json` | 数据转换完成后使用的单卡 ACT 小数据训练配置 |
| `validation.md` | 实际执行过的检查和未验证项 |

采集器与硬件继续由原有仓库提供；本仓库保存任务、工具、标注和训练配置。原始 MCAP 和生成的数据集放数据盘，避免提交到 Git。

## 1. 先完成桌面与现场参数

选一个红苹果和一个蓝色碗。记录真实材质、苹果直径、碗内径与深度；宽口碗有利于入门。标记桌面坐标原点与左右方向，并测量苹果可摆放范围及碗的位置。头部固定后，确认三路训练图像覆盖接近、抓取、释放与成功停留。

由现场操作员按现有机器人流程确定可到达的准备姿态。保存双臂各 7 维、双手各 20 维**实际**位置；确认名称映射和弧度单位。此工具不根据示例生成或发送准备姿态。左右手都能根据苹果位置参与任务，不要求所有位置都由两侧同时可达。

```bash
# 在 FEGkkd/lerobot 仓库根目录执行。
cp tasks/apple_to_blue_bowl_v1/site.example.json \
   tasks/apple_to_blue_bowl_v1/site.local.json
```

编辑 `site.local.json`：填入所有 null，核对相机实际尺寸/FPS，填写机器人现场 ROS_DOMAIN_ID、数据盘绝对路径与实际部署 commit。`verified` 中的布尔值只能在相应实测完成后改成 true。版本字段预填的是开发时读取的源码参考，必须改为现场实际使用版本。

现场数值未知时，模板会被校验器拒绝。这意味着需要完成测量，不是程序自动认定机器人可以运动。

```bash
# 离线工具仅需要 Python 3.12+ 和 PyYAML，不需要安装整套训练依赖。
uv run --no-project --python 3.12 --with pyyaml python \
  examples/tianji_wuji_collection/collection_tools.py validate \
  --site tasks/apple_to_blue_bowl_v1/site.local.json
```

## 2. 冻结本次会话配置

先计划 5 次尝试，成功不足 5 条时继续新增尝试并保留失败记录。正式采集目标 50–100 条有效成功示教，初始计划 80 条；建议左右手各有至少 25 条，按自然抓取结果统计，不通过强制换手凑数量。改变苹果摆放区域以覆盖两侧，碗初期维持在记录的位置。中间区域手选择可能有歧义，可少量覆盖并标注，后续按失败情况扩充。

```bash
uv run --no-project --python 3.12 --with pyyaml python \
  examples/tianji_wuji_collection/collection_tools.py session \
  --site tasks/apple_to_blue_bowl_v1/site.local.json \
  --session-name apple-pilot-001 --trials 5 \
  --out tasks/apple_to_blue_bowl_v1/runs/apple-pilot-001
```

该命令生成任务/现场配置快照、带绝对数据根目录的 `collection.generated.yaml`、平台元数据示例、空白 `trials.csv` 和磁盘预算。已有会话目录不会被覆盖。把这些文件随原始数据一起备份。

默认三路 RGB（1280×720 一路、720×720 两路，BGR8，30 Hz）仅图像约 176.256 MB/s。按每次最多 30 秒计算，5 次约 26.4 GB；工具加入 20% 余量和 5 GiB 保留空间，要求约 37.1 GB 可用空间。80 次约需 513 GB 可用空间。实际尺寸、额外选中的流和文件系统开销会改变用量；增加相机或深度后需重新估算。保留空间是数据质量策略，不是硬件安全参数。

## 3. 在采集机启动原有系统

Ubuntu 24.04 / ROS 2 Jazzy 上，先 source ROS、实际 teleop 工作区与平台自定义接口 overlay。不要混用不兼容版本的消息。

照原有方式分别启动相机、Tracker、MANUS、双臂与双手驱动。头部相机继续工作，头部电机和 PICO 跟随不参与本任务。协调节点可选用本任务 overlay：

```bash
# TASK_ROOT、PLATFORM_ROOT、TELEOP_WS 均填写本机绝对路径。
export TASK_ROOT=/absolute/path/to/lerobot/tasks/apple_to_blue_bowl_v1
export PLATFORM_ROOT=/absolute/path/to/dex_data_platform
export TELEOP_WS=/absolute/path/to/teleop_ws

ros2 launch whole_body_teleop whole_body_teleop.launch.py \
  config:="$TASK_ROOT/teleop.yaml"
```

已核对的 launch 会先加载其安装目录 `real.yaml`，再加载此 overlay；限位、滤波、超时、地址等保留现场配置。不要同时运行第二棵协调节点 launch。使用不同版本的 launch 时先确认仍采用此参数合并语义。

另一个终端启动平台。`collection.yaml` 是流 profile，不是 ROS 节点参数 YAML：

```bash
export DEX_MIN_FREE_BYTES=5368709120
bash "$PLATFORM_ROOT/scripts/start_platform.sh" --check \
  --config "$TASK_ROOT/runs/apple-pilot-001/collection.generated.yaml" \
  --teleop-workspace "$TELEOP_WS"

# 上一步通过后由操作员启动平台；必要时按原仓库说明加 --build。
bash "$PLATFORM_ROOT/scripts/start_platform.sh" \
  --config "$TASK_ROOT/runs/apple-pilot-001/collection.generated.yaml" \
  --teleop-workspace "$TELEOP_WS"
```

所有终端设置为 `site.local.json` 中同一个 ROS_DOMAIN_ID。平台只启动自己的采集器和 Web，不启动硬件。required 流会参与采集器准入；还应在 UI 检查实际选择，避免沿用升级前的保存选项。缺少头部右眼、深度或其他非必需流时可能产生可选流警告，不可把整体 COMPLETED 当作训练数据完整证明。

## 4. 运行只读预检

操作员通过原有界面完成 Connect/Home 后，先检查 connected 阶段。预检只订阅低维反馈、命令和采集器状态，查询发布者，不订阅原始 RGB，不创建任何硬件服务客户端。

```bash
# 必须先 source ROS、teleop 与平台接口 overlay。
# 此处特意用 ROS 对应的系统 Python，保持 rclpy 的 Python ABI 一致。
/usr/bin/python3 examples/tianji_wuji_collection/preflight.py \
  --site "$TASK_ROOT/site.local.json" --phase connected --trials 5 \
  --out "$TASK_ROOT/runs/apple-pilot-001/preflight-connected.json"
```

connected 阶段允许动作命令尚未开始发布，但要求对应发布者存在。它检查：实际加载的 required profile、唯一发布者、三路相机持续输入、双臂双手反馈、手部逐关节有效性/SDK 样本年龄、机器人连接回零状态、ROS 域和磁盘预算。

在操作员按原有流程 Start/Enable 并确认准备姿态稳定后，可用 `--phase active` 再检查一次，写到另一个报告文件。此时还要求新鲜且格式正确的命令、有效会话、双手 enabled 和机器人可接受命令。预检不会替你执行这些操作。

预检的 24 Hz 相机下限是 30 Hz 标称值的 80%；关节最低 30 Hz；500 ms 是现场连通性检查上限。它们不是允许离线填补长时间缺失的规则。导出要求更严格的逐样本质量预算，详见数据契约。预检通过不证明运动安全、曝光同步或 SDK 实际执行了目标命令。

## 5. 开始第一批示教

在数据平台填写 `session.json` 中的元数据：`task_name=apple_to_blue_bowl_v1`，实际操作员、物体 ID、会话名与 trial_index。每次尝试递增 trial_index；不要把训练所需的 task_id 只写在自由文本备注里。

摆放苹果与碗 → 准备姿态稳定 → 开始录制 → 根据位置自然选择一只手抓取 → 移到碗上方并释放 → 成功后保持至少 2 秒 → 结束录制 → 等待完成写入 → 记录 Episode ID 和结果 → 再人工复位。

每个有效任务片段最长 30 秒。失败或人工介入时保留原始数据并标注原因。录制启停与遥操作启停是两个独立操作；采用现场原有停止与急停流程。训练片段排除 Home、Enable 过渡、复位、持续 HOLD 和人工干预。

为每条完成写入的 Episode 创建标注模板：

```bash
uv run --no-project --python 3.12 --with pyyaml python \
  examples/tianji_wuji_collection/collection_tools.py annotation-template \
  --manifest /absolute/data/episode-id/manifest.yaml \
  --out "$TASK_ROOT/runs/apple-pilot-001/episode-id.annotation.json"
```

回放原始数据后填写：成功/失败/中断、实际抓取手、苹果与碗的桌面区域、有效任务片段起止、稳定停留时长、是否完全释放、是否人工介入、审核人。时间偏移相对于 manifest 的 `start_monotonic_ns`，不是视频播放器或 ROS 时钟的任意零点。

```bash
uv run --no-project --python 3.12 --with pyyaml python \
  examples/tianji_wuji_collection/collection_tools.py audit \
  --manifest /absolute/data/episode-id/manifest.yaml \
  --annotation "$TASK_ROOT/runs/apple-pilot-001/episode-id.annotation.json" \
  --out "$TASK_ROOT/runs/apple-pilot-001/episode-id.audit.json"
```

audit 检查 manifest 和人工标注，不读取/解码 CDR，也不自动证明像素、曝光时间或逐帧反馈有效。原始像素和位置仍需回放/后续转换器逐样本验证；标准 Image 没有源序号，零队列丢帧不代表上游无损。

## 6. 五条数据之后

先检查左右手与位置区域覆盖，再开发并用这五条真实数据验证 MCAP→LeRobot v3.0 转换器，执行逐样本时间/有效性检查和同步回放，最后启动 `train_pilot.json`。该训练配置不会触发硬件，也不会上传模型；数据集尚未生成时不能直接运行。

```bash
# 仅在 LeRobot 依赖已安装、数据已转换并验证后执行。
# 先编辑 train_pilot.json 中本机 dataset.root 和 output_dir。
uv run lerobot-train --config_path=tasks/apple_to_blue_bowl_v1/train_pilot.json
```

首轮 5 条可以全部用于拟合检查，配置中 eval_split=0；这不用于报告泛化效果。正式数据使用工具的 `split` 子命令生成左右手分层、整 Episode、固定种子的划分清单，转换时记录 raw Episode ID→LeRobot episode_index 映射。把清单交给训练/评估入口；不要直接用按时间顺序取最后 10% 的默认拆分。验证集中必须覆盖左右手，归一化统计只从训练部分计算。

本次交付范围是采集准备与数据检查。实际 MCAP 转换器、真实图像/时间验证、模型训练和策略执行桥在有真实样本后继续完成，没有将其标记为已验证。

# 验证记录

日期：2026-10-18（Asia/Shanghai）。

## 源码参考

- FEGkkd/lerobot main：146d8474932c8b566717eb2738ba459337ab71c4。
- FEGkkd/dex_data_platform codex/cpp-data-collection：d86faecffd11667b91081c5f5f82cb0512b2c93b。
- FEGkkd/whole-body-teleop codex/whole-body-sim-phase1：a7451ad51ef3b9733e90be177eaefb7eb922f6c2。

核对了 StartEpisode 元数据、CollectionStatus/StreamStatus、原始 manifest 格式、JointCommand、HandState、HandDiagnostics、Wuji 关节命名和 feedback_source，以及协调节点 launch 的 base YAML + overlay 合并方式。没有修改这些接口或原仓库运行状态。

## 已执行

- Python 3.12.14 下 18 项 unittest 全部通过。测试使用显式标为 synthetic 的内存数据和临时目录，不接触 ROS 或硬件。
- 正向对照：54 维顺序、左右手允许集合、有效现场配置、完整 manifest、有效人工标注、connected/active 数据快照。
- 反向对照：未知现场参数、非有限位置、错误/重复关节、无效/过期手部反馈、dry-run 数据、缺失流、丢样、未写完、错误任务/源模式、相机源序号缺口、低数据率、重复发布者、过期命令、无效机器人状态。
- 会话生成 CLI 写入快照、profile、元数据、试次日志；重复输出拒绝覆盖。
- 左右手分层划分具有固定种子、无交叉、无重复，拒绝缺少任一手的正式划分输入。
- Ruff 0.14.1 使用仓库 pyproject.toml 配置检查通过，format --check 通过。
- 两个 CLI 的 --help 和任务契约校验通过；未实测 site 模板按预期拒绝通过。
- collection.yaml 的全部 QoS、队列、预览设置和源话题继承参考配置；只改变选择、required、输出根目录示例与保留空间。

## 尚未执行

- ROS 2 Jazzy 上 live preflight 的消息反序列化、QoS、状态观察与计时（当前本地主机为 Windows，未启动 ROS 图）。
- 现场物体/容器测量、图像布局与颜色确认、真实 SDK 单位及准备姿态确认。
- 实机驱动/相机吞吐、曝光时间与接收时间偏差、写盘负载、机器人运动与停止表现。
- 真实 MCAP 的逐样本解码/转换、LeRobot 数据集加载、ACT 训练或部署。
- 新增 GitHub Actions workflow 的远端运行结果（以实际 Actions 结果为准）。

ROS 证据范围：接口/配置为 L0 静态核对，纯数据逻辑为 L1 离线测试；没有宣称 L3 ROS 图验证或 L4–L6 硬件验证。报告通过只代表注明范围内的数据检查，不是运动安全认证。

## 现场下一步

填写 site.local.json → 冻结会话 → 操作员启动既有系统 → connected/active 只读预检 → 先录 5 条 → 回放和标注 → 用真实样本验证转换器。现场发现与契约不符时，保存原因并修改对应配置版本，不用伪造实测值绕过检查。

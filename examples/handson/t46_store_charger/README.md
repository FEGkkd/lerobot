# HandsOn T46 → LeRobot v3 → GR00T N1.7

只处理 `bench320_fu006_hold_drawer_store_charger`。源数据固定为 [Cyan666/HandsOn-Datasets](https://huggingface.co/datasets/Cyan666/HandsOn-Datasets) revision `7ad38776d91e4e13c02dad4e6ad892e8b8a1d187`，复用仓库已有 v2.1→v3 转换器和 GR00T 策略。`VALIDATION.md` 是原提交的小样本历史记录，不能作为当前配置的 GPU 验证结果。

## 数据划分与动作语义

官方 T46 共300条：270 train、30 validation。本次隔离命令计数不完整的4条官方训练轨迹（8105、8189、8191、8218），因为计数无法证明缺失只发生在尾部；准备脚本直接拒绝不完整命令覆盖，不能擅自截取前缀。

2026-11-03 经用户更新要求，保留全部30条官方验证轨迹，并把官方训练部分的完整采集会话 `20260903-112559_dx_bench320_fu006_hold_drawer_store_charger` 留作测试集：

| 分区 | 轨迹 | 帧 | 用途 |
|---|---:|---:|---|
| train | 238 | 51,132 | 梯度更新、归一化统计 |
| validation | 30 | 6,319 | 选检查点、判断是否停止 |
| test | 28 | 5,914 | 最终一次独立评估，训练期间不用 |
| 隔离 | 4 | 795 | 不进入三个分区 |

采集会话和轨迹在三个分区间均不重叠，不按帧随机拆分。`selection.json` 保留官方身份；`source_provenance.json` 同时记录官方 `source_split` 和本次 `split`。训练统计只用238条 train，包含精确 q01/q99。测试集不参与统计、训练或检查点选择；闭环任务成功率需另测。

三路 RGB 为 front、wrist_left、wrist_right，30Hz。52D 状态/动作按右26、左26排列：每侧掌心世界系 XYZ 米3维、旋转向量弧度3维、手指关节弧度20维。动作是**世界系绝对控制目标**，`use_relative_actions=false`、`action_decode_transform=none`，不是旧 Tianji 的54D机械臂关节任务。16步块采样排除每条轨迹末尾15个不完整起点。

## 下载、转换与数据校验

Python 3.12 环境需要仓库 `groot,training` 依赖、ffmpeg、PyAV、pytest。数据放在代码仓库之外：

```bash
export PYTHONPATH=.:src
export DATA_ROOT=/absolute/path/to/handson_t46/datasets
python -m examples.handson.t46_store_charger.download_subset \
  --raw-root "$DATA_ROOT/raw" --all-t46 \
  --exclude-episodes 8105 8189 8191 8218 --download-workers 8
python -m examples.handson.t46_store_charger.prepare_dataset \
  --raw-root "$DATA_ROOT/raw" --output "$DATA_ROOT/v3" \
  --test-session 20260903-112559_dx_bench320_fu006_hold_drawer_store_charger
python -m examples.handson.t46_store_charger.validate_dataset \
  --raw-root "$DATA_ROOT/raw" --v3-root "$DATA_ROOT/v3"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q \
  --confcutdir=examples/handson/t46_store_charger/tests examples/handson/t46_store_charger/tests
```

只下载选中的296条 T46 Parquet/三路视频，绝不下载全部18,000条。逐文件核对两个官方 manifest 的 SHA256/大小；下载失败可重试，已有文件复用后重新核验。默认不加 `--all-t46` 时仍可下载最短1 train +1 validation作开发小样本，但不能作为正式集群数据。

转换只重编号 episode/frame/global index，保持数值与单位；`--force` 才会替换已生成的 v3 输出。完整数据校验检查原始/v3数值一致、视频全量解码、30Hz时间戳、52D形状、会话分区、训练统计、DataLoader、GR00T动作mask和归一化往返。它不加载完整模型，不能代替 GPU 训练验证。

## 单4090验证，再启动单H200

运行代码位于独立分支 `codex/handson-t46-cluster-validation`，基于用户指定提交 `863ec7d14e1d438f29c35cd7d3df319280fabc58`。新作业的数据、日志、检查点位于独立 `HANDSON_WORK_ROOT`，旧 Tianji 作业不重启、不覆盖。

复用已有的离线 GR00T/Cosmos 缓存及 Python 环境：

- `nvidia/GR00T-N1.7-3B`：`2fc962b973bccdd5d8ce4f67cc63b264d6886495`
- `nvidia/Cosmos-Reason2-2B`：`9ce19a195e423419c349abfc86fd07178b230561`

```bash
export HANDSON_WORK_ROOT=/inspire/path/to/personal/handson_t46
export HANDSON_RUNTIME_ROOT=/inspire/path/to/personal/tianji_wuji
export HANDSON_PYTHON=/absolute/path/to/prepared/environment/bin/python
export HANDSON_PROFILE=4090
export HANDSON_JOB_ID=handson-t46-4090-check-UNIQUE_TIMESTAMP
bash examples/handson/t46_store_charger/run_cluster.sh
```

4090 验证保持正式 batch4、100万步调度长度、100步 warmup，先实际更新1..3并保存，重新启动进程恢复到第4步，再重新加载保存模型/Processor进行验证轨迹动作预测。为快速验证执行链路，该短测每步仅用32个验证起点；它不消费测试集。检查有限 loss/梯度、Adam/scheduler计数、冻结报告、参数变化、TensorBoard和 `[1,16,52]` 输出后才产生 `verification.json`。

通过后才提交一个 H200 作业：设 `HANDSON_PROFILE=h200`、新的唯一 `HANDSON_JOB_ID`，以及 `HANDSON_VALIDATION_JOB` 指向通过的4090作业根。平台自动执行同一入口，不能另手动启动副本。入口核对代码提交/补丁、数据split/训练统计/完整校验报告hash、GPU型号和固定模型缓存；H200使用已校验的Ubuntu22.04离线系统运行包。不同代码或数据必须重新验证。

## 长训练、保存与停止

用户要求很高的训练上限并手动决定停止，当前上限 **1,000,000次更新**。保持部分微调、batch4、累积1、BF16、AdamW初始LR `5e-5`、100步warmup，随后cosine按100万步总长度调度；不设置loss自动达标阈值。每10步记录训练指标；正式H200每100步评估全部30条验证轨迹的有效16步起点（共5,869个），不再使用32个前缀起点。

每500步保存并结束阶段，下一阶段严格恢复模型、Adam、scheduler、RNG和采样进度；100万步总长度对应**7位检查点名**，例如 `train/checkpoints/0000500`。保存的检查点中，仅以全验证集loss选择 `best_checkpoint.json`，测试集始终保留。保留所有本作业检查点，不自动删除旧作业文件。

训练停止并通过技术核验后，在同一次作业的剩余预算允许时，对验证集选定的检查点执行一次最终测试：28条测试轨迹的5,494个完整动作块起点，样本加权loss写入 `test-evaluation.json`，不参与后续选模型。剩余时间不足时明确记录 `pending_runtime_budget`，不生成测试分数或启动新作业。

用户要求停止时，在当前唯一作业根写 `STOP_AFTER_PHASE`，当前500步阶段完成保存后停止。`progress.json` 是阶段进度，最终 `verification.json` 是技术核验，`stop.json` 单独说明停止原因；这些都不代表任务成功率达标。异常中断时保留最近完整检查点和原始日志，不伪造完整验收。

单个H200作业仍受原有720分钟平台上限约束；脚本11.5小时截止，并按阶段耗时预留下一阶段余量，共享盘低于60GiB停止。时间或空间用尽可以早于100万步停止；不自动追加预算、创建副本或提交下一作业。原始训练统计、事件文件和JSONL不重写。

## 后续闭环评测

训练结果先在 HandsOn 原始任务仿真中评测；[HandsOn任务映射](https://github.com/TsinghuaC3I/HandsOn/blob/main/handson/handson/tasks/assets/task_checkpoint_names.json) 定义任务身份。适配器在30Hz提供三路RGB与52D实测状态，使用保存的Processor输出世界系绝对目标，并按右26/左26交给控制器。该仿真适配器、测试集最终评估及真机映射均不能以训练loss替代。

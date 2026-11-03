# HandsOn T46 → LeRobot v3 → GR00T N1.7

此目录只处理 HandsOn `t46_store_charger`，其源任务为 `bench320_fu006_hold_drawer_store_charger`。示范指令原文是：

> Hold the drawer open with the left hand and place the charger fully inside with the right. Release the charger, then the drawer, and hold stable for half a second.

官方 [任务映射](https://github.com/TsinghuaC3I/HandsOn/blob/main/handson/handson/tasks/assets/task_checkpoint_names.json) 和 [数据集](https://huggingface.co/datasets/Cyan666/HandsOn-Datasets) 应同时用于后续闭环评测。数据集固定 revision `7ad38776d91e4e13c02dad4e6ad892e8b8a1d187`。本目录复用仓库现有 `lerobot.scripts.convert_dataset_v21_to_v30` 和 `groot` 策略，不训练新模型实现。

## 本机小样本复现

从仓库根目录运行，数据存放在仓库外：

```bash
python3.12 -m venv /home/ubuntu/lerobot_ws/.handson_env
/home/ubuntu/lerobot_ws/.handson_env/bin/pip install -e '.[groot,training]' pytest
source /home/ubuntu/lerobot_ws/.handson_env/bin/activate
export PYTHONPATH=.:src
export DATA_ROOT=/home/ubuntu/lerobot_ws/datasets/handson_t46
python -m examples.handson.t46_store_charger.download_subset --raw-root "$DATA_ROOT/raw" --train-count 1 --validation-count 1
python -m examples.handson.t46_store_charger.prepare_dataset --raw-root "$DATA_ROOT/raw" --output "$DATA_ROOT/v3"
python -m examples.handson.t46_store_charger.validate_dataset --raw-root "$DATA_ROOT/raw" --v3-root "$DATA_ROOT/v3"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q --confcutdir=examples/handson/t46_store_charger/tests examples/handson/t46_store_charger/tests
```

`download_subset.py` 使用 Hugging Face 官方 API，先取明确列出的元数据，再从 source manifest 与官方 split 选择 T46 轨迹，并且只逐文件下载所选 Parquet/三路视频；文件按两个官方 manifest 的 SHA-256 和大小检查。默认选择最短的 1 train + 1 validation；`selection.json` 记录原始 episode、split、路径、大小、hash。可重试下载。`prepare_dataset.py` 先产生只含所选 episode 的临时 v2.1 数据集，然后调用仓库转换器；原始 ID、source task、split、prompt、文件路径保存在 v3 的 `meta/source_provenance.json`。转换重编号 episode、frame、global index，数值与单位保持不变。输出已存在时需要明确指定 `--force` 才会替换。

本次所选原始 episode：train `8375`（128 帧）与 validation `8166`（156 帧）。完整 T46 manifest 有 270 train、30 validation。`meta/split_manifest.json` 的划分直接对应官方 split；训练统计 `meta/train_stats.json` 只取 train 轨迹，并计算精确 q01/q99。`LeRobotDataset` 使用相同的 manifest；test 分区为空，因为官方 T46 只有 train/validation。

T46 的 52 维按右手 26、左手 26 排列；每侧为掌心世界系 XYZ 米 3、旋转向量弧度 3、手指关节弧度 20。动作是世界系绝对控制目标。源数据没有逐帧 padding 或 active-side 列。完整 T46 中 4 条 train 轨迹（8105、8189、8191、8218）的命令计数短于总帧数；计数无法证明缺失位于尾部，因此转换器会拒绝这些轨迹。2026-11-03 经用户确认，本次训练隔离这 4 条，使用 **266 train + 30 validation**；官方验证集保持不变，不把隔离轨迹移入其他分区。隔离 ID 和原因写入 selection 与 source provenance。GR00T 的 16 步训练采样器再排除每条轨迹最后 15 个不足整块的起点；`action_is_pad` 仍在读取时产生，Processor 的动作 mask 也会屏蔽它。

`validate_dataset.py` 完整解码视频，检查 30Hz 时间戳、RGB 分辨率、52D 有限数值、episode 边界、源/v3 逐值一致、官方 split、训练 DataLoader 与 GR00T 数据打包、动作 mask、绝对动作归一化/反归一化及 256×256 图像预处理。它不会下载 GR00T/Cosmos 权重，因此不验证完整 VLM 前向或梯度更新。

## H200：全量 T46 准备与训练

在目标 GitHub 仓库创建的远端机器上，从新分支拉取代码，并准备 Python 3.12、ffmpeg 与单卡 H200：

```bash
git clone https://github.com/FEGkkd/lerobot.git
cd lerobot
git fetch origin codex/handson-t46-groot-n17
git switch codex/handson-t46-groot-n17
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[groot,training]' pytest
export PYTHONPATH=.:src
export HF_HOME=/path/to/large-disk/hf
export DATA_ROOT=/path/to/large-disk/handson_t46
export RUN_ROOT=/path/to/large-disk/runs/handson_t46
mkdir -p "$HF_HOME" "$DATA_ROOT" "$RUN_ROOT"
python -m examples.handson.t46_store_charger.download_subset --raw-root "$DATA_ROOT/raw" --all-t46 --exclude-episodes 8105 8189 8191 8218
python -m examples.handson.t46_store_charger.prepare_dataset --raw-root "$DATA_ROOT/raw" --output "$DATA_ROOT/v3"
python -m examples.handson.t46_store_charger.validate_dataset --raw-root "$DATA_ROOT/raw" --v3-root "$DATA_ROOT/v3"
```

`--all-t46` 仅选择 T46，绝不下载全部 18,000 条；上述隔离参数使本次选中 296 条。原始官方 train/validation 身份固定为 270/30，隔离后为 266/30，不重新随机划分。重新转换时显式加 `--force`。完整训练统计随转换重算，只用选中的 266 条 train。建议将 `selection.json`、`meta/source_provenance.json`、`meta/split_manifest.json`、`meta/train_stats.json` 和验证报告与作业日志归档。

在联网准备阶段预取并固定 GR00T 主模型和 Cosmos 视觉语言骨干；下列 revision 来自本仓库已有 H200 配置，若远端改用其他正式 checkpoint，应同步固定和记录其 SHA：

```bash
GROOT_PATH=$(hf download nvidia/GR00T-N1.7-3B --revision 2fc962b973bccdd5d8ce4f67cc63b264d6886495)
COSMOS_PATH=$(hf download nvidia/Cosmos-Reason2-2B --revision 9ce19a195e423419c349abfc86fd07178b230561)
test -f "$GROOT_PATH/config.json" && test -f "$COSMOS_PATH/config.json"
export HF_HUB_OFFLINE=1
```

本次执行顺序是单张 4090 验证真实 forward/backward、显存、优化器、checkpoint 保存与恢复，通过后才排单张 H200。短测应将 `log_freq` 设为 1，记录每次更新；保持正式配置的 batch 4、2000 步调度总长度及 100 步 warmup，使用 `stop_after_steps=3` 提前停止并保存，然后恢复到第 4 次更新。下列原始 `smoke_config.json` 是 batch 1、3 步独立调度的最小启动示例，不能代替上述正式配置验证。配置的 `base_model_path` 在命令行覆盖为已缓存本地路径：

```bash
lerobot-train --config_path=examples/handson/t46_store_charger/smoke_config.json \
  --dataset.root="$DATA_ROOT/v3" \
  --dataset.split_manifest="$DATA_ROOT/v3/meta/split_manifest.json" \
  --policy.base_model_path="$GROOT_PATH" \
  --output_dir="$RUN_ROOT/smoke" 2>&1 | tee "$RUN_ROOT/smoke.log"
```

确认 smoke 的 3 次更新、有限 loss、GR00T trainable 参数和 checkpoint 后，启动 2,000 步初始实验：

```bash
lerobot-train --config_path=examples/handson/t46_store_charger/train_config.json \
  --dataset.root="$DATA_ROOT/v3" \
  --dataset.split_manifest="$DATA_ROOT/v3/meta/split_manifest.json" \
  --policy.base_model_path="$GROOT_PATH" \
  --output_dir="$RUN_ROOT/formal" 2>&1 | tee "$RUN_ROOT/formal.log"
```

初始配置：三路 RGB、52D 状态/动作、16 步绝对动作、`new_embodiment`、BF16、LR `5e-5`、batch 4、仅训练 projector/diffusion/VLLN。batch 4 需以 H200 实测显存为准。训练日志在 `$RUN_ROOT/formal.log`，checkpoint 在 `$RUN_ROOT/formal/checkpoints/`，每 500 步保存。验证 loss 每 100 步计算；从保存的 500/1000/1500/2000 步里选最低验证 loss 对应的 checkpoint，并单独复制归档，例如：

```bash
grep -E 'step (500|1000|1500|2000): eval_loss=' "$RUN_ROOT/formal.log"
cp -a "$RUN_ROOT/formal/checkpoints/001000" "$RUN_ROOT/best_validation"
```

上述 `001000` 仅是示例，按实际最小验证 loss 替换。离线 loss 不能代表 HandsOn 闭环成功率。恢复训练请使用当前仓库的 checkpoint 恢复参数，并保持相同数据、split、统计及模型 revision；不要覆盖已有 checkpoint。

## 本次集群执行与验收

`run_cluster.sh` 使用独立的 `HANDSON_WORK_ROOT`，不会写旧 Tianji 数据或作业目录。运行前完成上述 296 条数据转换与完整数据校验，将 v3 放在 `$HANDSON_WORK_ROOT/datasets/v3`。`HANDSON_RUNTIME_ROOT` 只引用已有模型缓存与 `assets.json`；`HANDSON_PYTHON` 指向已有依赖环境，源码通过当前仓库的 `PYTHONPATH` 加载。

```bash
export HANDSON_WORK_ROOT=/inspire/path/to/personal/handson_t46
export HANDSON_RUNTIME_ROOT=/inspire/path/to/personal/tianji_wuji
export HANDSON_PYTHON=/absolute/path/to/prepared/environment/bin/python
export HANDSON_JOB_ID=handson-t46-4090-check-UNIQUE_TIMESTAMP
export HANDSON_PROFILE=4090
bash examples/handson/t46_store_charger/run_cluster.sh
```

4090 测试使用正式 batch 4、2000 步调度长度、100 步 warmup；第 3 步保存，重新启动进程严格恢复到第 4 步，并重新加载保存的 Processor/模型执行一次验证集动作预测。训练和验证逐步记录；每次 loss 验证仍按原配置抽取 32 个验证样本。验收检查有限指标、Adam/scheduler 计数、冻结参数报告、两次 checkpoint 的参数变化与输出 `[1,16,52]`。这些检查通过才产生 `verification.json`，不代表闭环任务成功。

通过后，由平台自动执行新的唯一 H200 作业：设 `HANDSON_PROFILE=h200`、新的 `HANDSON_JOB_ID`，并把 `HANDSON_VALIDATION_JOB` 指向通过的 4090 作业根目录。脚本会核对 GPU 型号、代码提交与工作区补丁、split/统计/完整数据验证报告 hash；不同则拒绝启动。正式训练保持 2000 步、每 500 步保存、每 100 步抽 32 样本验证。不要把抽样验证 loss 称为完整 30 条轨迹或闭环成功率。

## 后续闭环评测接口

HandsOn `t46_store_charger` 的仿真适配器应在每个 30Hz 控制时刻提供 front、wrist_left、wrist_right 三路 RGB 与右侧在前的 52D 当前状态，送入保存的 GR00T Processor；取解码且反归一化后的 52D 世界系绝对目标，按右 26/左 26 交给 HandsOn 控制器。保持 `use_relative_actions=false` 和 `action_decode_transform=none`，按 16 步 chunk 重规划；复位、终止条件和成功判定使用 HandsOn 原任务实现。该适配器及 Isaac Sim 部署不在本次范围，需在远端训练后单独实现并测试。

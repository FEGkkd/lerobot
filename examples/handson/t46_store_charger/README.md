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

T46 的 52 维按右手 26、左手 26 排列；每侧为掌心世界系 XYZ 米 3、旋转向量弧度 3、手指关节弧度 20。动作是世界系绝对控制目标。源数据没有逐帧 padding 或 active-side 列；manifest 的双手活动和 `pose_commanded_counts` 是转换依据。完整 T46 中 4 条 train 轨迹的命令计数短于总帧数，转换器保守地只保留计数范围内的前缀，并同步截短三路视频、重算统计。发布数据没有逐步命令有效性 mask，因此在远端全量训练前应复核这 4 条的命令缺失是否都发生在尾部（原始 ID：8105、8189、8191、8218）。GR00T 的 16 步训练采样器再排除每条轨迹最后 15 个不足整块的起点；`action_is_pad` 仍在读取时产生，Processor 的动作 mask 也会屏蔽它。

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
python -m examples.handson.t46_store_charger.download_subset --raw-root "$DATA_ROOT/raw" --all-t46
python -m examples.handson.t46_store_charger.prepare_dataset --raw-root "$DATA_ROOT/raw" --output "$DATA_ROOT/v3"
python -m examples.handson.t46_store_charger.validate_dataset --raw-root "$DATA_ROOT/raw" --v3-root "$DATA_ROOT/v3"
```

`--all-t46` 仅选择 T46 的 300 条，绝不下载全部 18,000 条。原始官方 train/validation 身份固定为 270/30，不重新随机划分。重新转换时显式加 `--force`。完整训练统计随转换重算，只用 train。建议将 `selection.json`、`meta/source_provenance.json`、`meta/split_manifest.json`、`meta/train_stats.json` 和验证报告与作业日志归档。

在联网准备阶段预取并固定 GR00T 主模型和 Cosmos 视觉语言骨干；下列 revision 来自本仓库已有 H200 配置，若远端改用其他正式 checkpoint，应同步固定和记录其 SHA：

```bash
GROOT_PATH=$(hf download nvidia/GR00T-N1.7-3B --revision 2fc962b973bccdd5d8ce4f67cc63b264d6886495)
COSMOS_PATH=$(hf download nvidia/Cosmos-Reason2-2B --revision 9ce19a195e423419c349abfc86fd07178b230561)
test -f "$GROOT_PATH/config.json" && test -f "$COSMOS_PATH/config.json"
export HF_HUB_OFFLINE=1
```

先在 H200 上执行最多 3 次参数更新，检查真实 forward/backward、显存、优化器和 checkpoint。配置的 `base_model_path` 在命令行覆盖为已缓存本地路径：

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

## 后续闭环评测接口

HandsOn `t46_store_charger` 的仿真适配器应在每个 30Hz 控制时刻提供 front、wrist_left、wrist_right 三路 RGB 与右侧在前的 52D 当前状态，送入保存的 GR00T Processor；取解码且反归一化后的 52D 世界系绝对目标，按右 26/左 26 交给 HandsOn 控制器。保持 `use_relative_actions=false` 和 `action_decode_transform=none`，按 16 步 chunk 重规划；复位、终止条件和成功判定使用 HandsOn 原任务实现。该适配器及 Isaac Sim 部署不在本次范围，需在远端训练后单独实现并测试。

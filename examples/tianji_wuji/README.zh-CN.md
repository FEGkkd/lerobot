# 天机双臂 + 无际双手：饮料瓶入袋（GR00T N1.7 / LeRobot）

## 任务和数据

本任务用真实双臂双手的 110 条成功演示，训练机器人把饮料瓶放入袋子。训练配置默认从 Hugging Face 的公开数据集 `fegkkd/bag_v3`（`v3.0` 标签）读取原始 LeRobot v3.0 数据；本机数据目录为：

`/home/ubuntu/MANUS_STEAMVR_ws/data_process/datasets/lerobot_bag_batter_head_stereo_640`

数据为 30 Hz、41,399 帧；导出记录中 110 条均为 `accepted`，丢帧数为 0。每一帧有四路**独立** RGB 视频、54 维关节状态和 54 维关节位置目标。54 维按左臂 7、右臂 7、左手 20、右手 20 排列。头部左右相机各为 640×640，腕部左右相机各为 640×480。原数据的任务名为 `bag_batter`。

## 改动

- `examples/tianji_wuji/train_config.json`：固定 GR00T N1.7 的任务定义、五项输入特征、54 维动作、16 帧动作块和训练参数。
- `src/lerobot/configs/default.py` 与 `src/lerobot/scripts/lerobot_train.py`：增加 `dataset.task_prompt_map` 和 `dataset.exclude_features`。加载后的批次把 `bag_batter` 映射为 `Put the drink bottle into the bag.`，并移除只在遥操作时存在的 `observation.teleop_input`。原始 parquet、视频、元数据不改动。
- `src/lerobot/policies/groot/processor_groot.py`：当相机分辨率不同时，先对较小画面居中补黑边，再按相机顺序组成模型输入，修复直接 `np.stack` 会报形状错误的问题。四路图像保持独立视角，没有横向拼接。同时合并基础权重的 embodiment 映射与 LeRobot 的新机器人默认映射，使 `new_embodiment` 使用 ID 10。
- `examples/tianji_wuji/validate_dataset.py`：检查 v3.0、110 条 episode、41,399 帧、任务名、四路图像、状态和动作维度，以及抽样视频解码和 16 帧动作窗口。

## 数据怎样进入模型

1. LeRobot 按 30 Hz 读当前帧四路视频和 `observation.state`，取从当前帧起的 16 个 `action`。一个动作块对应约 0.53 秒；每次动作均为绝对关节位置目标，故 `use_relative_actions=false`。
2. 训练批次把任务标签换成完整的英文指令，去掉 `observation.teleop_input`。训练配置只把四路 RGB 与 `observation.state` 列为策略输入，动作是监督目标。
3. GR00T 处理器按 `head_left, head_right, wrist_left, wrist_right` 顺序处理视频。两路腕部图像各在上下补 80 像素黑边，成为 640×640；头部图像保持 640×640。随后按原始 N1.7 权重的处理配置，用 OpenCV 缩至 256×256、按 `crop_fraction=0.95` 裁剪约 243×243、再缩至 256×256；训练时裁剪位置随机且四路视角共用同一相对位置，最后送入 Cosmos/Qwen3-VL 图像编码器。状态和动作按数据集统计量归一化，并填充到基础模型的最大维度 132；真实有效维度仍是 54。
4. 语言、四路视觉、当前关节状态一起用于预测动作。原始 N1.7 动作头的最大时间长度为 40；这份配置从数据中提供 16 步，处理器把其余位置补零并用有效步掩码排除损失，因此监督仅覆盖前 16 步。配置不做评估或推理，也不自动上传模型。

## 安装和验证

在仓库根目录执行（本机的 `uv` 位于 `/home/ubuntu/groot_ws/.tools/uv`）：

```bash
uv sync --locked --extra groot --extra training --extra test
.venv/bin/python examples/tianji_wuji/validate_dataset.py
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/pytest \
  tests/scripts/test_lerobot_train_batch_preprocessing.py \
  tests/policies/groot/test_groot_n1_7.py -q
```

远端默认会从 `fegkkd/bag_v3` 下载数据；若已把原始 v3.0 数据放到远端目录，可用 `--dataset.root` 直接读取该目录。此前上传的 `fegkkd/bag_batter` 是另一份衍生数据，不应替换成本配置的数据源。若模型权重已下载到本地，可用 `--policy.base_model_path` 覆盖默认的 `nvidia/GR00T-N1.7-3B`。任务标签需仍为 `bag_batter`，相机键和维度需保持一致。

## 开始训练

单卡 H200 的固定数据划分、100 次参数更新、恢复验证和日志配置见 [H200 验证任务](h200_smoke/README.zh-CN.md)。

```bash
bash examples/tianji_wuji/train.sh \
  --output_dir=/path/to/output
```

可先运行一个优化步骤作为启动检查：

```bash
bash examples/tianji_wuji/train.sh \
  --dataset.root=/path/to/lerobot_bag_batter_head_stereo_640 \
  --policy.base_model_path=/path/to/GR00T-N1.7-3B \
  --policy.dtype=bfloat16 \
  --steps=1 --batch_size=1 --accelerator.gradient_accumulation.steps=1 \
  --num_workers=0 --persistent_workers=false --save_checkpoint=false \
  --output_dir=/path/to/smoke-output
```

默认训练 30,000 个微批次、每卡批量 1、梯度累积 4，约对应 7,500 次参数更新。使用 GR00T 的优化器和余弦学习率预设，每 5,000 个微批次保存一次。有效 batch 为卡数 × 1 × 4；改变卡数会改变总训练样本量。默认冻结语言骨干和视觉塔，训练投影层、扩散动作头及 VL 层。显存及训练速度取决于远端 GPU；可通过同名命令行参数覆盖。模型加载可能需要访问 Hugging Face 的 GR00T N1.7 和 Cosmos-Reason2-2B 权重。

## 已完成的验证

- 原始数据检查通过：110 条 `accepted` episode、41,399 帧、无导出丢帧；全部引用的 parquet / 视频文件存在，状态和动作数值均为有限值。
- Hugging Face 上传验证通过：38 个数据文件、2.846 GiB，远端大小与 Xet SHA-256 均匹配；LeRobot 能通过 `fegkkd/bag_v3` 的 `v3.0` 标签直接读取 110 条 episode 元数据。
- LeRobot 数据读取器抽样解码通过：四路相机尺寸正确，状态为 54 维，动作窗口为 16×54。
- 相关单元测试：80 项通过。真实样本经 GR00T 完整预处理后，任务语言为 `put the drink bottle into the bag`，图像编码包含 4 个视角，embodiment ID 为 10，动作有效掩码覆盖 16×54，批次中无 `observation.teleop_input`。
- 本机 RTX 5090（32GB）用 `--policy.dtype=bfloat16` 覆盖后，真实数据的 1 步训练（前向、反向、AdamW 更新）成功结束。默认 FP32 参数在这张卡上会于 AdamW 状态初始化时显存不足；有更大显存的远端 GPU 可用默认值，32GB 级别显卡可使用 BF16 参数覆盖。

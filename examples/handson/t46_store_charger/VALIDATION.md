# T46 小样本验证记录

数据集：`Cyan666/HandsOn-Datasets`，revision `7ad38776d91e4e13c02dad4e6ad892e8b8a1d187`。源任务 `bench320_fu006_hold_drawer_store_charger` 共 300 条，官方 split 为 train 270 / validation 30。本次只下载并转换两条；数据与自动生成的 JSON 报告保存在 `/home/ubuntu/lerobot_ws/datasets/handson_t46/`，不进入 Git。

| 原始 episode | split | 帧数 | Parquet | Front | Left wrist | Right wrist |
|---:|---|---:|---:|---:|---:|---:|
| 8375 | train | 128 | 78,507 B | 419,944 B | 701,047 B | 905,736 B |
| 8166 | validation | 156 | 95,718 B | 614,246 B | 1,316,674 B | 1,488,081 B |

以上 8 个文件以及必需元数据逐项通过官方 `FILE_MANIFEST.jsonl` 的字节数和 SHA-256 校验；数据/视频同时匹配 `source_manifest.jsonl`。固定 revision 的数据集卡片 `README.md` 与仓库内文件 manifest 的旧哈希不一致，该卡片不参与自动下载或数据转换。

运行命令：

```bash
PYTHONPATH=.:src python -m examples.handson.t46_store_charger.download_subset
PYTHONPATH=.:src python -m examples.handson.t46_store_charger.prepare_dataset
PYTHONPATH=.:src python -m examples.handson.t46_store_charger.validate_dataset
```

输出 v3 路径：`/home/ubuntu/lerobot_ws/datasets/handson_t46/v3`。转换后共 284 帧，train 本地 episode 0、validation 本地 episode 1，分别对应原始 8375/8166。状态与动作均为有限 52D，右手 26 维在前；原始/v3 的全部数值和时间戳逐值相等，最大原始时间戳误差约 `2.23e-7 s`。Front 为 `480×854`，双腕为 `480×848`；原始六路视频和 v3 三路合并视频均完整解码，帧数与 30Hz 时间戳一致。

`LeRobotDataset` 重新读取成功。训练 sampler 从 128 帧中提供 113 个完整 16 步起点；episode 尾部 `action_is_pad` 正确出现，训练 batch 无 padding，也不会跨 episode。DataLoader batch 状态 `[2,52]`、动作 `[2,16,52]`、三路图像 `[2,3,480,854/848]`。GR00T N1.7 pack 产生 `[2,16,132]` 动作 mask，52D 有效维正确；绝对动作归一化/反归一化往返通过；图像处理输出 `[3,256,256]`。train-only 统计的 mean/q01/q99 与原始 train 帧独立重算一致。

本次两条轨迹的 `pose_commanded_counts` 都等于帧数。完整 T46 另有 4 条计数较短的 train 轨迹，转换器会保守截短前缀；由于源数据未提供逐步有效性 mask，这 4 条的缺失位置仍需在 H200 全量训练前核查。

`train_config.json` 与 `smoke_config.json` 均经当前 `TrainPipelineConfig`/draccus 解析为 `GrootConfig`。新增测试 9 项、既有 episode sampler 测试 20 项、既有 GR00T processor 回归测试 12 项通过。机器有 RTX 5090，但本地没有 GR00T/Cosmos 权重；未做完整 VLM Processor、真实模型 forward/backward、优化器更新或 checkpoint 保存。远端 H200 的权重预取、3 步 smoke 和 2,000 步实验命令见 [README.md](README.md)。

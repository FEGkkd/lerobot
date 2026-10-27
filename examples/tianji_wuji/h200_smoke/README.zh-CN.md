# 单卡 H200：100 次参数更新的部分微调验证

本目录基于 `FEGkkd/lerobot` 的 `codex/tianji-wuji-groot-n17-bag-batter` 分支，源提交为 `9c3dbbd3203ce0d1192ad9150e90f2901f29dd94`。用户已确认验证任务配置；尚未提交 H200 训练，也尚未运行 GPU 验证。

## 数据划分

固定数据为 `fegkkd/bag_v3@e82d64b44deff9dc4d42c149e6048010a5427ccd`。

| 分区 | 完整 episode | 帧数 | 本次用途 |
|---|---:|---:|---|
| train | 88 | 32,910 | 参数更新 |
| validation | 11 | 4,242 | 每 25 次更新，最多抽样 32 个窗口计算验证损失 |
| test | 11 | 4,247 | 保留，本任务不构造测试数据加载器 |

`splits/split_manifest.json` 保存全部 episode 列表、原始示教 ID、源文件哈希和统计量哈希。按 `sha256(42:source_episode_id)` 排序后划分，跨 Python 版本可复现，整条轨迹不会跨分区。当前元数据没有明确的采集场次/摆放组标记，因此这是 episode 级划分；不能据此声称验证了跨场次泛化。后续若补充这些分组，应在正式比较前另建版本并固定。

`splits/train_stats.json` 使用训练集全部 32,910 帧，重新计算状态/动作的 mean/std/min/max 和分位数。训练和验证处理器均使用同一份训练集统计，原始数据集不改动。相机沿用模型预处理及固定图像归一化。任务语言、四视角、54 维关节顺序、30 Hz 和绝对动作定义继承上层任务配置。

仅下载元数据和关节 Parquet 也可重现统计，不需要先下载视频：

```bash
uv run --no-sync python examples/tianji_wuji/h200_smoke/prepare_splits.py \
  --root /path/to/bag_v3
```

## 训练和恢复方式

- 单节点、单张 H200 141GB；初拟 20 CPU / 200GB 内存 / 64GB 共享内存。
- 每批 1，梯度累积 1；计划总计 100 步，明确对应 100 次参数更新。
- 冻结完整视觉语言骨干；训练投影、扩散动作头及 VL 连接模块。未启用 LoRA。
- FP32 参数，BF16 混合精度计算，AdamW、峰值学习率 `1e-4`、余弦调度。
- 第一进程 `steps=100, stop_after_steps=50`，在第 50 次更新保存并正常退出。
- 第二进程读取第 50 步的完整 checkpoint，恢复模型、优化器、调度器、随机数与采样进度，然后运行到 100。
- 两段始终保留同一条 100 步学习率计划；不会用 50 步计划重新启动后半段。
- 每步记录训练指标；25/50/75/100 步计算验证损失；50/100 步保存完整状态。
- 容器运行上限建议 120 分钟，完成即退出；不自动重试以免重复执行。文件在共享盘持久保存。

`verify_run.py` 要求训练日志恰好是 1..100，检查两次保存的步数、AdamW step、调度器状态、随机数文件，以及动作头权重发生变化、骨干样例权重保持不变。最后写入 `verification.json`，其中 `status=passed` 才代表本次验证通过。它验证恢复后可以继续更新；不承诺与不中断训练逐位相同，DataLoader worker 的预取/随机状态可能不同。

这 100 次更新用于检查工程链路和资源消耗，不能据此判断实机任务成功率或模型是否收敛。

## 联网准备和离线启动

H200 空间没有外网。先在可联网的 Linux 准备机上安装环境、下载完整数据及两个模型。使用这个分支的 `uv.lock`，不要套用原版 Isaac-GR00T 的旧依赖。当前锁定 Torch 2.11.0+cu128、Transformers 5.5.4、Accelerate 1.15.0；日志额外安装 TensorBoard 2.20.0。

共享盘目录已在联网 CPU 准备实例中创建。实际持久化挂载点包含账号目录 `jianghaoran-253208120274`，不可省略；其父级本身属于临时容器文件系统。环境和两份完整 checkpoint 建议预留至少 100GB：

```text
/inspire/hdd/project/roboticsystem2/jianghaoran-253208120274/fengenguang/tianji_wuji/
  code/lerobot/                  # 包含本次修改的代码
  datasets/bag_v3/               # 固定版本数据及视频
  hf/                           # 模型缓存
  assets.json                   # 下载版本与本地路径
  jobs/<JOB_ID>/                 # 每个任务独立目录
```

准备机上设置 `HF_HOME` 后，在同一环境中完成有 Cosmos 权限的 HF 登录，无需发送 Token 给他人：

```bash
export TIANJI_WORK_ROOT=/inspire/hdd/project/roboticsystem2/jianghaoran-253208120274/fengenguang/tianji_wuji
export HF_HOME="$TIANJI_WORK_ROOT/hf"
cd "$TIANJI_WORK_ROOT/code/lerobot"
uv sync --locked --extra groot --extra training --extra test
uv run --no-sync hf auth login
bash examples/tianji_wuji/h200_smoke/prepare_online.sh
```

模型下载会检查固定 SHA，并准备骨干加载器使用的离线缓存引用。若远端模型 main 改变，脚本会停止，要求复核版本。准备机需有 Python 3.12；保存依赖环境和系统库到项目镜像，或验证共享盘 `.venv` 在目标镜像内可用。不要把登录 Token 打进镜像。

`job_spec.json` 是平台填写清单。2026-10-27 已在分布式训练表单确认可选规格：开发区-H200-3号机房，`lcg-a91ad10b-415d-4abd-8170-828a2feae5d2`；单卡 quota `79fe954a-be92-4772-ac0b-94ad8a79b7bb`，20 核 CPU、200GB 内存、H200 141GB，1 点券/小时。节点数 1、普通优先级 4。检查时组内显示 8 张可用，仅为实时快照，没有预留资源。

官方基础镜像已确认可选：`docker.sii.shaipower.online/base/ngc-cuda:12.8.0-devel-ubuntu22.04`。这是基础镜像，不能直接宣称已含可训练环境；需要完成联网准备、核验共享盘 Python 3.12 环境与系统库，并保存项目镜像。CPU 准备实例为 `fengenguang-groot-prepare`（4 核 / 16GB、零 GPU、4 小时自动停止），不运行训练。TensorBoard 2.20.0 使用 `setuptools==80.9.0` 保留其依赖的 `pkg_resources`。

启动命令：

```bash
export TIANJI_WORK_ROOT=/inspire/hdd/project/roboticsystem2/jianghaoran-253208120274/fengenguang/tianji_wuji
export TIANJI_JOB_ID=tianji-h200-smoke100-001
cd "$TIANJI_WORK_ROOT/code/lerobot"
bash examples/tianji_wuji/h200_smoke/run_h200.sh
```

此命令应在用户确认后填入任务并提交。当前配置和脚本不会调用任何排队 API。

## 日志在哪里看

令 `JOB=$TIANJI_WORK_ROOT/jobs/$TIANJI_JOB_ID`。任务启动时也会打印这个路径。

| 文件或目录 | 内容 |
|---|---|
| `$JOB/logs/console.log` | 全流程终端日志，平台任务日志中也可查看 |
| `$JOB/logs/phase1.log` | 1..50 次参数更新 |
| `$JOB/logs/phase2-resume.log` | 从 checkpoint 恢复和 51..100 次更新 |
| `$JOB/train/logs/metrics.jsonl` | 按步追加的训练/验证指标；UTC 时间戳 |
| `$JOB/train/logs/tensorboard/` | 离线 TensorBoard 事件，恢复后继续同一组曲线 |
| `$JOB/train/logs/trainable_parameters.json` | 骨干与动作头的可训练/冻结参数数量 |
| `$JOB/train/checkpoints/000050/`、`000100/` | 权重、处理器及完整训练状态 |
| `$JOB/verification.json` | 最终验收结果；失败时不会写入 passed |
| `$JOB/environment.json`、`assets.json`、`split_manifest.json` | 软件、权重和数据版本 |

查看文本：

```bash
tail -f "$JOB/logs/console.log"
tail -f "$JOB/train/logs/metrics.jsonl"
cat "$JOB/verification.json"
```

在平台“分布式训练 → Tensorboard”中，将日志目录设为 `$JOB/train/logs/tensorboard`。本次尚未创建平台 Tensorboard 实例。也可在能挂载共享盘的环境中运行：

```bash
uv run --no-sync tensorboard --logdir "$JOB/train/logs/tensorboard" --host 127.0.0.1 --port 6006
```

通过平台端口转发或 SSH 隧道查看 6006，保持本地绑定。

Windows 本机也可以读取这些事件文件。通过平台项目文件下载完整的 `tensorboard` 目录，复制到本机日志目录后运行 TensorBoard；保留事件文件名，恢复训练新增的事件文件也一起同步。需要安装 `tensorboard==2.20.0` 和 `setuptools==80.9.0`。示例：

```powershell
python -m tensorboard.main --logdir C:\training-logs\tianji_wuji --host 127.0.0.1 --port 6006
```

打开 `http://127.0.0.1:6006`。本机服务只显示已经同步的事件；下载一次不会持续更新。如已建立 SSH 连接，可在共享盘准备机运行上面的远端 TensorBoard，再以 `ssh -N -L 6006:127.0.0.1:6006 <准备机SSH配置名>` 实时转发，具体连接名须在 SSH 配置完成后确定。当前尚未启动训练，因此没有真实训练曲线。

## 重点看什么

| 指标 | 本次判断方式 |
|---|---|
| `optimizer_updates` | 顺序从 1 到 100；恢复后从 51 继续 |
| `train/loss` | 数值有限，排查持续异常增大；短期波动正常 |
| `validation/loss` | 25/50/75/100 各一条，抽样检查；不能要求单调下降 |
| `train/grad_norm` | 有限且不是持续为零；这里记录裁剪前的范数，超过 1 不等于训练失败 |
| `train/lr` | 预热后约 `1e-4`，按同一计划下降；恢复不会重新预热 |
| `train/gpu_mem_gb` | PyTorch 分配显存峰值，平台显示的总占用可能更高；确认无 OOM |
| `train/samples_per_s`、`step_s` | 实测吞吐和耗时，用于估计正式任务长度 |
| `dataloading_s`、`preprocessing_s`、`update_s` | 区分视频/存储、预处理和 GPU 计算瓶颈 |
| 参数报告与 `verification.json` | backbone 的 trainable 必须为 0，动作头 trainable > 0，最终验收 passed |

NaN/Inf 指标会被写入 JSONL 并主动终止任务。GPU 分配不足、缺权重、数据哈希不符、恢复失败也会直接失败。验证集损失与之后的实机成功率应分别报告。

## 本地测试范围

CPU 测试覆盖划分互斥/完整性、训练统计哈希、拒绝错误版本与污染、工厂向训练/验证注入训练统计、日志跨进程追加、NaN 停止、恢复计数和权重检查。运行：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --no-sync pytest \
  examples/tianji_wuji/h200_smoke/tests \
  --confcutdir=examples/tianji_wuji/h200_smoke/tests -q
```

本地没有本项目 GPU 运行环境，完整 LeRobot typed-config/模型加载、TensorBoard 真实写入和 H200 恢复训练由待提交任务执行；本地 CPU 测试成功不表示 GPU 任务已经成功。

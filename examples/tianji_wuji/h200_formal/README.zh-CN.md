# Tianji/Wuji 单卡 H200 部分微调正式基线 v1

状态：**用户已确认单卡H200、batch4、30,000步、最长12小时，待同步代码并提交一次任务**。当前已完成的 H200100步任务保持结束状态。

## 已知证据与配置选择

2026-10-28 的真实 H200100步验证完成了50步保存、进程退出、恢复到100步，最终验收passed。
模型总参数3,144,016,000，冻结backbone1,523,500,032，训练action head1,620,515,968。
batch1下峰值PyTorch显存35.96GiB，排除首次加载和恢复后的首步，更新耗时中位数0.1914秒。
这证明单卡可做本方案的部分微调；**batch4尚未实测，不能将batch1的显存/速度直接当作batch4结果**。

正式基线采用下列预先固定的参数，确认后才使用GPU：

| 项目 | 配置 |
|---|---|
| 资源 | 1 H200141GB、20CPU、200GB内存、64GB共享内存、优先级4 |
| 单卡物理batch / 梯度累积 | 4 / 1，有效batch4 |
| 参数更新预算 | 30,000次；当前循环每次micro-batch推进steps，累积1保证steps等于真实更新次数 |
| 训练样本预算 | 120,000个窗口起点，约3.65遍32,910训练帧；窗口重叠，不是独立演示重复数 |
| 起始权重 | 原始固定revision的GR00T N1.7-3B，不接续100步验收权重 |
| 学习率 | AdamW峰值5e-5，betas=(0.9,0.999)，weight decay1e-5，梯度裁剪1.0 |
| 调度 | 1,500次更新warmup（5%），之后cosine衰减到0；总horizon始终30,000 |
| 精度与范围 | FP32参数、BF16计算；冻结视觉/语言backbone，训练projector、diffusion、vlln |
| 数据 | 固定88train/11validation/11test；训练统计仅来自train，test不参与选模 |
| 验证 | 每1,000次更新，遍历完整11条validation轨迹、4,242个窗口起点；max_eval_samples=0 |
| 保存 | 第1,000步完整保存；之后每5,000步完整保存，最终30,000步保存，共7份 |
| 最佳模型 | 从这7个已保存checkpoint中选validation loss最低者；不是每个验证点都保存模型 |
| 日志 | 每10次更新写loss、梯度、LR、显存、速度；TensorBoard/JSONL，W&B关闭 |
| 运行上限 | 12小时，无平台自动重试，不自动另申请GPU |

这是第一轮正式基线的预算，不承诺30,000步一定收敛或具备实机成功率。暂不引入新的增强、相对动作或LoRA，以保留已验收的数据/动作语义。
validation loss含flow-matching采样噪声；小幅单点差异不能等同于实机效果。

## 启动和断点策略

`run_h200.sh`首先检查共享盘至少有220GiB空闲（用户/项目配额另受平台限制），校验离线资源并生成绝对路径配置。
随后从原始权重训练到1,000步，生成完整checkpoint和`phase1-verification.json`，检查梯度、冻结策略、日志、优化器/调度器状态，并用实测中位耗时重估剩余时间。
通过后自动重启训练进程，从`001000`恢复到30,000；第1,000步不是新的训练horizon，学习率不会重置。
如果OOM、NaN/Inf、空间不足或验收失败，脚本停止并保留证据，不静默更改batch或另行提交任务。

不删除历史checkpoint，预计7份约179GB（参数、AdamW、RNG及配置），另有日志和少量元数据。
任务意外中断时，保留的完整checkpoint支持恢复；应先检查最新完整checkpoint并处理其后的未保存日志尾部，避免重复计步，再明确恢复。脚本默认拒绝复用已有任务目录，防止误覆盖。

## 时间和点券估算

batch1实测中位数0.1914秒；按batch4线性放大估计，30,000次更新纯训练约6.38小时。
完整验证每次4,242帧，共30次。按100步任务每32帧验证约2.7秒外推，完整验证开销约3小时，批处理并行可能减少这部分开销。
考虑批处理并行收益和数据/IO差异，暂以**6–10小时**作为总运行规划范围（含验证/保存/加载，不含排队），12小时作为停止上限。
batch4实际速度和显存尚未测量，因此这个区间是推算，不是性能保证。1,000步后用真实速度更新ETA。
平台已核实单卡1点券/小时，对应预计6–10点券，12小时上限约12点券；实际按平台计费。
不沿用100步任务的120分钟限制。

## 确认后的启动命令

```bash
export TIANJI_WORK_ROOT=/inspire/hdd/project/roboticsystem2/jianghaoran-253208120274/fengenguang/tianji_wuji
# 每次新任务必须唯一。默认脚本自动生成，平台提交时也可显式指定。
export TIANJI_JOB_ID=tianji-h200-formal-v1-YYYYMMDD-HHMMSS
cd "$TIANJI_WORK_ROOT/code/lerobot"
bash examples/tianji_wuji/h200_formal/run_h200.sh
```

输出为`$TIANJI_WORK_ROOT/jobs/$TIANJI_JOB_ID`：

- `train/logs/metrics.jsonl`及`train/logs/tensorboard/`：已配置的本机每分钟同步会自动发现。
- `train/checkpoints/{001000,005000,010000,015000,020000,025000,030000}`：完整模型和训练状态。
- `phase1-verification.json`：启动期验收与实测ETA。
- `verification.json`：最终日志连续性、checkpoint状态、冻结/更新样本权重验收及最佳checkpoint路径。
- `best_checkpoint.json`：最佳已保存checkpoint的路径和对应验证loss；只记录指针，不复制巨型权重。
- `logs/phase1.log`、`logs/phase2-resume.log`：启动和实际恢复证据。

本机TensorBoard为`http://127.0.0.1:6006/`，Runs选择正式任务。同步需电脑唤醒、Codex运行、集群登录有效。

## 配方参考

- [LeRobot GR00T官方训练文档](https://huggingface.co/docs/lerobot/groot)
- [NVIDIA自定义机器人微调文档（包含单GPU配置）](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/finetune_new_embodiment.md)

这里的batch4、30,000步、5e-5和验证保存频率是基于本数据规模与实测资源制定的基线选择，不是官方对本任务验证过的最优参数。

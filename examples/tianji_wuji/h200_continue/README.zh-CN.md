# Tianji/Wuji 继续训练到训练 loss 目标

用户授权继续单卡H200部分微调。总上限300000次更新，源任务已完成30000步。
本次平台申请最长12小时，无自动重试/重新排队；脚本预留保存时间，在预算不足以完成下一阶段时正常结束。

- 来源：`tianji-h200-formal-v1-20261028-143520/train/checkpoints/030000`。
- 保留模型、Adam动量、RNG和全局步数；独立任务目录，旧checkpoint/日志不修改。
- 学习率重新规划：从全局30000开始500步warmup到1e-5，然后恒定。
- batch4、累积1、冻结视觉/语言骨干、原episode划分和归一化保持不变。
- 每5000步保存完整checkpoint并重启恢复，检查最近1000更新的训练loss均值。
- 该均值<=0.001时停止；不是某单步偶然低于阈值。不会因验证loss持平擅自提前停止。
- 每1000步验证完整11个episode/4242帧；每10步记录loss/梯度/LR/显存/速度及TensorBoard。
- 达到时间/存储/300000步上限也停止，并明确报告目标是否达到。不会自动再申请一个GPU任务。
- 可在本任务根目录创建`STOP_AFTER_PHASE`，当前5000步阶段保存后正常停止。
- checkpoint全部保留；每阶段前要求至少60GiB剩余空间，不自动删除模型。

入口：`bash examples/tianji_wuji/h200_continue/run_h200.sh`，需设置`TIANJI_WORK_ROOT`和唯一`TIANJI_JOB_ID`。

`progress.json`记录当前阶段均值/验证指标/checkpoint；最终`verification.json`中的`status=passed`
仅表示本次续训过程验收通过。是否达到0.001必须看`target_reached`和`stop_reason`。
`best_checkpoint.json`只在新续训已保存模型中挑选最低验证loss；原020000继续保留作对照。

CPU测试包括真实Adam状态与调度器跨进程恢复，以及只读源checkpoint的链接准备；GPU结果以平台实际运行验收为准。

# ELEN90098 Reinforcement Learning for Engineering
# Combined Report: Workshops 2–5

> 合并报告目录草稿：按 WS2 → WS3 → WS4 → WS5 组织，保留原题对应关系，仅列章节标题。英文标题可直接用于英文报告，括号内为中文说明。页码待正文合并后自动生成。

## Abstract（摘要）

## 1. Introduction（引言）

- 1.1 Report Scope and Objectives（报告范围与目标）
- 1.2 From Tabular Control to Deep Reinforcement Learning（从表格型控制到深度强化学习）
- 1.3 Overview of Environments and Evaluation Protocols（环境与评估方法概览）

## 2. Workshop 2: Q-Learning and SARSA（Q-Learning 与 SARSA）

- 2.1 CliffWalking Environment: States, Actions and Rewards（环境定义；Task 1 Q1）
- 2.2 Q-Learning, SARSA(0), n-Step SARSA and SARSA(λ)（算法与初步训练；Task 1 Q2）
- 2.3 Experimental Settings and Evaluation Metrics（实验配置与评估指标）
- 2.4 Comparison of the Four Algorithms（四种算法比较；Task 2 Q1）
- 2.5 Effect of the Q-Learning Learning Rate（学习率的影响；Task 2 Q2）
- 2.6 Effect of the SARSA(λ) Trace Parameter（资格迹参数的影响；Task 2 Q3）
- 2.7 Findings and Limitations（主要发现与局限）

## 3. Workshop 3: Deep Q-Learning（深度 Q 学习）

- 3.1 CartPole Environment: States, Actions and Rewards（环境定义；Task 1 Q1）
- 3.2 DQN Implementation and Baseline Training（实现与基准训练；Task 1 Q2）
- 3.3 Experimental Settings and Evaluation Metrics（实验配置与评估指标）
- 3.4 Effect of the Learning Rate（学习率的影响；Task 2 Q1）
- 3.5 Effect of Exploration Schedules（探索策略的影响；Task 2 Q2）
- 3.6 Effects of Replay Capacity and Batch Size（经验池容量与批量大小的影响；Task 2 Q3）
- 3.7 Findings and Limitations（主要发现与局限）

## 4. Workshop 4: Policy Gradient（策略梯度）

- 4.1 InvertedPendulum Environment: States, Actions and Rewards（环境定义；Task 1 Q1）
- 4.2 Vanilla Policy Gradient, Reward-to-Go and Baseline（方法与基准训练；Task 1 Q2）
- 4.3 Experimental Settings and Evaluation Metrics（实验配置与评估指标）
- 4.4 Effect of the Learning Rate（学习率的影响；Task 2 Q1）
- 4.5 Effect of Steps per Batch and Gradient Variance（批量步数与梯度方差；Task 2 Q2）
- 4.6 Policy Gradient versus DQN on CartPole（CartPole 上的算法比较；Task 2 Q3）
- 4.7 Policy Gradient Control of HalfCheetah（HalfCheetah 控制与运动观察；Task 2 Q4）
- 4.8 Findings and Limitations（主要发现与局限）

## 5. Workshop 5: Actor–Critic and Soft Actor–Critic（Actor–Critic 与 SAC）

- 5.1 HalfCheetah Environment and Policy Representation（环境与策略表示；Task 1 Q1）
- 5.2 Actor–Critic with Generalized Advantage Estimation（AC/GAE 基准训练；Task 1 Q2）
- 5.3 Soft Actor–Critic Implementation and Baseline Training（SAC 实现与基准训练；Task 1 Q3）
- 5.4 Experimental Settings and Evaluation Metrics（实验配置与评估指标）
- 5.5 Effect of GAE λ on InvertedPendulum and HalfCheetah（GAE 参数的影响；Task 2 Q1）
- 5.6 Initial Entropy Temperature and Automatic Tuning（初始熵温度与自动调节；Task 2 Q2）
- 5.7 Actor and Critic Learning Rates（Actor 与 Critic 学习率；Task 2 Q3）
- 5.8 Replay Capacity, Batch Size and Learning Start（经验池、批量与学习起点；Task 2 Q4）
- 5.9 Findings and Limitations（主要发现与局限）

## 6. Discussion Across Workshops（跨 Workshop 综合讨论）

- 6.1 Value-Based, Policy-Based and Actor–Critic Methods（价值方法、策略方法与 Actor–Critic）
- 6.2 Sample Efficiency, Learning Stability and Exploration（样本效率、训练稳定性与探索）
- 6.3 Hyperparameter Sensitivity and Practical Trade-Offs（超参数敏感性与实际权衡）
- 6.4 Limits of Comparisons Across Environments and Evaluation Protocols（跨环境与评估方法的比较边界）

## 7. Conclusion（结论）

## References（参考文献）

## Appendices（附录）

- Appendix A. Hyperparameters and Run Configurations（完整超参数与运行配置）
- Appendix B. Supplementary Learning Curves and Diagnostics（补充学习曲线与诊断）
- Appendix C. Reproducibility Materials and Execution Instructions（复现材料与运行说明）

---

编写依据：本文件夹中的 [WS2](Workshop2_Final_Report_EN.pdf)、[WS3](Workshop3_Final_Report_EN.pdf)、[WS4](Workshop4_Final_Report_EN.pdf)、[WS5](Workshop5_Final_Report_EN.pdf) 四份现有报告。第 1、6、7 章为合并报告建议增加的统整章节，其余章节对应现有内容；本文件未展开正文或添加实验结果。

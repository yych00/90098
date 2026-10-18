"""Build concise bilingual final answers from frozen, audited WS3 results."""
from pathlib import Path
import csv
import hashlib
import json
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

WS3 = Path(__file__).resolve().parents[2]
SOURCE = WS3 / 'task2_workspace/corrected_v2/ws3'
OUT = WS3 / 'outputs/final_report_v3'
FIG = OUT / 'figures'
FIG.mkdir(parents=True, exist_ok=True)
ROWS = list(csv.DictReader((SOURCE / 'analysis/condition_summary.csv').open(encoding='utf-8-sig')))
CURVES = list(csv.DictReader((SOURCE / 'analysis/learning_curves.csv').open(encoding='utf-8-sig')))

def fmt(value):
    return f'{float(value):.1f}'

def mean_sd(row, key):
    return f"{fmt(row[key + '_mean'])} ± {fmt(row[key + '_sd'])}"

def confirmation(row):
    return f"{float(row['median_confirmation_step_among_reached']):,.0f} ({row['seeds_with_3evals_ge475']}/3)"

def table(question, zh=False):
    selected = [row for row in ROWS if row['question'] == question]
    if question == 'q1':
        names = ['0.0001', '0.001', '0.01']
    elif question == 'q2':
        names = ['固定 0.1', '每回合乘 0.9，最低 0.02', '每回合乘 0.995，最低 0.02'] if zh else ['Constant 0.1', 'Decay 0.9, floor 0.02', 'Decay 0.995, floor 0.02']
    else:
        selected.sort(key=lambda r: (int(r['condition'].split(',')[0].split('=')[1]), int(r['condition'].split(',')[1].split('=')[1])))
        names = [r['condition'].replace('memory=', '').replace(', batch=', ' / ') for r in selected]
    heading = ['设置', '最终回报', '最后三次评估均值', '达标步数（种子数）'] if zh else ['Setting', 'Final return', 'Last-three return', 'Steps to threshold (seeds)']
    result = ['| ' + ' | '.join(heading) + ' |', '|---|---:|---:|---:|']
    for name, row in zip(names, selected):
        result.append('| ' + ' | '.join([name, mean_sd(row, 'final_return'), mean_sd(row, 'last3_mean'), confirmation(row)]) + ' |')
    return '\n'.join(result)

for name in ['q1_evaluation_returns.png', 'q2_evaluation_returns.png', 'q3_evaluation_returns.png']:
    shutil.copy2(SOURCE / 'analysis' / name, FIG / name)

plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
for q in ('q1', 'q2', 'q3'):
    if q == 'q3':
        fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.25), sharey=True, constrained_layout=True)
        groups = [([r for r in ROWS if r['question']==q and f'batch={batch}' in r['condition']], ax)
                  for batch, ax in zip((32,128),axes)]
    else:
        fig, ax = plt.subplots(figsize=(8.6, 3.05), constrained_layout=True)
        groups = [([r for r in ROWS if r['question']==q], ax)]
    for selected, ax in groups:
        for i,row in enumerate(selected):
            curve=[r for r in CURVES if r['question']==q and r['condition']==row['condition']]
            xx=[float(r['env_steps']) for r in curve]
            yy=[float(r['mean_return']) for r in curve]
            ss=[float(r['training_seed_sd']) for r in curve]
            label=row['condition']
            if q=='q3': label='Capacity '+label.split(',')[0].split('=')[1]
            elif q=='q2': label=['Constant 0.1','Decay 0.9','Decay 0.995'][i]
            ax.plot(xx,yy,label=label,linewidth=1.4)
            ax.fill_between(xx,[max(0,a-b) for a,b in zip(yy,ss)],[min(500,a+b) for a,b in zip(yy,ss)],alpha=.10)
        ax.set(xlabel='Training environment steps',ylim=(0,520),xlim=(0,200000))
        ax.grid(alpha=.2)
        ax.ticklabel_format(axis='x',style='sci',scilimits=(5,5))
        ax.legend(loc='lower right',fontsize=8.5)
    if q=='q3':
        for ax,batch in zip(axes,(32,128)): ax.set_title(f'Batch size {batch}',fontsize=10)
        axes[0].set_ylabel('Evaluation return')
    else: ax.set_ylabel('Evaluation return')
    fig.savefig(FIG/f'{q}_evaluation_returns.png',dpi=240)
    plt.close(fig)
data = [r for r in CURVES if r['question'] == 'q1' and r['condition'] == 'lr=0.001']
x = [float(r['env_steps']) for r in data]
y = [float(r['mean_return']) for r in data]
sd = [float(r['training_seed_sd']) for r in data]
fig, ax = plt.subplots(figsize=(8.2, 3.6), constrained_layout=True)
ax.plot(x, y, color='#27648f', label='DQN, learning rate 0.001')
ax.fill_between(x, [max(0,a-b) for a,b in zip(y,sd)], [min(500,a+b) for a,b in zip(y,sd)], color='#27648f', alpha=.17)
ax.axhline(500, color='#78858f', linestyle='--', linewidth=.8, label='Maximum return: 500')
ax.set(xlabel='Training environment steps', ylabel='Evaluation return', ylim=(0,520), xlim=(0,200000))
ax.ticklabel_format(axis='x', style='plain')
ax.grid(alpha=.2)
ax.legend(loc='lower right', fontsize=9)
fig.savefig(FIG / 'task1_dqn_evaluation.png', dpi=210)
plt.close(fig)

formula = r"y = r + \gamma(1-d)\max_{a'} Q_{\mathrm{target}}(s', a')"
plt.rcParams['mathtext.fontset'] = 'stix'
fig = plt.figure(figsize=(.1,.1))
fig.text(0, 0, '$' + formula + '$', fontsize=13)
formula_path = FIG / 'dqn_target_formula.png'
fig.savefig(formula_path, dpi=600, transparent=True, bbox_inches='tight', pad_inches=.035)
plt.close(fig)
with Image.open(formula_path) as image:
    image.verify()

en = '''# Workshop 3: Deep Q-Learning - Final Report

Authors: Zhengnan Cao, Yuchen Yang

This report answers Task 1 and all three Task 2 questions using the audited experiment set. Each condition uses three training seeds (0, 1, 2), 200,000 training steps, and 10 greedy evaluation episodes per checkpoint. Evaluation occurs every 10,000 training steps using reset seeds 10000-10009. Evaluation steps are excluded from the horizontal axis. Curves show the mean and one sample standard deviation across training seeds, without smoothing.

## Task 1 - Question 1: CartPole environment

### Question

What are the state space, action space and reward structure of CartPole? Are they finite or infinite, discrete or continuous? Is the environment deterministic or stochastic?

### Answer

| Item | Answer |
|---|---|
| State | Four real-valued variables: cart position, cart velocity, pole angle and pole angular velocity. The state space is continuous and infinite. |
| Action | Two actions: 0 pushes left and 1 pushes right. The action space is discrete and finite. |
| Reward | +1 per step, including the first failure step. Immediate reward is discrete and finite. CartPole-v1 allows at most 500 steps, so the maximum episode return is 500. |
| Dynamics | Deterministic given a state and action. Reset adds random initial-state perturbations, so episodes can still differ. |

The episode terminates when cart position leaves approximately ±2.4 m or pole angle leaves approximately ±12 degrees. The 500-step time limit truncates an otherwise continuing episode.

## Task 1 - Question 2: DQN training

### Question

Train DQN on CartPole-v1 and check whether it can achieve the maximum return of 500. Use TensorBoard, submit evaluation returns versus environment steps, and briefly comment on training success.

### Submit

The required evaluation curve is shown below. This uses the learning-rate 0.001 runs from the corrected experiment set, rather than the historical Task 1 run.

Configuration: three hidden layers of 64 units, learning rate 0.001, discount 0.99, replay capacity 100,000, batch size 64, and target update every 500 gradient updates. Learning starts after 1,000 steps. Epsilon starts at 1, decays by 0.995 after each episode and has a minimum of 0.02.

The DQN target is:

![Formula: DQN target](figures/dqn_target_formula.png)

Here d is 1 for true environment termination and 0 otherwise; a time-limit truncation still permits bootstrapping.

![Figure 1. DQN evaluation returns versus training environment steps.](figures/task1_dqn_evaluation.png)

### Discussion

**Does the curve show successful training?**

All three seeds finish with a 10-episode evaluation mean of 500, so the agent achieves the required maximum. However, the average of the last three checkpoints is 351.9 ± 82.1, showing earlier late-stage drops; this is successful maximum-return performance, not stable convergence throughout training.

TensorBoard scalar exports of these saved evaluations are provided in the separate verification material. The submitted curve uses the same evaluation records.

## Task 2 - Question 1: Learning rate

### Question

Investigate how learning rate affects training by comparing at least three values. Here the rates are 0.0001, 0.001 and 0.01, with all other settings fixed as above.

### Submit

Submit evaluation returns versus environment steps for each learning rate.

![Figure 2. Learning-rate comparison.](figures/q1_evaluation_returns.png)

{q1}

For this and the following tables, returns are mean ± sample SD across three training seeds. Last-three return averages the final three checkpoints within each seed. A threshold is confirmed at the third consecutive evaluation mean of at least 475; the step is the median among successful seeds, with the successful-seed count shown. This measures when near-maximum performance is first confirmed, not permanent convergence.

### Discussion

**Which learning rate achieves faster convergence?**

Learning rate 0.001 confirms the threshold fastest across all three seeds: median 60,000 steps, compared with 150,000 for 0.0001. Rate 0.01 confirms in only one of three seeds.

**Which learning rate is more stable?**

Among the two useful rates, 0.0001 has smaller late-training fluctuations: mean temporal SD over the last five checkpoints is 95.0, versus 180.4 for 0.001. Its final return is lower, so stability and endpoint performance are different criteria.

**Does too high a learning rate cause divergence or unstable oscillations?**

At 0.01 the curves fluctuate and final return is only 88.7 ± 4.5. This indicates poor and unstable learning here; finite losses do not establish numerical divergence.

**Does too low a learning rate cause slow learning or poor final performance?**

Rate 0.0001 takes longer to confirm near-maximum performance and ends below 0.001. It still learns effectively and is more stable late in training, so a smaller rate is not simply ineffective.

## Task 2 - Question 2: Exploration schedules

### Question

Compare constant epsilon 0.1 with two episode-based decays: multiply epsilon by 0.9 or 0.995 after each completed episode, starting from 1 and stopping at a minimum of 0.02.

### Submit

Submit evaluation returns versus environment steps for all three schedules.

![Figure 3. Exploration-schedule comparison.](figures/q2_evaluation_returns.png)

{q2}

### Discussion

**How do the schedules affect exploration and learning?**

Constant epsilon maintains a 10% random-action probability. The 0.9 schedule reaches its floor at 399-470 steps; the 0.995 schedule reaches it at 114,641-143,081 steps. The fast schedule therefore becomes mostly greedy before the 1,000-step learning start, yet gives the best last-three return here.

**Does keeping epsilon higher for longer improve performance or unnecessarily slow learning?**

Longer exploration does not improve late performance in these runs: the last-three means are 500.0 for fast decay, 351.9 for slow decay and 359.1 for constant epsilon. Slow decay confirms the near-maximum threshold earlier, however, so this is not a universal speed ranking.

**What happens if epsilon is too small or decays too quickly?**

In general, too little exploration can miss useful actions. That failure is not demonstrated here: fast decay performs well on CartPole; state-space coverage was not measured, so the curves alone cannot explain why.

## Task 2 - Question 3: Replay capacity and batch size

### Question

Compare replay capacities 1,000, 10,000 and 100,000, and batch sizes 32 and 128. All six combinations use the same other settings.

### Submit

Submit evaluation returns versus environment steps for all cases. In the table, the setting is replay capacity / batch size.

![Figure 4. Replay-capacity and batch-size comparison.](figures/q3_evaluation_returns.png)

{q3}

### Discussion

**How does memory size change curve stability and smoothness?**

The effect depends on batch size. With batch 32, capacity 1,000 is steadier late in training than 100,000 (temporal SD 0.0 versus 185.7); with batch 128, increasing capacity reduces late temporal SD from 166.2 to 113.7.

**How does batch size affect convergence speed and final performance?**

Batch 128 does not always improve performance. At capacity 1,000 it lowers last-three return from 500.0 to 409.8; at 10,000 it raises it from 349.1 to 416.5 and confirms the threshold 10,000 steps earlier; at 100,000 it raises return but confirms 60,000 steps later.

**Which combination gives the best CartPole performance in these experiments?**

Capacity 1,000 with batch 32 is best by last-three return: 500.0 ± 0.0. This is the clearest late-performance result within the tested budget, not a claim that it is best for every task or performance measure.

## Reference and evidence

ELEN90098, Workshop 3: Deep Q Learning, provided handout. All answers use the audited DQN experiment set; no new formal training was run for this final report. Three training seeds and fixed evaluation seeds limit generalization beyond these experiments.
'''

zh = '''# Workshop 3：深度 Q 学习最终报告

作者：Zhengnan Cao、Yuchen Yang

本报告合并 Task 1 与 Task 2 全部三道题，统一采用已审核的实验结果。每个设置使用三个训练种子（0、1、2），训练 200,000 步；每 10,000 个训练步进行一次贪心评估，每次 10 回合，评估初始种子为 10000-10009。横轴不包含评估步数。曲线为三个训练种子的均值，阴影为一倍样本标准差，不平滑。

## Task 1 - Question 1：CartPole 环境

### Question / 题目

CartPole 的状态空间、动作空间和奖励结构是什么？它们是有限还是无限、离散还是连续？环境是确定性的还是随机的？

### Answer / 解答

| 项目 | 解答 |
|---|---|
| 状态 | 小车位置、小车速度、杆角度和杆角速度，共四个实数变量。状态空间连续、无限。 |
| 动作 | 0 表示向左推，1 表示向右推。动作空间离散、有限，共两个动作。 |
| 奖励 | 每步 +1，包括首次失败的终止步。即时奖励离散、有限；CartPole-v1 最多 500 步，因此每回合最大回报为 500。 |
| 确定性 | 给定状态和动作后，物理转移是确定的；重置时的初始状态带随机扰动，因此不同回合的回报仍可能不同。 |

小车位置超出约 ±2.4 米或杆角度超出约 ±12 度时终止；达到 500 步时间上限属于截断。

## Task 1 - Question 2：DQN 训练

### Question / 题目

在 CartPole-v1 上训练 DQN，使其达到最高回报 500。使用 TensorBoard，提交评估回报随环境步数变化的曲线，并简要说明训练是否成功。

### Submit / 提交结果

下图为所需评估曲线，采用本次修正版实验中学习率 0.001 的三个种子结果，不再使用旧 Task 1 实验。

参数：三层、每层 64 个隐藏单元，学习率 0.001，折扣系数 0.99，Replay 容量 100,000，batch size 64，每 500 次梯度更新同步目标网络。先采集 1,000 步再学习；epsilon 从 1 开始，每回合乘 0.995，最低为 0.02。

DQN 的目标值为：

![公式：DQN 目标值](figures/dqn_target_formula.png)

其中 d 在环境真正终止时为 1，否则为 0；时间上限截断仍允许自举。

![图 1：DQN 评估回报随训练环境步数的变化。](figures/task1_dqn_evaluation.png)

### Discussion / 讨论与解答

**学习曲线是否说明训练成功？**

三个种子最终的 10 回合评估均值都达到 500，满足最高回报要求。但最后三次评估的平均值为 351.9 ± 82.1，说明训练后期仍出现过下降，不能称为全程稳定收敛。

这些保存的评估结果另有 TensorBoard 标量导出，放在独立验收材料中；提交曲线使用相同的评估记录。

## Task 2 - Question 1：学习率

### Question / 题目

比较至少三种学习率，研究学习率对训练的影响。本实验使用 0.0001、0.001、0.01，其他设置与前述配置相同。

### Submit / 提交结果

提交各学习率的评估回报随环境步数变化的曲线。

![图 2：学习率比较。](figures/q1_evaluation_returns.png)

{q1}

本题及后续表格的回报均为三个训练种子的均值 ± 样本标准差。“最后三次评估均值”先在每个种子内平均最后三个检查点。“达标”指连续三次评估均值不低于 475，在第三次确认；表中给出成功种子的步数中位数及成功种子数。这不表示此后永久保持稳定。

### Discussion / 讨论与解答

**哪个学习率收敛更快？**

0.001 在三个种子中都达标，确认步数中位数为 60,000，快于 0.0001 的 150,000。0.01 只有一个种子达标。

**哪个学习率更稳定？**

在两个有效学习率中，0.0001 的后期波动更小：最后五个检查点的时间标准差均值为 95.0，0.001 为 180.4。不过它的最终回报较低，稳定性和最后一次表现应分开判断。

**学习率过大是否导致发散或不稳定振荡？**

0.01 的曲线波动且最终回报只有 88.7 ± 4.5，说明本实验中表现较差且不稳定。损失保持有限，因此不能据此声称发生了数值发散。

**学习率过小是否导致学习缓慢或最终表现差？**

0.0001 达标较慢，最终回报低于 0.001，但仍学到了有效策略，且后期波动较小。因此不能简单认为较小学习率无效。

## Task 2 - Question 2：探索策略

### Question / 题目

比较固定 epsilon=0.1 与两种按回合衰减的策略：epsilon 从 1 开始，每个完成的回合后分别乘 0.9 或 0.995，最低为 0.02。

### Submit / 提交结果

提交三种探索策略的评估回报随环境步数变化的曲线。

![图 3：探索策略比较。](figures/q2_evaluation_returns.png)

{q2}

### Discussion / 讨论与解答

**不同探索策略如何影响探索与学习？**

固定 epsilon 保持 10% 的随机动作概率；0.9 衰减在 399-470 步达到下限，0.995 衰减在 114,641-143,081 步达到下限。快速衰减在 1,000 步开始学习前已基本转为贪心，但在本实验中获得了最高的最后三次评估均值。

**较长时间保持较高 epsilon 能否提高表现，还是会不必要地拖慢学习？**

本次较长探索没有提高后期表现：快速衰减、慢速衰减和固定 epsilon 的最后三次均值分别为 500.0、351.9、359.1。但慢速衰减更早确认达标，因此不能把后期排名直接解释为普遍的学习速度排名。

**epsilon 太小或衰减太快会怎样？**

通常可能因探索不足而错过有用动作，但本实验没有显示这种失败，快速衰减在 CartPole 上表现较好。我们没有测量状态空间覆盖率，不能仅凭曲线确定其原因。

## Task 2 - Question 3：Replay 容量与 batch size

### Question / 题目

比较 Replay 容量 1,000、10,000、100,000，以及 batch size 32、128，共六种组合，其他设置相同。

### Submit / 提交结果

提交所有设置的评估回报随环境步数变化的曲线。表中设置为 Replay 容量 / batch size。

![图 4：Replay 容量与 batch size 比较。](figures/q3_evaluation_returns.png)

{q3}

### Discussion / 讨论与解答

**Replay 容量如何影响曲线稳定性和平滑程度？**

影响与 batch size 有关。batch=32 时，容量 1,000 比 100,000 的后期曲线更稳，时间标准差为 0.0 和 185.7；batch=128 时，增大容量反而使后期时间标准差从 166.2 降到 113.7。

**batch size 如何影响收敛速度和最终表现？**

batch=128 并不总是更好。容量 1,000 时，最后三次回报从 500.0 降至 409.8；容量 10,000 时，从 349.1 升至 416.5，达标提前 10,000 步；容量 100,000 时回报提高，但达标推迟 60,000 步。

**本实验中哪种组合的 CartPole 表现最好？**

以最后三次评估均值判断，容量 1,000、batch=32 最好，达到 500.0 ± 0.0。这是本预算下最明确的后期表现结果，不代表在所有任务或评价指标下都最好。

## 参考与证据说明

依据课程 ELEN90098《Workshop 3: Deep Q Learning》讲义。全部答案使用已审核的 DQN 实验结果，本次整理没有新增正式训练。仅三个训练种子及固定评估种子的结果，不能直接推广到所有初始状态。
'''

for language, content in [('EN', en), ('ZH', zh)]:
    content = content.replace('{q1}', table('q1', language == 'ZH')).replace('{q2}', table('q2', language == 'ZH')).replace('{q3}', table('q3', language == 'ZH'))
    if language == 'EN':
        content = content.split('## Reference and evidence')[0].rstrip() + '\n'
        old = "For this and the following tables, returns are mean ± sample SD across three training seeds. Last-three return averages the final three checkpoints within each seed. A threshold is confirmed at the third consecutive evaluation mean of at least 475; the step is the median among successful seeds, with the successful-seed count shown. This measures when near-maximum performance is first confirmed, not permanent convergence."
        short = "Tables report mean ± SD. Last-three return averages three final checkpoints per seed. Threshold time confirms three consecutive evaluation means of at least 475; times are medians among successful seeds and counts show how many reached it. Confirmation does not guarantee later stability."
        short += ' Reference: provided ELEN90098 Workshop 3 handout. Three seeds give descriptive results, not a universal ranking.'
        content = content.replace(old+'\n\n','').replace('## Task 1 - Question 1:',short+'\n\n## Task 1 - Question 1:',1)
    else:
        content = content.split('## 参考与证据说明')[0].rstrip() + '\n'
        old = "本题及后续表格的回报均为三个训练种子的均值 ± 样本标准差。“最后三次评估均值”先在每个种子内平均最后三个检查点。“达标”指连续三次评估均值不低于 475，在第三次确认；表中给出成功种子的步数中位数及成功种子数。这不表示此后永久保持稳定。"
        short = "表格回报为均值 ± 标准差；最后三次评估先在每个种子内平均。“达标”指连续三次评估均值不低于 475，步数为成功种子的确认步数中位数，并列出成功种子数；不表示此后永久稳定。"
        short += ' 参考：课程 ELEN90098 Workshop 3 讲义。三个种子的结果仅作本实验描述，不代表普遍排名。'
        content = content.replace(old+'\n\n','').replace('## Task 1 - Question 1：',short+'\n\n## Task 1 - Question 1：',1)
    (OUT / f'Workshop3_Final_Report_{language}.md').write_text(content, encoding='utf-8')

coverage = {
    'task1_q1': ['state space', 'action space', 'reward', 'finite/continuous', 'determinism'],
    'task1_q2': ['TensorBoard scalar export', 'evaluation curve', 'successful training'],
    'task2_q1': ['3 learning rates', 'evaluation plot', 'speed', 'stability', 'too high', 'too low'],
    'task2_q2': ['3 schedules', 'evaluation plot', 'exploration effects', 'longer exploration', 'too small or fast decay'],
    'task2_q3': ['3 capacities x 2 batches', 'evaluation plot', 'stability', 'speed and final performance', 'best combination'],
}
qa = WS3 / 'tmp/final_v3_report_qa'
qa.mkdir(parents=True, exist_ok=True)
(qa / 'requirements_coverage.json').write_text(json.dumps(coverage, indent=2), encoding='utf-8')
(qa / 'formula_manifest.json').write_text(json.dumps({'formula': formula, 'file': str(formula_path), 'dpi': 600, 'sha256': hashlib.sha256(formula_path.read_bytes()).hexdigest()}, indent=2), encoding='utf-8')
print(json.dumps({'out': str(OUT), 'figures': [p.name for p in FIG.glob('*.png')]}, ensure_ascii=False))

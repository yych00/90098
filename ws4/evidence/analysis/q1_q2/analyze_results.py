"""Aggregate completed Q1/Q2 runs, plot actual-step evaluation curves, and report."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TASK_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK_ROOT / "shared"))
from manifest_paths import load_jobs

METHODS = ("vanilla", "rtg", "baseline")
LABELS = {"vanilla": "Vanilla", "rtg": "Reward-to-go", "baseline": "Previous-batch baseline"}
LEARNING_RATES = (1e-4, 1e-3, 1e-2)
BATCHES = (1000, 2000, 5000)
CORRECTED_V2_HASH = "54d3600cf459dd85ee20b8af00f1aacf47ed022eb0714807e99e1f4ba6794341"
BASELINE_DEFINITION = "mean of discounted trajectory returns from previous batches only; initial zero"
RTG_CONVENTION = "sum_{k=t}^{T-1} gamma**k * reward[k]"


def require_v2_config(config):
    expected = {"code_hash": CORRECTED_V2_HASH, "version": "task2-pg-2.0.0",
                "estimator_version": "corrected-v2", "baseline_definition": BASELINE_DEFINITION,
                "rtg_discount_convention": RTG_CONVENTION}
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Corrected-v2 analysis requires {key}={value!r}; found {config.get(key)!r}. "
                             "Historical v1 evidence must use its separate historical analysis workflow.")


def require_fresh_v2_jobs(jobs):
    if len(jobs) != 45 or len({job["id"] for job in jobs}) != 45:
        raise ValueError("The corrected-v2 Q1/Q2 catalogue must contain 45 distinct fresh runs")
    for job in jobs:
        if (job.get("estimator_version") != "corrected-v2"
                or job.get("expected_code_hash") != CORRECTED_V2_HASH
                or job.get("attempt") != 1 or job.get("recovery_status") != "fresh_corrected_v2"):
            raise ValueError(f"Not a fresh corrected-v2 catalogue entry: {job['id']}")


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_gradient_diagnostic(root):
    directory = root / "gradient_diagnostic"
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if (metadata.get("code_hash") != CORRECTED_V2_HASH or metadata.get("estimator_version") != "corrected-v2"
            or metadata.get("baseline_definition") != "unweighted mean discounted full-trajectory return from independent calibration batch, fixed for every diagnostic replicate"
            or metadata.get("actor_parameters_unchanged") is not True
            or metadata.get("status") != "completed"):
        raise ValueError("Gradient diagnostic is not the completed corrected-v2 independent-calibration experiment")
    return read_csv(directory / "gradient_variance.csv")


def diagnostic_completion(root):
    """A final analysis requires this round's diagnostic and semantic checks."""
    try:
        diagnostics = read_gradient_diagnostic(root)
        expected = {(method, batch) for method in METHODS for batch in BATCHES}
        actual = {(row["method"], int(row["batch_steps"])) for row in diagnostics}
        if actual != expected or len(diagnostics) != len(expected):
            raise ValueError("Gradient diagnostic must contain exactly all nine method/batch combinations")
        if any(int(row["replicates"]) < 2 or not np.isfinite(float(row["gradient_covariance_trace"])) for row in diagnostics):
            raise ValueError("Gradient diagnostic has incomplete or non-finite covariance statistics")
        checks = json.loads((root / "contract_checks.json").read_text(encoding="utf-8"))
        if (checks.get("status") != "passed" or checks.get("estimator_version") != "corrected-v2"
                or checks.get("code_hash") != CORRECTED_V2_HASH):
            raise ValueError("Semantic contract checks are not passed for the corrected-v2 trainer")
        return True, None
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return False, str(exc)


def baseline_batch_diagnostics(root, records):
    """Count actual single-trajectory updates; never infer them from batch size."""
    runs = []
    for record in records:
        if record["method"] != "baseline":
            continue
        metrics = read_csv(Path(record["formal_run_directory"]) / "metrics.csv")
        if len(metrics) != int(record["updates"]):
            raise ValueError(f"Metric/update count mismatch: {record['run_id']}")
        single = [row for row in metrics if int(row["batch_episodes"]) == 1]
        runs.append({"run_id": record["run_id"], "learning_rate": record["learning_rate"],
                     "batch_steps": record["batch_steps"], "seed": record["seed"],
                     "updates": len(metrics), "single_trajectory_updates": len(single),
                     "single_trajectory_fraction": len(single) / len(metrics) if metrics else 0,
                     "zero_gradient_updates": sum(float(row["gradient_norm"]) == 0 for row in metrics),
                     "code_hash": record["code_hash"]})
    configurations = []
    for learning_rate, batch_steps in sorted({(row["learning_rate"], row["batch_steps"]) for row in runs}):
        selected = [row for row in runs if (row["learning_rate"], row["batch_steps"]) == (learning_rate, batch_steps)]
        total = sum(row["updates"] for row in selected)
        single = sum(row["single_trajectory_updates"] for row in selected)
        configurations.append({"learning_rate": learning_rate, "batch_steps": batch_steps,
                               "seeds": len(selected), "updates": total,
                               "single_trajectory_updates": single,
                               "zero_gradient_updates": sum(row["zero_gradient_updates"] for row in selected),
                               "single_trajectory_fraction": single / total if total else 0})
    write_csv(root / "baseline_batch_run_statistics.csv", runs)
    write_csv(root / "baseline_batch_configuration_statistics.csv", configurations)
    diagnostic = {"definition": "batch_episodes == 1 in saved metrics.csv, counted once per optimizer update",
                  "corrected_v2_hash": CORRECTED_V2_HASH,
                  "interpretation": "Previous-batch historical baseline is fixed before collecting the current batch; a single trajectory does not subtract its own full return. Single-trajectory counts are descriptive, not counts of missing reward gradients. Exact saved zero-gradient counts are reported separately and do not imply frozen Adam parameters.",
                  "runs": runs, "configurations": configurations}
    (root / "baseline_batch_diagnostics.json").write_text(json.dumps(diagnostic, indent=2), encoding="utf-8")
    return diagnostic


def first_sustained_threshold(x, y, threshold=950, count=3):
    for index in range(len(y) - count + 1):
        if np.all(y[index:index + count] >= threshold):
            return float(x[index + count - 1])
    return None


def run_statistics(config, summary, evaluations):
    require_v2_config(config)
    x = np.array([float(row["env_steps"]) for row in evaluations])
    y = np.array([float(row["mean_return"]) for row in evaluations])
    if np.any(np.diff(x) <= 0):
        raise ValueError("evaluation training steps must be strictly increasing")
    requested = int(config["total_steps"])
    numerical_failure = summary.get("status") == "failed"
    if x[0] != 0 or (x[-1] < requested and not numerical_failure):
        raise ValueError("completed run must cover the requested training-step range and include step 0")
    # Piecewise-linear integration ends at the requested budget; no extrapolation.
    coverage_end = min(requested, x[-1])
    auc_x = np.r_[x[x < coverage_end], coverage_end]
    auc_y = np.interp(auc_x, x, y)
    auc = float(np.trapezoid(auc_y, auc_x) / requested) if not numerical_failure else None
    method = "rtg" if config["use_reward_to_go"] else "baseline" if config["use_baseline"] else "vanilla"
    sustained = first_sustained_threshold(x, y)
    record = {"method": method, "learning_rate": config["learning_rate"], "batch_steps": config["batch_steps"],
              "seed": config["seed"], "requested_steps": requested, "actual_training_steps": summary["actual_training_steps"],
              "updates": summary["updates"], "wall_seconds": summary["wall_seconds"],
              "status": "numerical_failure" if numerical_failure else "completed",
              "failure_reason": summary.get("error", ""),
              "last_finite_eval_step": int(x[-1]), "last_finite_eval_return": float(y[-1]),
              "initial_return": float(y[0]), "final_return": float(y[-1]) if not numerical_failure else None,
              "tail5_return": float(np.mean(y[-5:])) if not numerical_failure else None, "normalized_auc_return": auc,
              "sustained_950_steps": sustained, "evaluation_points": len(y),
              "sustained_950_after_requested_budget": sustained is not None and sustained > requested,
              "code_hash": config["code_hash"]}
    return record, (x, y)


def aggregate(records):
    groups = {}
    for record in records:
        if record["status"] != "completed":
            continue
        key = (record["method"], record["learning_rate"], record["batch_steps"])
        groups.setdefault(key, []).append(record)
    rows = []
    for (method, learning_rate, batch_steps), values in groups.items():
        solved = [row["sustained_950_steps"] for row in values if row["sustained_950_steps"] is not None]
        row = {"method": method, "learning_rate": learning_rate, "batch_steps": batch_steps,
               "seeds": len(values), "seed_ids": ",".join(str(row["seed"]) for row in values),
               "sustained_950_count": len(solved),
               "sustained_950_median_steps_among_successes": float(np.median(solved)) if solved else None,
               "actual_steps_min": min(value["actual_training_steps"] for value in values),
               "actual_steps_max": max(value["actual_training_steps"] for value in values)}
        for field in ("initial_return", "final_return", "tail5_return", "normalized_auc_return", "updates", "wall_seconds"):
            array = np.array([value[field] for value in values], dtype=float)
            row[field + "_mean"] = float(array.mean())
            row[field + "_std"] = float(array.std(ddof=1)) if len(array) > 1 else None
        rows.append(row)
    return rows


def curve(ax, run_curves, identifiers, label, color, max_steps):
    grid = np.arange(0, max_steps + 1, 10000)
    curves = [np.interp(grid, run_curves[identifier][0], run_curves[identifier][1]) for identifier in identifiers]
    values = np.stack(curves)
    mean = values.mean(axis=0)
    sd = values.std(axis=0, ddof=1) if len(values) > 1 else np.zeros_like(mean)
    ax.plot(grid, mean, label=label + (f" (n={len(values)})" if len(values) != 3 else ""), color=color, linewidth=1.7)
    if len(values) > 1:
        ax.fill_between(grid, mean - sd, mean + sd, color=color, alpha=0.15, linewidth=0)


def format_axes(ax, max_steps):
    ax.set_xlabel("Training environment steps")
    ax.set_ylabel("Evaluation return")
    ax.set_xlim(0, max_steps)
    ax.set_ylim(0, 1020)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    ax.grid(alpha=0.2)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No completed runs yet", transform=ax.transAxes, ha="center")


def plot_results(root, groups, run_curves, max_steps, failed_groups):
    plt.rcParams.update({"font.size": 10, "figure.dpi": 150, "savefig.dpi": 220})
    colors = ["#0072B2", "#D55E00", "#009E73"]
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    for ax, method in zip(axes, METHODS):
        for lr, color in zip(LEARNING_RATES, colors):
            ids = groups.get((method, lr, 1000), [])
            if ids:
                curve(ax, run_curves, ids, f"learning rate {lr:.0e}", color, max_steps)
            for index, identifier in enumerate(failed_groups.get((method, lr, 1000), [])):
                ax.plot(*run_curves[identifier], linestyle="--", color=color, alpha=0.7,
                        label=f"LR {lr:.0e}: numerical failure" if index == 0 else None)
        ax.set_title(LABELS[method])
        format_axes(ax, max_steps)
    fig.suptitle("WS4 Q1: Learning rate | batch 1000, gamma 0.99", fontsize=12)
    fig.text(0.5, 0.015, "Mean +/- 1 sample SD across training seeds; fixed-seed stochastic evaluation (10 episodes per checkpoint).", ha="center", fontsize=9)
    fig.tight_layout(rect=[0, 0.045, 1, 0.94])
    fig.savefig(figures / "q1_learning_rates.png")
    fig.savefig(figures / "q1_learning_rates.pdf")
    plt.close(fig)
    # Individual panels remain readable when embedded in portrait report pages.
    for method in METHODS:
        fig, ax = plt.subplots(figsize=(7.0, 4.6))
        for lr, color in zip(LEARNING_RATES, colors):
            ids = groups.get((method, lr, 1000), [])
            if ids:
                curve(ax, run_curves, ids, f"learning rate {lr:.0e}", color, max_steps)
            for index, identifier in enumerate(failed_groups.get((method, lr, 1000), [])):
                ax.plot(*run_curves[identifier], linestyle="--", color=color, alpha=0.7,
                        label=f"LR {lr:.0e}: numerical failure" if index == 0 else None)
        ax.set_title(f"WS4 Q1: {LABELS[method]} | minimum batch 1000")
        format_axes(ax, max_steps)
        fig.text(0.5, 0.015, "Mean +/- 1 sample SD across training seeds; gamma 0.99.", ha="center", fontsize=9)
        fig.tight_layout(rect=[0, 0.04, 1, 1])
        fig.savefig(figures / f"q1_learning_rates_{method}.png")
        fig.savefig(figures / f"q1_learning_rates_{method}.pdf")
        plt.close(fig)
    for method in METHODS:
        fig, ax = plt.subplots(figsize=(7.0, 4.6))
        for batch, color in zip(BATCHES, colors):
            ids = groups.get((method, 1e-3, batch), [])
            if ids:
                curve(ax, run_curves, ids, f"minimum batch steps {batch}", color, max_steps)
            for index, identifier in enumerate(failed_groups.get((method, 1e-3, batch), [])):
                ax.plot(*run_curves[identifier], linestyle="--", color=color, alpha=0.7,
                        label=f"batch {batch}: numerical failure" if index == 0 else None)
        ax.set_title(f"WS4 Q2: {LABELS[method]} | learning rate 1e-3")
        format_axes(ax, max_steps)
        fig.text(0.5, 0.015, "Mean +/- 1 sample SD across training seeds; gamma 0.99.", ha="center", fontsize=9)
        fig.tight_layout(rect=[0, 0.04, 1, 1])
        fig.savefig(figures / f"q2_batch_{method}.png")
        fig.savefig(figures / f"q2_batch_{method}.pdf")
        plt.close(fig)


def table(rows, question, language="en"):
    def mean_sd(row, prefix):
        sd = row[prefix + "_std"]
        return f"{row[prefix + '_mean']:.1f} +/- {sd:.1f}" if sd is not None else f"{row[prefix + '_mean']:.1f} (SD undefined)"
    header = ("| 方法 | 学习率 | Batch步数 | 完成种子 | 末段回报(均值 +/- SD) | 最终回报(均值 +/- SD) | AUC回报 | 持续>=950 |" if language == "zh" else
              "| Method | LR | Batch | Completed seeds | Tail return (mean +/- SD) | Final return (mean +/- SD) | AUC return | Sustained >=950 |")
    lines = [header,
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        if (question == "Q1" and row["batch_steps"] != 1000) or (question == "Q2" and row["learning_rate"] != 1e-3):
            continue
        lines.append(f"| {LABELS[row['method']]} | {row['learning_rate']:.0e} | {row['batch_steps']} | {row['seeds']}/3 | "
                     f"{mean_sd(row, 'tail5_return')} | "
                     f"{mean_sd(row, 'final_return')} | "
                     f"{row['normalized_auc_return_mean']:.1f} | {row['sustained_950_count']}/3 |")
    return "\n".join(lines)


def report(root, rows, records, completed, expected, partial, batch_diagnostic, gradient_diagnostic_complete):
    by_key = {(row["method"], row["learning_rate"], row["batch_steps"]): row for row in rows}
    title = "WS4 Task 2: Q1 and Q2" + (" - INCOMPLETE EXPERIMENT SNAPSHOT" if partial else "")
    failures = [record for record in records if record["status"] == "numerical_failure"]
    budget = records[0]["requested_steps"]
    provenance_en = ("These are fresh corrected-v2 runs (task2-pg-2.0.0; trainer hash "
                     f"`{CORRECTED_V2_HASH}`), trained into a separate workspace. Historical-v1 PG results are not pooled "
                     "with this experiment set. Every analyzed config must declare the v2 baseline and RTG contracts.")
    provenance_zh = (f"这些是独立工作区重新训练的 corrected-v2 实验（task2-pg-2.0.0，训练器哈希 `{CORRECTED_V2_HASH}`）。"
                     "历史 v1 的 PG 结果不混入本组统计；每个运行配置均须满足 v2 的 baseline 与 RTG 约定。")
    estimator_en = ("Vanilla repeats R = sum_k gamma^k r_k at every step of a trajectory. Corrected RTG uses "
                    "sum_{k>=t} gamma^k r_k, including the outer gamma^t factor, so dropping past rewards preserves "
                    "the episode-origin discount convention. All losses average over collected time steps. Complete-episode "
                    "batches overshoot a minimum step target and have random lengths; this normalization and stopping rule "
                    "are shared practical choices, not a proof of strict unbiasedness for a fixed-episode gradient. "
                    "Training-return differences alone do not isolate estimator variance.")
    estimator_zh = ("Vanilla 在每个时间步重复使用 R = sum_k gamma^k r_k；修正 RTG 使用 sum_{k>=t} gamma^k r_k，"
                    "包含外层 gamma^t，去除过去奖励后保持从回合起点折扣的约定。loss 均按收集的时间步求平均。"
                    "完整轨迹 batch 在达到最小步数后结束，长度随机且可能超出目标；共同的归一化和停止规则不等于固定轨迹数梯度的严格无偏证明。"
                    "训练回报的差异本身不能隔离梯度估计方差。")
    baseline_en = ("The previous-batch baseline is the unweighted mean discounted full-trajectory return over all "
                   "episodes in earlier batches, initially zero. Its scalar value is fixed before the current batch and "
                   "does not subtract the current trajectory's own return; the historical same-batch single-trajectory "
                   "degeneracy is removed. Policy drift can make this historical estimate stale, and it is not guaranteed "
                   "to be variance-optimal. The table reports actual single-trajectory batch frequency; this is no longer "
                   "a count of updates without reward signal. Separate CSV/JSON fields count exactly zero saved gradient norms.")
    baseline_zh = ("Previous-batch baseline 为所有先前 batch 中各轨迹折扣全回报的等权平均，初始为零。"
                   "该标量在当前 batch 之前固定，不减去当前轨迹自身的回报，消除了历史同批均值在单轨迹 batch 中自我抵消的问题。"
                   "策略变化会使历史估计滞后，且不保证是方差最优 baseline。下表只描述真实单轨迹 batch 频率，已不代表缺少奖励信号的更新次数；"
                   "另在 CSV/JSON 中单独统计保存梯度范数精确为零的次数。")
    batch_table_en = ["| LR | Minimum batch steps | Single-trajectory updates / all updates | Fraction |", "|---|---:|---:|---:|"]
    for row in batch_diagnostic["configurations"]:
        batch_table_en.append(f"| {row['learning_rate']:.0e} | {row['batch_steps']} | {row['single_trajectory_updates']}/{row['updates']} | {row['single_trajectory_fraction']:.1%} |")
    selected = [row for row in batch_diagnostic["runs"] if row["learning_rate"] == 1e-2 and row["batch_steps"] == 1000]
    seed_counts = "; ".join(f"seed {row['seed']}: {row['single_trajectory_updates']}/{row['updates']} ({row['single_trajectory_fraction']:.1%})" for row in selected)
    baseline_en += " For LR 1e-2 / batch 1000: " + seed_counts + "."
    baseline_zh += " 学习率 1e-2、batch 1000 的逐种子统计为：" + seed_counts + "。"
    batch_table_en = "\n".join(batch_table_en)
    batch_table_zh = batch_table_en.replace("Minimum batch steps", "最小 batch 步数").replace("Single-trajectory updates / all updates", "单轨迹更新 / 全部更新").replace("Fraction", "比例")
    actual_steps = [record["actual_training_steps"] for record in records if record["status"] == "completed"]
    setup_en = (f"This analysis includes {completed}/{expected} completed runs and {len(failures)} numerically terminated runs. The design has 15 configurations and three training seeds "
                "(0, 1, 2), covering 45 runs. Q1 varies learning rates 1e-4, 1e-3 and 1e-2 at batch 1000. Q2 varies minimum batch "
                "steps 1000, 2000 and 5000 at learning rate 1e-3; the three common configurations are reused. "
                "All core runs use gamma 0.99, a 64x2 ReLU policy, Adam, no gradient clipping, no advantage normalization and no GAE. "
                "Baseline means the historical mean discounted full-trajectory return from previous batches, not a learned state-value function. "
                "It is initialized at zero and updated only after the current advantages have been formed. "
                "Gamma differs from historical Task 1 (0.9), so these are new controlled experiments rather than a direct continuation.")
    setup_en += (" All 45 catalogue entries are fresh corrected-v2 training runs in this separate workspace. "
                 "No historical PG run is reused or resumed. Shared Q1/Q2 configurations reuse only the same "
                 "new run within this v2 design, not an old result.")
    evidence_en = (f"Each run requests {budget:,} training environment steps, completing its final trajectory/batch and recording the actual count. "
                   "Ten stochastic evaluation episodes use fixed environment/action-sampling seeds 10000-10009 at step 0, approximately every 10,000 "
                   "training steps, and after the final update. Evaluation has a separate environment and restores training RNG state. "
                   "Curves interpolate individual evaluations linearly onto a 10k grid up to the requested budget; bands are one sample standard "
                   "deviation across training-seed evaluation means, not confidence intervals. Tail return averages each run's final five evaluations; "
                   "AUC return is the step-integrated return divided by the requested budget. Sustained success requires evaluation mean >=950 "
                   "at three consecutive checkpoints, confirmed at the third checkpoint; unsuccessful runs do not receive an invented convergence time. "
                   "Tail/final returns and threshold confirmation use all saved evaluations, including the final whole-batch overshoot; "
                   "any confirmation after the nominal budget is marked separately in the per-run statistics. "
                   "These are descriptive threshold confirmations, not guarantees of later stability.")
    if actual_steps:
        evidence_en += (f" Completed runs actually collected {min(actual_steps):,}-{max(actual_steps):,} training steps each; "
                        "run_statistics.csv retains every seed's exact actual step and update counts.")
    q1_en, q2_en, q1_zh, q2_zh = [], [], [], []
    for method in METHODS:
        candidates = [by_key[(method, lr, 1000)] for lr in LEARNING_RATES if (method, lr, 1000) in by_key]
        if candidates:
            best = max(candidates, key=lambda row: row["tail5_return_mean"])
            successful = [row for row in candidates if row["sustained_950_count"]]
            fastest_text = "None of these configurations achieved sustained success in any completed seed."
            fastest_zh = "这些已完成配置均没有种子满足持续成功判据。"
            if successful:
                fastest = min(successful, key=lambda row: row["sustained_950_median_steps_among_successes"])
                fastest_text = (f"The lowest median sustained-threshold confirmation step among successful seeds was "
                                f"{fastest['sustained_950_median_steps_among_successes']:,.0f} at LR {fastest['learning_rate']:.0e} "
                                f"({fastest['sustained_950_count']}/3 seeds succeeded); this success-conditioned number must be read with its success count.")
                fastest_zh = (f"成功种子中的首次持续达标确认步数中位数最低为 {fastest['sustained_950_median_steps_among_successes']:,.0f}，"
                              f"对应学习率 {fastest['learning_rate']:.0e}；成功 {fastest['sustained_950_count']}/3，不能忽略失败种子。")
            q1_en.append(f"For {LABELS[method]}, the highest mean tail return was {best['tail5_return_mean']:.1f} at LR {best['learning_rate']:.0e}. {fastest_text}")
            q1_zh.append(f"{LABELS[method]} 的最高末段平均回报为 {best['tail5_return_mean']:.1f}，对应学习率 {best['learning_rate']:.0e}。{fastest_zh}")
        low, high = by_key.get((method, 1e-3, 1000)), by_key.get((method, 1e-3, 5000))
        if low and high:
            delta = high["tail5_return_mean"] - low["tail5_return_mean"]
            q2_en.append(f"For {LABELS[method]}, increasing the minimum batch from 1000 to 5000 changed mean tail return "
                         f"from {low['tail5_return_mean']:.1f} to {high['tail5_return_mean']:.1f} (change {delta:+.1f}); "
                         f"mean update count changed from {low['updates_mean']:.1f} to {high['updates_mean']:.1f}. "
                         f"The normalized AUC return changed from {low['normalized_auc_return_mean']:.1f} to {high['normalized_auc_return_mean']:.1f}.")
            q2_zh.append(f"{LABELS[method]} 的最小 batch 从 1000 增至 5000 后，末段回报从 {low['tail5_return_mean']:.1f} "
                         f"变为 {high['tail5_return_mean']:.1f}（{delta:+.1f}），平均更新次数从 {low['updates_mean']:.1f} "
                         f"变为 {high['updates_mean']:.1f}；AUC 平均回报从 {low['normalized_auc_return_mean']:.1f} 变为 {high['normalized_auc_return_mean']:.1f}。")
    higher_en, higher_zh = [], []
    for method in METHODS:
        low, high = by_key.get((method, 1e-3, 1000)), by_key.get((method, 1e-2, 1000))
        if low and high:
            higher_en.append(f"{LABELS[method]} {high['tail5_return_mean'] - low['tail5_return_mean']:+.1f}")
            higher_zh.append(f"{LABELS[method]} {high['tail5_return_mean'] - low['tail5_return_mean']:+.1f}")
    discussion_en = ("Changing LR from 1e-3 to 1e-2 changed tail return by: " + "; ".join(higher_en) + ". "
                     "A larger learning rate can magnify noisy parameter changes, while a smaller one can delay improvement within a fixed budget. "
                     "Adam rescales coordinates, so a simple learning-rate-squared rule is not an exact description of these optimizer updates. "
                     "Return variability, advantage standard deviation and gradient norm are not direct measurements of gradient-estimator variance.")
    discussion_zh = ("学习率由 1e-3 增至 1e-2，末段回报变化分别为：" + "；".join(higher_zh) + "。"
                     "较大学习率可能放大含噪更新，较小学习率也可能使有限步数内的改善更慢。这里使用 Adam，其逐坐标缩放使简单的学习率平方规律"
                     "不能直接等同实际参数更新方差。回报波动、优势标准差和梯度范数均不等于梯度估计方差。")
    for method in ("rtg", "baseline"):
        pairs = [(by_key.get((method, lr, 1000)), by_key.get(("vanilla", lr, 1000))) for lr in (1e-3, 1e-2)]
        if all(technique and vanilla for technique, vanilla in pairs):
            gaps = [technique["tail5_return_mean"] - vanilla["tail5_return_mean"] for technique, vanilla in pairs]
            discussion_en += (f" Relative to Vanilla, {LABELS[method]}'s mean tail-return gap was {gaps[0]:+.1f} at LR 1e-3 "
                              f"and {gaps[1]:+.1f} at LR 1e-2; this describes the observed method-by-learning-rate interaction.")
            discussion_zh += (f" 相对 Vanilla，{LABELS[method]} 的末段回报差在学习率 1e-3 时为 {gaps[0]:+.1f}，"
                              f"在 1e-2 时为 {gaps[1]:+.1f}，这是本次方法与学习率交互作用的实际表现。")
    variance_file = root / "gradient_diagnostic" / "gradient_variance.csv"
    variance_en, variance_zh = "The separate gradient-variance diagnostic has not been completed.", "独立梯度方差诊断尚未完成。"
    if gradient_diagnostic_complete:
        diagnostics = read_gradient_diagnostic(root)
        diagnostic_metadata = json.loads((variance_file.parent / "metadata.json").read_text(encoding="utf-8"))
        lookup = {(row["method"], int(row["batch_steps"])): row for row in diagnostics}
        ratios = {method: float(lookup[(method, 5000)]["gradient_covariance_trace"]) / float(lookup[(method, 1000)]["gradient_covariance_trace"]) for method in METHODS}
        weakest = max(("rtg", "baseline"), key=ratios.get)
        statistics = "; ".join(f"{LABELS[method]} {ratios[method]:.3f}" for method in METHODS)
        variance_en = ("A separate paired diagnostic freezes one common randomly initialized policy and estimates the trace of the sample covariance "
                       f"of batch gradient vectors from {diagnostic_metadata['replicates']} independent trajectory collections. The variance-trace ratios (batch 5000 / batch 1000) are "
                       f"{statistics}. Among RTG and the previous-batch baseline, the smallest proportional reduction at this policy is for {LABELS[weakest]}. "
                       "All methods share identical trajectories within each replicate; batch sizes use nested complete-episode prefixes. "
                       "The second panel divides covariance trace by the squared norm of the sample mean gradient, an empirical noise-to-signal ratio. "
                       "The corrected-v2 baseline is fixed from a separate calibration collection, independent of all diagnostic replicates. "
                       f"Calibration seed {diagnostic_metadata['calibration_seed']} collected {diagnostic_metadata['calibration_actual_steps']} steps in "
                       f"{diagnostic_metadata['calibration_episodes']} complete episodes, giving baseline {diagnostic_metadata['baseline_value']:.6f}. "
                       "These extra diagnostic samples are excluded from formal training budgets. "
                       "It is conditional on that calibration estimate; random batch lengths and step averaging limit a strict unbiased-estimator interpretation. "
                       "This is a local initialization diagnostic, not evidence that the same ordering or reduction holds throughout training.")
        variance_zh = (f"另外固定同一个随机初始化策略，用 {diagnostic_metadata['replicates']} 组独立轨迹采样、同组同轨迹比较三种梯度估计器，直接计算梯度向量样本协方差的迹。"
                       f"batch 5000 / batch 1000 的方差迹比值分别为 {statistics}；RTG 与 previous-batch baseline 中，此初始化策略下相对下降最小的是 {LABELS[weakest]}。"
                       "各 batch 大小是完整轨迹的嵌套前缀；右图将方差迹除以样本平均梯度的模长平方，作为经验噪声信号比。"
                       "v2 诊断使用独立 calibration 采样得到的固定 baseline，不使用任何诊断 replicate 自身回报估计 baseline。统计条件于该校准值；随机批长和按时间步平均限制了严格无偏估计器的解释。"
                       f"校准 seed {diagnostic_metadata['calibration_seed']} 共采样 {diagnostic_metadata['calibration_actual_steps']} 步、"
                       f"{diagnostic_metadata['calibration_episodes']} 条完整轨迹，baseline 为 {diagnostic_metadata['baseline_value']:.6f}；"
                       "这些额外诊断样本不计入正式训练预算。"
                       "该局部诊断不能证明整个训练过程都保持相同方差排名或降幅。")
    tradeoff_en = ("Larger batches reduce update frequency under a fixed environment-step budget. They can stabilize an individual gradient estimate "
                   "while spending more samples before adapting the policy, so sample efficiency need not improve monotonically. "
                   "Each backward pass processes more data and uses more batch memory, but fewer passes and concurrent execution mean the recorded wall times are not a controlled computational-speed benchmark. "
                   "The learning outcomes and the frozen-policy variance diagnostic address different questions. Descriptive rankings from three "
                   "training seeds are not significance tests, and fixed evaluation seeds limit generalization claims.")
    tradeoff_zh = ("固定环境步数下，较大 batch 会减少策略更新次数。单次梯度可能更稳定，但更新前需要更多采样，因此样本效率不保证单调提高。"
                   "每次反向传播处理更多数据、占用更多 batch 内存，但更新次数也减少；并发执行下的运行耗时不能作为严格的计算速度对照。"
                   "训练表现与固定策略下的方差诊断回答不同问题。三个训练种子的描述性排名不是显著性检验，固定评估种子也限制了泛化结论。")
    monotonic, nonmonotonic = [], []
    for method in METHODS:
        values = [by_key.get((method, 1e-3, batch)) for batch in BATCHES]
        if all(values):
            destination = monotonic if np.all(np.diff([row["final_return_mean"] for row in values]) >= 0) else nonmonotonic
            destination.append(LABELS[method])
    if monotonic or nonmonotonic:
        answer_en = "The final-evaluation means increase monotonically with batch size for " + (", ".join(monotonic) or "none of the fully observed settings")
        answer_zh = "最终单次评估的跨种子均值随 batch 增大单调不减的方法为：" + ("、".join(monotonic) or "无")
        if nonmonotonic:
            answer_en += "; they do not for " + ", ".join(nonmonotonic)
            answer_zh += "；不单调的方法为：" + "、".join(nonmonotonic)
        q2_en.append(answer_en + ".")
        q2_zh.append(answer_zh + "。")
    endpoint_gains = {}
    for method in ("rtg", "baseline"):
        low, high = by_key.get((method, 1e-3, 1000)), by_key.get((method, 1e-3, 5000))
        if low and high:
            endpoint_gains[method] = high["tail5_return_mean"] - low["tail5_return_mean"]
    if len(endpoint_gains) == 2:
        weakest_endpoint = min(endpoint_gains, key=endpoint_gains.get)
        q2_en.append(f"Among RTG and the previous-batch baseline, {LABELS[weakest_endpoint]} has the smaller observed tail-return gain "
                     f"from batch 1000 to 5000 ({endpoint_gains[weakest_endpoint]:+.1f}). This is a performance comparison, not a measurement of gradient noise.")
        q2_zh.append(f"RTG 与 previous-batch baseline 中，batch 1000 到 5000 的末段回报增量较小者为 {LABELS[weakest_endpoint]}"
                     f"（{endpoint_gains[weakest_endpoint]:+.1f}）。这是训练表现比较，不能替代梯度噪声测量。")
    q1_images_en = "\n\n".join(f"![Learning rates: {LABELS[method]}](figures/q1_learning_rates_{method}.png)" for method in METHODS)
    q1_images_zh = "\n\n".join(f"![学习率：{LABELS[method]}](figures/q1_learning_rates_{method}.png)" for method in METHODS)
    en = f"# {title}\n\n{provenance_en}\n\n{setup_en}\n\n{estimator_en}\n\n{baseline_en}\n\n{batch_table_en}\n\n{evidence_en}\n\n## Q1: Learning rate\n\n{q1_images_en}\n\n{table(rows, 'Q1')}\n\n" + "\n\n".join(q1_en) + f"\n\n{discussion_en}\n\n## Q2: Batch steps\n\n"
    for method in METHODS:
        en += f"![Batch steps: {LABELS[method]}](figures/q2_batch_{method}.png)\n\n"
    en += table(rows, "Q2") + "\n\n" + "\n\n".join(q2_en) + f"\n\n{tradeoff_en}\n\n{variance_en}\n"
    if gradient_diagnostic_complete:
        en += "\n![Frozen-policy gradient variance](figures/q2_gradient_variance.png)\n"
    zh = f"# WS4 Task 2 Q1/Q2 中文说明{'（实验未完成快照）' if partial else ''}\n\n已纳入 {completed}/{expected} 个已完成实验，另有 {len(failures)} 个数值发散提前终止；15 种配置，每种训练种子 0、1、2。"
    zh += "Q1 使用三个学习率 1e-4、1e-3、1e-2，batch=1000；Q2 使用 batch=1000、2000、5000，学习率 1e-3。重叠配置直接复用结果。"
    zh += "公共参数为 gamma=0.99、64x2 ReLU 策略、Adam；核心对照不使用 GAE、优势标准化或梯度裁剪。Baseline 为先前所有 batch 中轨迹折扣全回报的等权历史均值，初始为零；不是学习状态价值函数，也不使用当前 batch 的回报形成当前 baseline。"
    zh += "历史 Task 1 的 gamma=0.9，因此本实验不是历史结果直接续训。\n\n"
    zh += f"{provenance_zh}\n\n{estimator_zh}\n\n{baseline_zh}\n\n{batch_table_zh}\n\n"
    zh += ("本轮清单的45个运行全部在独立目录从头训练 corrected-v2，不复用或续训历史PG结果。"
           "Q1/Q2重叠配置仅在本轮设计内部共用同一个新运行，不引用旧实验。\n\n")
    zh += f"每次请求 {budget:,} 训练环境步，完整批次结束可能超出；保存实际训练步数。评估为独立环境下十个固定环境及动作采样种子 10000-10009 的随机策略，每次保存、恢复训练 RNG。"
    if actual_steps:
        zh += f"已完成运行的实际训练步数为 {min(actual_steps):,}-{max(actual_steps):,}，各个种子的精确步数与更新次数见 run_statistics.csv。"
    zh += "曲线将实际评估点线性插值到每 10k 步网格，显示三个训练种子均值与一倍样本标准差，非置信区间。末段回报是各运行最后五次评估均值；AUC 回报是预算内曲线积分除以预算。"
    zh += "持续成功定义为连续三次评估平均回报至少 950，并在第三次评估确认；未达标运行不人为赋予收敛时间。末段、最终回报及达标确认使用全部保存评估，包含最后完整批次超出的步数；若确认时刻超过标称预算，须与预算内收敛区别解读，精确时刻见逐运行统计。这只是描述性达标确认，不保证此后一直稳定。\n\n## Q1 学习率\n\n"
    zh += q1_images_zh + "\n\n" + table(rows, "Q1", "zh") + "\n\n" + "\n\n".join(q1_zh) + f"\n\n{discussion_zh}\n\n## Q2 Batch 大小\n\n"
    for method in METHODS:
        zh += f"![Batch大小：{LABELS[method]}](figures/q2_batch_{method}.png)\n\n"
    zh += table(rows, "Q2", "zh") + "\n\n" + "\n\n".join(q2_zh) + f"\n\n{tradeoff_zh}\n\n{variance_zh}\n"
    if gradient_diagnostic_complete:
        zh += "\n![初始化固定策略的梯度方差](figures/q2_gradient_variance.png)\n"
    if failures:
        failure_table = ["| Run | Training steps at termination | Last finite evaluation step | Last finite return | Reason |",
                         "|---|---:|---:|---:|---|"]
        for record in failures:
            reason = record["failure_reason"].replace("|", "/").replace("\n", " ")
            failure_table.append(f"| {record['run_id']} | {record['actual_training_steps']} | {record['last_finite_eval_step']} | {record['last_finite_eval_return']:.1f} | {reason} |")
        failure_text = "\n".join(failure_table)
        en += ("\n## Numerical termination\n\n" + failure_text + "\n\nFailed runs are plotted only through their last finite evaluation (dashed curves). "
               "The untrained interval is not filled. Their last finite return is not a final-budget performance estimate. Tables of tail/final/AUC means "
               "use completed runs only and show the completion count; threshold success retains all three intended seeds as its denominator. "
               "These numerical failures are outcomes, not rerun with a reduced learning rate.\n")
        zh += ("\n## 数值发散提前终止\n\n" + failure_text + "\n\n发散运行用虚线只绘至最后有限评估点，不补齐未训练区间。"
               "最后有限回报不能当作完整预算的最终表现。末段/最终/AUC 均值只统计完成运行，并标明完成数；成功率分母保留原定三个种子。"
               "不通过悄悄调小学习率重跑来替换发散结果。\n")
    (root / "report_en.md").write_text(en, encoding="utf-8")
    (root / "report_zh.md").write_text(zh, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow_partial", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    jobs = load_jobs(root / "jobs.json")
    require_fresh_v2_jobs(jobs)
    records, incomplete, run_curves, groups, failed_groups = [], [], {}, {}, {}
    for job in jobs:
        output = Path(job["output_dir"])
        assert output.resolve().is_relative_to(TASK_ROOT), "Refusing to read a host run outside the current package root"
        if not (output / "summary.json").exists():
            incomplete.append(job["id"])
            continue
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        error = summary.get("error", "")
        numerical_failure = summary.get("status") == "failed" and (
            "FloatingPointError" in error or ("Expected parameter" in error and "invalid values" in error))
        if summary.get("status") != "completed" and not numerical_failure:
            incomplete.append(job["id"])
            continue
        config = json.loads((output / "config.json").read_text(encoding="utf-8"))
        stats, values = run_statistics(config, summary, read_csv(output / "evaluations.csv"))
        stats = {"run_id": job["id"], **stats, "formal_run_directory": str(output),
                 "recorded_output_dir": config["output_dir"], "attempt": 1, "run_origin": "fresh_corrected_v2"}
        records.append(stats)
        run_curves[job["id"]] = values
        key = stats["method"], stats["learning_rate"], stats["batch_steps"]
        (failed_groups if numerical_failure else groups).setdefault(key, []).append(job["id"])
    if incomplete and not args.allow_partial:
        raise SystemExit(f"{len(incomplete)}/{len(jobs)} runs incomplete. Use --allow_partial for an explicitly marked snapshot.")
    if not records:
        raise SystemExit("No completed experiment runs to analyze")
    gradient_diagnostic_complete, diagnostic_error = diagnostic_completion(root)
    if not gradient_diagnostic_complete and not args.allow_partial:
        raise SystemExit("Refusing a complete corrected-v2 report: run verify_contracts.py and "
                         "gradient_variance.py for this trainer first. Diagnostic/contract evidence missing or invalid: "
                         + str(diagnostic_error))
    hashes = sorted({record["code_hash"] for record in records})
    if len(hashes) != 1:
        raise ValueError("Core experiments contain multiple trainer code hashes")
    if hashes != [CORRECTED_V2_HASH]:
        raise ValueError(f"Corrected-v2 report requires only {CORRECTED_V2_HASH}; found {hashes}")
    rows = aggregate(records)
    write_csv(root / "run_statistics.csv", records)
    write_csv(root / "configuration_statistics.csv", rows)
    max_steps = records[0]["requested_steps"]
    assert len({record["requested_steps"] for record in records}) == 1, "mixed requested budgets cannot be aggregated as one controlled comparison"
    plot_results(root, groups, run_curves, max_steps, failed_groups)
    variance_file = root / "gradient_diagnostic" / "gradient_variance.csv"
    if gradient_diagnostic_complete:
        diagnostics = read_gradient_diagnostic(root)
        plt.rcParams.update({"font.size": 11.5})
        fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.5))
        for method, color in zip(METHODS, ("#0072B2", "#D55E00", "#009E73")):
            selected = sorted([row for row in diagnostics if row["method"] == method], key=lambda row: int(row["batch_steps"]))
            x = [int(row["batch_steps"]) for row in selected]
            for ax, field in zip(axes, ("gradient_covariance_trace", "relative_noise")):
                ax.plot(x, [float(row[field]) for row in selected], marker="o", color=color, label=LABELS[method])
                ax.set_yscale("log")
                ax.set_xlabel("Minimum batch steps")
                ax.set_xticks(BATCHES)
                ax.grid(alpha=0.2)
        axes[0].set_ylabel("Gradient covariance trace")
        axes[1].set_ylabel("Gradient noise / squared signal")
        fig.suptitle("Frozen initial policy: 20 independent paired batches", fontsize=12)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.91), ncol=3, fontsize=9.5, frameon=False)
        fig.tight_layout(rect=[0, 0, 1, 0.83])
        fig.savefig(root / "figures" / "q2_gradient_variance.png")
        fig.savefig(root / "figures" / "q2_gradient_variance.pdf")
        plt.close(fig)
    completed_count = sum(record["status"] == "completed" for record in records)
    failed_count = sum(record["status"] == "numerical_failure" for record in records)
    batch_diagnostic = baseline_batch_diagnostics(root, records)
    report(root, rows, records, completed_count, len(jobs), bool(incomplete) or not gradient_diagnostic_complete,
           batch_diagnostic, gradient_diagnostic_complete)
    result = {"status": "partial" if incomplete or not gradient_diagnostic_complete else "completed_with_numerical_failures" if failed_count else "completed",
              "completed_runs": completed_count, "numerical_failures": failed_count, "expected_runs": len(jobs),
              "incomplete_ids": incomplete, "code_hashes": hashes,
              "training_evidence_version": "corrected_v2",
              "corrected_v2_results_included": True,
              "gradient_diagnostic_complete": gradient_diagnostic_complete,
              "gradient_diagnostic_error": diagnostic_error,
              "baseline_batch_diagnostics": "baseline_batch_diagnostics.json",
              "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "formal_manifest_provenance": "45 distinct fresh corrected-v2 runs; overlapping Q1/Q2 conditions share only the same new v2 run",
              "fresh_v2_jobs": len(jobs), "completed_fresh_v2_runs": completed_count,
              "reused_historical_pg_runs": 0,
              "actual_training_steps_min": min(row["actual_training_steps"] for row in records),
              "actual_training_steps_max": max(row["actual_training_steps"] for row in records),
              "curve_alignment": "piecewise linear interpolation to 10k-step grid, no extrapolation, ending at requested budget",
              "spread": "one sample SD across training-seed evaluation means; not confidence interval",
              "tail_return": "mean of final five evaluation means per seed, then mean and sample SD over seeds",
              "auc_return": "piecewise-linear integral up to requested step budget divided by requested budget",
              "sustained_success": "mean evaluation return >=950 for three consecutive checkpoints, confirmation at third checkpoint; report count and success-conditioned median"}
    (root / "analysis_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

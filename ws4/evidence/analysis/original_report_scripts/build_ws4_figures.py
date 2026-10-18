"""Build report-only figures from audited corrected-v2 evidence; never train."""
from pathlib import Path
import csv
import hashlib
import inspect
import json
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import gymnasium
from gymnasium.envs.mujoco.inverted_pendulum_v5 import InvertedPendulumEnv

COURSE = Path(__file__).resolve().parents[3]
SOURCE = COURSE / "ws3/task2_workspace/corrected_v2"
OUTPUT = COURSE / "ws4/outputs/final_report_v3"
FIGURES = OUTPUT / "figures"
QA = Path(__file__).resolve().parent
FIGURES.mkdir(parents=True, exist_ok=True)

METHODS = ("vanilla", "rtg", "baseline")
LABELS = {"vanilla": "Vanilla PG", "rtg": "Reward-to-go", "baseline": "Previous-batch baseline"}
COLORS = ("#2377B4", "#D45A20", "#27865F")


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def run_curves(method, learning_rate, batch_steps):
    values = []
    grid = np.arange(0, 500001, 10000)
    for seed in (0, 1, 2):
        directory = SOURCE / f"ws4/q1_q2/runs/{method}_lr{learning_rate:.0e}_b{batch_steps}_s{seed}"
        config = json.loads((directory / "config.json").read_text())
        assert config["estimator_version"] == "corrected-v2"
        rows = read_csv(directory / "evaluations.csv")
        x, y = np.array([int(row["env_steps"]) for row in rows]), np.array([float(row["mean_return"]) for row in rows])
        assert x[0] == 0 and x[-1] >= grid[-1] and np.all(np.diff(x) > 0)
        values.append(np.interp(grid, x, y))
    return grid, np.asarray(values)


def curve(axis, method, rate, batch, label, color):
    grid, values = run_curves(method, rate, batch)
    mean, sd = values.mean(axis=0), values.std(axis=0, ddof=1)
    axis.plot(grid, mean, label=label, color=color, linewidth=1.9)
    axis.fill_between(grid, np.maximum(0, mean - sd), mean + sd, color=color, alpha=.14, linewidth=0)


def style(axis, upper=1050):
    axis.set(xlim=(0, 500000), ylim=(0, upper), xlabel="Training environment steps", ylabel="Evaluation return")
    ticks = np.arange(0, 500001, 100000)
    axis.set_xticks(ticks, ["0", "100k", "200k", "300k", "400k", "500k"])
    axis.grid(alpha=.2)
    axis.spines[["right", "top"]].set_visible(False)


plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
                     "legend.fontsize": 11, "xtick.labelsize": 11, "ytick.labelsize": 11})
fig, axis = plt.subplots(figsize=(8.5, 3.7), constrained_layout=True)
for method, color in zip(METHODS, COLORS):
    curve(axis, method, .001, 1000, LABELS[method], color)
style(axis)
axis.axhline(950, color="#777777", linestyle="--", linewidth=.8)
axis.set_title("InvertedPendulum-v5: common learning rate 0.001, batch 1000")
axis.legend(loc="upper left", frameon=False)
fig.savefig(FIGURES / "task1_q2_methods.png", dpi=220)
plt.close(fig)

fig, axes = plt.subplots(3, 1, figsize=(8.5, 7.6), sharex=True, sharey=True, constrained_layout=True)
for axis, method in zip(axes, METHODS):
    for rate, color in zip((.0001, .001, .01), COLORS):
        curve(axis, method, rate, 1000, f"Learning rate {rate:g}", color)
    style(axis)
    axis.set_title(LABELS[method], loc="left")
    if axis is not axes[-1]:
        axis.set_xlabel("")
axes[0].legend(loc="upper left", ncol=3, frameon=False, fontsize=10.5)
fig.savefig(FIGURES / "task2_q1_learning_rates.png", dpi=220)
plt.close(fig)

for method in METHODS:
    fig, axis = plt.subplots(figsize=(8.5, 3.25), constrained_layout=True)
    maxima = []
    for batch, color in zip((1000, 2000, 5000), COLORS):
        curve(axis, method, .001, batch, f"Batch {batch}", color)
        _, values = run_curves(method, .001, batch)
        maxima.append(float((values.mean(axis=0) + values.std(axis=0, ddof=1)).max()))
    style(axis, upper=np.ceil(max(maxima) * 1.05 / 50) * 50)
    axis.set_title(f"{LABELS[method]}: learning rate 0.001")
    axis.legend(loc="upper left", ncol=3, frameon=False)
    fig.savefig(FIGURES / f"task2_q2_batch_{method}.png", dpi=220)
    plt.close(fig)

for source, destination in [("ws4/q3_q4/figures/q3_pg_vs_dqn.png", "task2_q3_pg_vs_dqn.png"),
                            ("ws4/q3_q4/figures/q4_halfcheetah.png", "task2_q4_halfcheetah.png"),
                            ("ws4/q3_q4/representative_seed0_frames.png", "task2_q4_frames.png")]:
    shutil.copy2(SOURCE / source, FIGURES / destination)

env = gymnasium.make("InvertedPendulum-v5")
environment = {"gymnasium": gymnasium.__version__, "class_source": inspect.getfile(InvertedPendulumEnv),
               "observation_space": str(env.observation_space), "action_space": str(env.action_space),
               "max_episode_steps": env.spec.max_episode_steps,
               "step_source": inspect.getsource(InvertedPendulumEnv.step),
               "reset_source": inspect.getsource(InvertedPendulumEnv.reset_model)}
env.close()
(QA / "installed_environment_evidence.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
record = {"source": str(SOURCE), "formal_training_started": False,
          "task1_q2_reuses": "nine corrected-v2 runs: three methods x seeds 0,1,2; lr=.001, batch=1000",
          "curve_alignment": "linear interpolation within observed points, 10k grid ending at 500k; one sample SD over training seeds",
          "figures": [{"file": str(path.relative_to(OUTPUT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                      for path in sorted(FIGURES.glob("*.png"))]}
(QA / "figure_provenance.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
print(json.dumps(record, indent=2))

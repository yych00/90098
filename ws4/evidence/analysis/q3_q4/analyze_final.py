"""Audit corrected-v2 WS4 Q3/Q4 and generate reports from saved configurations."""
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

ROOT = Path(__file__).resolve().parent
TASK = ROOT.parents[1]
EXPECTED_PG_HASH = "54d3600cf459dd85ee20b8af00f1aacf47ed022eb0714807e99e1f4ba6794341"
PARAMETERS = ("learning_rate", "batch_steps", "hidden_size", "layer_num", "discount_gamma",
              "use_reward_to_go", "use_baseline", "normalize_advantage", "gae_lambda",
              "baseline_learning_rate", "max_grad_norm")
sys.path.insert(0, str(TASK / "shared"))
from validate_runs import validate
from manifest_paths import load_jobs


def read_json(path):
    return load_jobs(path) if path.name.endswith(".jobs.json") else json.loads(path.read_text(encoding="utf-8"))


def read_run(job):
    audit = validate(job)
    path = Path(job["output_dir"])
    config = read_json(path / "config.json")
    if "batch_steps" in config:
        if config.get("code_hash") != EXPECTED_PG_HASH or config.get("estimator_version") != "corrected-v2":
            raise ValueError(f"Not the frozen corrected-v2 PG estimator: {job['id']}")
    with (path / "evaluations.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return {"id": job["id"], "path": path, "config": config, "audit": audit,
            "x": np.array([float(row["env_steps"]) for row in rows]),
            "y": np.array([float(row["mean_return"]) for row in rows]),
            "summary": read_json(path / "summary.json")}


def method(config):
    if config["gae_lambda"] < 1:
        name = f"GAE (lambda={config['gae_lambda']:g})"
    elif config["use_baseline"]:
        name = "Baseline (previous batches)"
    elif config["use_reward_to_go"]:
        name = "RTG"
    else:
        name = "Vanilla"
    return name + (" + advantage normalization" if config["normalize_advantage"] else "")


def method_note(config, language):
    if config["gae_lambda"] < 1:
        en = ("GAE uses a learned state-value critic, TD residuals and episode-boundary recursion; true "
              "terminations suppress bootstrap while time-limit truncations retain bootstrap. It is an "
              "extension using local discounted advantages without outer gamma^t weighting, rather than "
              "a pure variance-only modification of the full-trajectory REINFORCE objective.")
        zh = ("GAE使用学习到的状态价值critic、TD残差和按回合边界重置的递推；真实终止不bootstrap，时间截断保留bootstrap。"
              "该扩展使用局部折扣优势且没有外层gamma^t权重，不是完整轨迹REINFORCE目标的纯方差改动。")
    elif config["use_baseline"]:
        en = ("The baseline is the mean discounted trajectory return from previous batches only (initially zero). "
              "It is computed before adding the current batch to history, so a single-trajectory batch does "
              "not subtract its own return away.")
        zh = "baseline仅取此前批次的轨迹折扣回报均值，初始为零；先计算当前优势再更新历史，因此单轨迹批次不会减去自身回报。"
    elif config["use_reward_to_go"]:
        en = ("RTG uses sum_{k=t}^{T-1} gamma^k r_k, with discount powers measured from episode origin "
              "(equivalent to outer gamma^t times relative RTG). It shares the full-return discounted weighting.")
        zh = "RTG采用sum_{k=t}^{T-1} gamma^k r_k，折扣幂从回合起点计算，等价于相对RTG乘外层gamma^t，与完整回报的折扣权重一致。"
    else:
        en = "Vanilla repeats the full trajectory's episode-origin discounted return at every time step."
        zh = "Vanilla在每个时间步使用该完整轨迹从回合起点折扣的回报。"
    if config["normalize_advantage"]:
        en += " Batch advantage normalization is enabled and changes the finite-batch estimator."
        zh += " 启用批次优势标准化，会改变有限批次估计量。"
    if config["max_grad_norm"] > 0:
        en += f" Gradient norm clipping is enabled at {config['max_grad_norm']:g}."
        zh += f" 梯度范数裁剪阈值为{config['max_grad_norm']:g}。"
    else:
        en += " Gradient norm clipping is disabled."
        zh += " 不启用梯度范数裁剪。"
    return en if language == "en" else zh


def parameter_table(config, language):
    headings = "| Saved setting | Value |\n|---|---|" if language == "en" else "| 保存的设置 | 值 |\n|---|---|"
    critic = f"{config['baseline_learning_rate']:g}" if config['gae_lambda'] < 1 else "unused"
    rows = [("Environment", config["env_name"]), ("Requested training steps", f"{config['total_steps']:,}"),
            ("Network width x layers", f"{config['hidden_size']} x {config['layer_num']} ReLU"),
            ("Gamma", config["discount_gamma"]),
            ("Actor / critic learning rate", f"{config['learning_rate']:g} / {critic}"),
            ("Minimum batch steps", config["batch_steps"]), ("GAE lambda", config["gae_lambda"]),
            ("Advantage normalization", config["normalize_advantage"]),
            ("Max gradient norm", f"{config['max_grad_norm']:g} (0 disables clipping)"),
            ("Evaluation", f"{config['evaluation_policy']}; {config['eval_episodes']} episodes; seeds "
             f"{config['eval_seed']}-{config['eval_seed']+config['eval_episodes']-1}; every {config['eval_every']:,} steps")]
    note = ("Full config.json files retain all flags and version identifiers." if language == "en" else
            "完整config.json保留全部开关和版本标识；表内列出比较所需设置。")
    return headings + "\n" + "\n".join(f"| {key} | {value} |" for key, value in rows) + "\n\n" + note


def selection_report(phase, runs, language):
    selection = read_json(ROOT / (phase + "_selection.json"))
    if selection.get("code_hash") != EXPECTED_PG_HASH or selection.get("estimator_version") != "corrected-v2":
        raise ValueError("Selection is not from corrected-v2 pilots: " + phase)
    pilots = [read_run(job) for job in read_json(ROOT / (phase + "_pilot.jobs.json"))]
    ranked = sorted(pilots, key=lambda run: (-float(run["y"][-3:].mean()), run["id"]))
    if ranked[0]["id"] != selection["pilot"]["run"]:
        raise ValueError("Selection disagrees with the declared last-three-checkpoint rule")
    for run in runs:
        for key in PARAMETERS:
            if run["config"][key] != selection["parameters"][key]:
                raise ValueError(f"Final parameters differ from pilot selection: {run['id']} {key}")
    header = ("| Pilot | Method | Requested steps | LR | Batch steps | Hidden width x layers | Gamma | Last 3 mean |\n"
              "|---|---|---:|---:|---:|---|---:|---:|" if language == "en" else
              "| 调参运行 | 方法 | 申请步数 | 学习率 | 批次步数 | 隐层宽度x层数 | Gamma | 末3次均值 |\n"
              "|---|---|---:|---:|---:|---|---:|---:|")
    lines = [header]
    for run in ranked:
        c = run["config"]
        if c["seed"] != 0:
            raise ValueError("Pilot ranking is restricted to training seed 0")
        short_id = run['id'].removeprefix(phase + '_pilot_').removesuffix('_s0')
        lines.append(f"| {short_id} | {method(c)} | {c['total_steps']:,} | {c['learning_rate']:g} | "
                     f"{c['batch_steps']} | {c['hidden_size']} x {c['layer_num']} | {c['discount_gamma']:g} | "
                     f"{run['y'][-3:].mean():.1f} |")
    intro = (f"Seed-0 pilots were ranked by the mean of their final three evaluation means, with run ID as "
             f"a deterministic tie-breaker. The selected run was {ranked[0]['id']}. This is exploratory "
             "selection; reusing seed 0 in the formal comparison may favor its selected configuration."
             if language == "en" else
             f"seed0调参按最后三次评估均值排序，同分按运行ID确定顺序；选中{ranked[0]['id']}。"
             "这是探索性选择；正式实验仍含seed0，存在选择偏差。")
    return intro + "\n\n" + "\n".join(lines)


def show_spread(axis, runs, grid, label, color):
    values = np.array([np.interp(grid, run["x"], run["y"]) for run in runs])
    mean, spread = values.mean(axis=0), values.std(axis=0, ddof=1)
    axis.plot(grid, mean, label=label, color=color, linewidth=1.8)
    axis.fill_between(grid, mean - spread, mean + spread, color=color, alpha=.16)
    return values


def describe(values):
    return f"{np.mean(values):.1f} +/- {np.std(values, ddof=1):.1f}"


def sustained_step(run, horizon):
    keep = run["x"] <= horizon
    x, y = run["x"][keep], run["y"][keep]
    return next((int(x[i]) for i in range(2, len(y)) if np.all(y[i-2:i+1] >= 475)), None)


def save_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def verify_seed_group(runs):
    runs.sort(key=lambda run: run["config"]["seed"])
    assert [run["config"]["seed"] for run in runs] == [0, 1, 2]
    first = runs[0]["config"]
    keys = ("env_name", "total_steps", "eval_every", "eval_seed", "eval_episodes", "hidden_size", "layer_num")
    if "batch_steps" in first:
        keys += PARAMETERS
    for run in runs:
        assert all(run["config"][key] == first[key] for key in keys)
        assert run["config"]["eval_seed"] == 10000 and run["config"]["eval_episodes"] == 10


def media_note(cheetah, pg):
    media = {"status": "incomplete", "motion": "missing", "video": "missing", "tensorboard": "missing"}
    path = ROOT / "motion_summary.json"
    if not path.exists():
        return ("No corrected-v2 fixed-episode motion inspection has been generated; no video or motion "
                "validation is claimed.", "尚未生成修正v2的固定回合运动检查；本报告不声称已验证视频或运动表现。", media)
    motion = read_json(path)
    if (motion.get("code_hash") != EXPECTED_PG_HASH
            or motion.get("checkpoint_env_steps") != cheetah[0]["audit"]["training_steps"]
            or motion.get("checkpoint_sha256") != cheetah[0]["audit"]["checkpoint_sha256"]
            or motion.get("training_seed") != 0
            or motion.get("evaluation_seed") != cheetah[0]["config"]["eval_seed"]):
        raise ValueError("Motion summary does not describe the corrected-v2 seed-0 final checkpoint")
    media["motion"] = "passed"
    media["checkpoint_sha256"] = motion["checkpoint_sha256"]
    with (cheetah[0]["path"] / "evaluation_episodes.csv").open(newline="", encoding="utf-8") as stream:
        final = [float(row["return"]) for row in csv.DictReader(stream)
                 if int(row["env_steps"]) == cheetah[0]["audit"]["training_steps"]]
    en = (f"The separately reproduced seed-0 final checkpoint, evaluation seed {motion['evaluation_seed']}, "
          f"ran for {motion['simulated_seconds']:.2f} simulated seconds and returned {motion['episode_return']:.2f}. "
          f"It advanced {motion['forward_displacement_m']:.2f} m; absolute wrapped torso pitch exceeded 45 degrees "
          f"in {100*motion['fraction_abs_pitch_over_45_degrees']:.1f}% of recorded states. This is a descriptive "
          f"posture threshold, not a solved rule. The checkpoint's worst of ten final evaluation episodes "
          f"returned {min(final):.2f}, with {sum(v < 0 for v in final)} negative-return episodes. "
          "A single episode does not establish uniformly reliable control.")
    zh = (f"独立重载seed0最终checkpoint、评估种子{motion['evaluation_seed']}后，同一回合模拟时长"
          f"{motion['simulated_seconds']:.2f}秒，回报{motion['episode_return']:.2f}，前进"
          f"{motion['forward_displacement_m']:.2f}米；记录状态中躯干俯仰绝对值超过45度的比例为"
          f"{100*motion['fraction_abs_pitch_over_45_degrees']:.1f}%。该角度是描述性姿态指标，不是通关规则。"
          f"该checkpoint最终10回合评估的最差回报为{min(final):.2f}，有{sum(v < 0 for v in final)}个负回报回合；"
          "单个回合不能证明控制始终可靠。")
    if (ROOT / "representative_seed0_frames.png").exists():
        decoded = read_json(ROOT / "video_decode_check.json")
        video = ROOT / "representative_seed0.mp4"
        sheet = ROOT / "representative_seed0_frames.png"
        render_path = TASK / "shared/reload_halfcheetah_pg_seed0.json"
        render = read_json(render_path)
        video_sha = hashlib.sha256(video.read_bytes()).hexdigest()
        if (render.get("status") != "passed" or render.get("kind") != "pg"
                or render.get("checkpoint_sha256") != cheetah[0]["audit"]["checkpoint_sha256"]
                or render.get("video_sha256") != video_sha
                or render.get("code_hash") != EXPECTED_PG_HASH
                or render.get("video_seed") != cheetah[0]["config"]["eval_seed"]
                or render.get("training_steps") != cheetah[0]["audit"]["training_steps"]):
            raise ValueError("Actual video render record does not bind this MP4 to the final checkpoint")
        if (decoded.get("status") != "decoded_all_frames"
                or decoded.get("checkpoint_sha256") != cheetah[0]["audit"]["checkpoint_sha256"]
                or decoded.get("sha256") != video_sha
                or decoded.get("code_hash") != EXPECTED_PG_HASH
                or decoded.get("video_seed") != cheetah[0]["config"]["eval_seed"]
                or decoded.get("render_record_sha256") != hashlib.sha256(render_path.read_bytes()).hexdigest()
                or decoded.get("contact_sheet_sha256") != hashlib.sha256(sheet.read_bytes()).hexdigest()):
            raise ValueError("Decoded video/contact sheet is not bound to this final checkpoint")
        media["video"] = "passed"
        media["video_sha256"] = decoded["sha256"]
        media["video_render_record_sha256"] = decoded["render_record_sha256"]
        picture = "\n\n![Fixed-episode sampled frames](../q3_q4/representative_seed0_frames.png)"
        en += picture
        zh += picture
    tb = ROOT / "tensorboard_export_check.json"
    if tb.exists():
        checks = read_json(tb)
        if checks.get("status") != "passed" or any(row.get("code_hash") != EXPECTED_PG_HASH for row in checks["runs"]):
            raise ValueError("TensorBoard export record is not a passed corrected-v2 export")
        check_by_id = {row["id"]: row for row in checks["runs"]}
        if set(check_by_id) != {run["id"] for run in pg + cheetah}:
            raise ValueError("TensorBoard export record does not cover the six current final PG runs")
        for run in pg + cheetah:
            record = check_by_id[run["id"]]
            if (record.get("checkpoint_sha256") != run["audit"]["checkpoint_sha256"]
                    or record.get("evaluations_csv_sha256") != hashlib.sha256((run["path"] / "evaluations.csv").read_bytes()).hexdigest()):
                raise ValueError("TensorBoard export source differs from the current checkpoint/CSV: " + run["id"])
        if (media["video"] != "passed" or check_by_id["halfcheetah_final_s0"].get("video_sha256") != media["video_sha256"]):
            raise ValueError("TensorBoard video preview is not bound to the verified MP4")
        media["tensorboard"] = "passed"
        en += "\n\nTensorBoard scalars were exported after training from raw CSV and checked against event files. "
        zh += "\n\nTensorBoard曲线在训练后由原始CSV导出，并与事件文件核对。"
        en += "This numeric event-file check does not assert a browser inspection."
        zh += "事件文件数值检查不代表已在浏览器验收。"
    if all(media[key] == "passed" for key in ("motion", "video", "tensorboard")):
        media["status"] = "complete"
    else:
        en += " Media evidence is incomplete; this is not a complete submission snapshot."
        zh += " 媒体证据尚不完整，本稿不是完整交付快照。"
    return en, zh, media


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q3-only", action="store_true")
    args = parser.parse_args()
    pg = [read_run(job) for job in read_json(ROOT / "cartpole_final.jobs.json")]
    dqn_jobs = [job for job in read_json(TASK / "shared/dqn_formal.jobs.json")
                if job["id"].startswith("ws3_q1_lr1e-3_seed")]
    assert len(dqn_jobs) == 3
    dqn = [read_run(job) for job in dqn_jobs]
    cheetah = [] if args.q3_only else [read_run(job) for job in read_json(ROOT / "halfcheetah_final.jobs.json")]
    for group in ([pg, dqn] if args.q3_only else [pg, dqn, cheetah]):
        verify_seed_group(group)
    assert all(run["config"]["env_name"] == "CartPole-v1" for run in pg+dqn)
    cap = min(run["config"]["total_steps"] for run in pg+dqn)
    clipped = [{**run, "x":run["x"][run["x"] <= cap], "y":run["y"][run["x"] <= cap]} for run in pg+dqn]
    horizon = min(run["x"][-1] for run in clipped)
    grid = np.unique(np.append(np.arange(0, horizon, 10000), horizon))
    figures = ROOT / "figures"
    figures.mkdir(exist_ok=True)
    pg_name = "PG: " + method(pg[0]["config"])
    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    pg_values = show_spread(axis, clipped[:3], grid, pg_name+" (stochastic)", "#2878b5")
    dqn_values = show_spread(axis, clipped[3:], grid, "DQN (greedy)", "#d65f30")
    axis.axhline(500,color="#555555",linestyle=":",linewidth=1)
    axis.set(xlabel="Training environment steps",ylabel="Mean evaluation episode return",ylim=(0,525),
             title="CartPole-v1: three training seeds, ten evaluation episodes per checkpoint")
    axis.grid(alpha=.25)
    axis.legend(fontsize=8)
    figure.savefig(figures / "q3_pg_vs_dqn.png", dpi=180)
    plt.close(figure)
    rows=[]
    for name,runs,values in ((pg_name,pg,pg_values),("DQN",dqn,dqn_values)):
        for i,run in enumerate(runs):
            rows.append({"algorithm":name,"training_seed":run["config"]["seed"],
                         "actual_training_steps":run["audit"]["training_steps"],"comparison_horizon":int(horizon),
                         "return_at_common_horizon":float(values[i,-1]),"tail3_grid_mean":float(values[i,-3:].mean()),
                         "auc_return":float(np.trapezoid(values[i],grid)/horizon),
                         "sustained_ge475_confirmed_step":sustained_step(run,horizon),
                         "final_saved_checkpoint_return":float(run["y"][-1]),"output_dir":str(run["path"])})
    save_rows(ROOT / "q3_statistics.csv",rows)
    save_rows(ROOT / "q3_aligned_curves.csv",[{"env_steps":int(step),"algorithm":name,"training_seed":run["config"]["seed"],
                 "mean_return":float(values[i,j])} for name,runs,values in ((pg_name,pg,pg_values),("DQN",dqn,dqn_values))
                 for i,run in enumerate(runs) for j,step in enumerate(grid)])
    # Validate pilot selection even for a Q3-only report.
    pilot_cart = {lang:selection_report("cartpole",pg,lang) for lang in ("en","zh")}
    if args.q3_only:
        print(json.dumps({"q3_runs_audited":6,"comparison_horizon":int(horizon),"statistics":rows},indent=2))
        return
    assert all(run["config"]["env_name"] == "HalfCheetah-v5" for run in cheetah)
    pilot_cheetah = {lang:selection_report("halfcheetah",cheetah,lang) for lang in ("en","zh")}
    ch_horizon = min(run["x"][-1] for run in cheetah)
    ch_grid = np.unique(np.append(np.arange(0,ch_horizon,10000),ch_horizon))
    figure,axis=plt.subplots(figsize=(9,5),constrained_layout=True)
    show_spread(axis,cheetah,ch_grid,method(cheetah[0]["config"]),"#399875")
    for run in cheetah:
        axis.plot(run["x"],run["y"],alpha=.5,linewidth=.8,label=f"Training seed {run['config']['seed']}")
    axis.set(xlabel="Training environment steps",ylabel="Mean evaluation episode return",
             title="HalfCheetah-v5: three training seeds, ten stochastic evaluation episodes")
    axis.grid(alpha=.25)
    axis.legend(fontsize=8)
    figure.savefig(figures / "q4_halfcheetah.png",dpi=180)
    plt.close(figure)
    ch_rows=[{"training_seed":run["config"]["seed"],"training_steps":run["audit"]["training_steps"],
              "initial_return":float(run["y"][0]),"final_return":float(run["y"][-1]),
              "tail5_mean":float(run["y"][-5:].mean()),"best_return":float(run["y"].max()),
              "output_dir":str(run["path"])} for run in cheetah]
    save_rows(ROOT / "q4_statistics.csv",ch_rows)
    q12_jobs=read_json(TASK / "ws4/q1_q2/jobs.json")
    assert len(q12_jobs)==45
    q12_runs=[read_run(job) for job in q12_jobs]
    q12_summary=read_json(TASK / "ws4/q1_q2/analysis_summary.json")
    if (q12_summary.get("status") != "completed" or q12_summary.get("completed_runs") != 45
            or q12_summary.get("numerical_failures") != 0
            or q12_summary.get("training_evidence_version") != "corrected_v2"
            or q12_summary.get("code_hashes") != [EXPECTED_PG_HASH]):
        raise ValueError("Q1/Q2 report summary is not the completed 45-run corrected-v2 analysis")
    pg_hashes=sorted({run["config"]["code_hash"] for run in q12_runs+pg+cheetah})
    assert pg_hashes == [EXPECTED_PG_HASH]
    audit={"complete":True,"pg_training_evidence_version":"corrected_v2","corrected_v2_results_included":True,
           "pg_saved_code_hashes":pg_hashes,"formal_pg_runs_passed":len(q12_runs)+len(pg)+len(cheetah),
           "reused_q3_dqn_runs_passed":len(dqn),"q3_q4_runs_passed":len(pg)+len(dqn)+len(cheetah),
           "comparison_horizon":int(horizon),"q1_q2":q12_summary,
           "q1_q2_run_audit":[run["audit"] for run in q12_runs],
           "q3_q4_audit":[run["audit"] for run in pg+dqn+cheetah]}
    assert audit["formal_pg_runs_passed"]==51 and audit["reused_q3_dqn_runs_passed"]==3
    motion_en,motion_zh,media=media_note(cheetah,pg)
    audit["media_validation"] = media
    audit["complete"] = media["status"] == "complete"
    pg_auc=[row["auc_return"] for row in rows if row["algorithm"]==pg_name]
    dqn_auc=[row["auc_return"] for row in rows if row["algorithm"]=="DQN"]
    higher=pg_name if np.mean(pg_auc)>np.mean(dqn_auc) else "DQN" if np.mean(dqn_auc)>np.mean(pg_auc) else "Both algorithms"
    table=("| Algorithm | Training seeds | Return at common horizon | Last 3 grid points | AUC return |\n"
           "|---|---:|---:|---:|---:|\n"
           f"| {pg_name} | 3 | {describe(pg_values[:,-1])} | {describe(pg_values[:,-3:].mean(axis=1))} | {describe(pg_auc)} |\n"
           f"| DQN | 3 | {describe(dqn_values[:,-1])} | {describe(dqn_values[:,-3:].mean(axis=1))} | {describe(dqn_auc)} |")
    success={name:[r['sustained_ge475_confirmed_step'] for r in rows if r['algorithm']==name and r['sustained_ge475_confirmed_step'] is not None] for name in (pg_name,"DQN")}
    successes_en="; ".join(f"{name}: {len(vals)}/3 reached, median among reached " + (f"{np.median(vals):,.0f} steps" if vals else "not reached") for name,vals in success.items())
    successes_zh="；".join(f"{name}：{len(vals)}/3达标，达标种子的中位确认步数" + (f"{np.median(vals):,.0f}" if vals else "未达到") for name,vals in success.items())
    pg_steps=", ".join(str(run["audit"]["training_steps"]) for run in pg)
    d= dqn[0]["config"]
    dc=(f"DQN: hidden width {d['hidden_size']}, {d['layer_num']} ReLU layers, gamma {d['discount_factor']}, "
        f"learning rate {d['learning_rate']}, replay {d['memory_size']}, minibatch {d['batch_size']}, "
        f"warm-up {d['learning_start']}, target synchronization every {d['target_update_freq']} updates, "
        f"epsilon {d['epsilon']} with per-completed-episode decay {d['epsilon_decay']} to {d['epsilon_min']}.")
    cheetah_table="| Seed | Actual training steps | Initial return | Final return | Last 5 mean |\n|---|---:|---:|---:|---:|\n"+"\n".join(f"| {r['training_seed']} | {r['training_steps']:,} | {r['initial_return']:.1f} | {r['final_return']:.1f} | {r['tail5_mean']:.1f} |" for r in ch_rows)
    q3_en=f"""## Q3: Policy Gradient versus DQN on CartPole

All PG data in this section were generated by corrected-v2 ({pg[0]['config']['version']}; saved code hash {EXPECTED_PG_HASH}). The three original WS3 DQN baseline runs are reused as recorded; PG is newly trained.

{pilot_cart['en']}

The final PG method is {method(pg[0]['config'])}. {method_note(pg[0]['config'],'en')}

{parameter_table(pg[0]['config'],'en')}

Formal training seeds are 0, 1 and 2. {dc} DQN evaluates greedily; PG samples its policy. Each checkpoint averages ten evaluation episodes with environment and sampling RNG isolated from training. Fixed starts aid controlled comparison but do not test unseen reset seeds.

PG actually completed {pg_steps} training steps after complete final batches. Comparisons exclude observations after the common requested budget {cap:,}; all six seeds support a common horizon of {int(horizon):,} steps. Linear interpolation stays within observed ranges. AUC is the trapezoidal integral on the common 10,000-step grid including its endpoint, divided by the horizon. Bands are one sample SD across three training seeds, not confidence intervals.

![PG and DQN](../q3_q4/figures/q3_pg_vs_dqn.png)

{table}

The descriptive threshold requires three consecutive observed evaluation means >=475, confirmed at the third checkpoint: {successes_en}. Missing times are not imputed. AUC measures performance over the whole budget, whereas confirmation measures when a specified level is reached; later decline is possible.

{higher} has the higher mean AUC in these configurations. DQN reuses replay transitions; PG collects fresh on-policy trajectories. Different evaluation action selection and selected parameters mean this does not isolate update rules. Three training seeds limit uncertainty estimates, and concurrent wall times are not a controlled speed benchmark.
"""
    q3_zh=f"""## 问题3：CartPole上的PG与DQN

本节PG数据全部由corrected-v2（{pg[0]['config']['version']}；源码哈希{EXPECTED_PG_HASH}）新训练产生。DQN复用WS3原有三个基准运行的实测数据。

{pilot_cart['zh']}

最终PG方法为{method(pg[0]['config'])}。{method_note(pg[0]['config'],'zh')}

{parameter_table(pg[0]['config'],'zh')}

正式训练种子为0、1、2。DQN保存参数如下：{dc} DQN使用贪心评估，PG仍采样策略；每检查点评估10个回合，环境和采样随机流与训练隔离。固定初始种子便于受控比较，但不代表未见初始种子的泛化测试。

PG按完整批次结束，实际步数依次为{pg_steps}。比较剔除共同申请预算{cap:,}步之后的评估点，六个运行均有观测支持的共同终点为{int(horizon):,}步；只在观测区间内插值，不外推。AUC为共同1万步网格（含终点）上的梯形积分除以共同区间长度。阴影为三个训练种子间的样本标准差，不是置信区间。

![PG与DQN](../q3_q4/figures/q3_pg_vs_dqn.png)

{table}

描述性阈值要求连续三个真实检查点评估均值≥475，第三点为确认时刻：{successes_zh}。未达标者不插补时间；中位数仅针对达标种子。AUC反映全预算表现，确认步数衡量何时达到指定水平，确认后仍可能退化。

当前配置中{higher}的平均AUC更高。DQN复用经验回放，PG采集新的同策略轨迹；动作选择和调参配置不同，因此不是只隔离更新规则的对照。三个训练种子的不确定性估计有限，并发耗时不能作严格计算速度比较。
"""
    q4_en=f"""## Q4: HalfCheetah controller

All pilots and final policies in this section use the corrected-v2 code hash {EXPECTED_PG_HASH}.

{pilot_cheetah['en']}

The selected final method is {method(cheetah[0]['config'])}. {method_note(cheetah[0]['config'],'en')}

{parameter_table(cheetah[0]['config'],'en')}

Formal seeds are 0, 1 and 2. A ReLU Gaussian actor uses raw sampled actions in its policy loss, sums log probabilities across action components and clips only environment execution to action bounds. Evaluations use stochastic actions, ten fixed reset/sampling seeds, and RNG isolated from training; complete final batches may exceed the requested training budget. Critic and normalization settings are disclosed above; optional GAE and normalization are extensions beyond the core Q1/Q2 estimators.

![HalfCheetah learning curve](../q3_q4/figures/q4_halfcheetah.png)

Across training seeds, initial return was {describe([r['initial_return'] for r in ch_rows])}; final return was {describe([r['final_return'] for r in ch_rows])}; the mean of each seed's final five checkpoints was {describe([r['tail5_mean'] for r in ch_rows])}. These are measured returns, not percentages or a solved threshold. Three seeds and exploratory seed-0 selection limit generalization.

{cheetah_table}

### Fixed-episode inspection and visualization

{motion_en}
"""
    q4_zh=f"""## 问题4：HalfCheetah控制器

本节调参和正式策略全部使用corrected-v2源码哈希{EXPECTED_PG_HASH}。

{pilot_cheetah['zh']}

最终方法为{method(cheetah[0]['config'])}。{method_note(cheetah[0]['config'],'zh')}

{parameter_table(cheetah[0]['config'],'zh')}

正式种子为0、1、2。ReLU高斯actor保留原始采样动作计算损失，跨动作维度求和log probability，仅环境执行动作裁剪至动作边界。每次评估使用10个固定环境/采样种子和随机策略，随机流与训练隔离；完整末批次可能超过申请预算。critic及标准化设置见上表；GAE、优势标准化属于核心Q1/Q2估计器之外的扩展。

![HalfCheetah曲线](../q3_q4/figures/q4_halfcheetah.png)

跨训练种子的初始回报为{describe([r['initial_return'] for r in ch_rows])}，最终回报为{describe([r['final_return'] for r in ch_rows])}；各种子末五次评估均值的跨种子统计为{describe([r['tail5_mean'] for r in ch_rows])}。这些是实测回报，不是百分比或预设通关标准。三个种子和探索性的seed0选择限制泛化结论。

{cheetah_table}

### 固定回合检查与可视化

{motion_zh}
"""
    (ROOT / "report_en.md").write_text(q3_en+"\n"+q4_en,encoding="utf-8")
    (ROOT / "report_zh.md").write_text(q3_zh+"\n"+q4_zh,encoding="utf-8")
    reports=TASK / "ws4/reports"
    reports.mkdir(exist_ok=True)
    (reports / "audit.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
    for lang,suffix,sections in (("en","Report_EN",q3_en+"\n"+q4_en),("zh","Analysis_ZH",q3_zh+"\n"+q4_zh)):
        q12=(TASK / "ws4/q1_q2" / f"report_{lang}.md").read_text(encoding="utf-8")
        q12=q12.split("\n",1)[1].lstrip().replace("](figures/", "](../q1_q2/figures/")
        title="Workshop 4: Policy Gradient - Task 2 (corrected v2)" if lang=="en" else "Workshop 4：策略梯度Task 2（修正v2）"
        if not audit["complete"]:
            title += " (INCOMPLETE MEDIA SNAPSHOT)" if lang == "en" else "（媒体证据未齐）"
        (reports / f"Workshop4_Task2_{suffix}.md").write_text(f"# {title}\n\nAuthors: Zhengnan Cao, Yuchen Yang\n\n"+q12+"\n\n"+sections,encoding="utf-8")
    print(json.dumps({"q3_q4_audited":9,"formal_pg_runs_passed":51,"q1_q2_complete":True,"horizon":horizon}))


if __name__ == "__main__":
    main()

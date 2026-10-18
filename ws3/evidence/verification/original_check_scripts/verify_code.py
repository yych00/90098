import ast
import contextlib
import csv
import hashlib
import importlib
import importlib.metadata
import importlib.util
import io
import json
import os
from pathlib import Path
import pickle
import random
import shutil
import subprocess
import sys
import tokenize
import zipfile

sys.dont_write_bytecode = True
import gymnasium as gym
import numpy as np
import torch

QA = Path(__file__).resolve().parent
ROOT = QA.parents[1]
TARGET = ROOT / "outputs/final_report_v3/Workshop3-Code-Cao-Yang"
EARLY = ROOT / "DQN"
FROZEN = ROOT / "task2_workspace/corrected_v2/shared/historical_v1/ws3/DQN"
FILES = ("run_dqn.py", "dqn/agent.py", "dqn/network.py", "utils/util.py", "utils/logger.py")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def comments(path):
    return [token.string for token in tokenize.generate_tokens(io.StringIO(path.read_text(encoding="utf-8")).readline)
            if token.type == tokenize.COMMENT]


def interfaces(path):
    found = {}
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            found[node.name] = None
            for method in node.body:
                if isinstance(method, ast.FunctionDef):
                    found[node.name + "." + method.name] = [a.arg for a in method.args.posonlyargs + method.args.args]
        elif isinstance(node, ast.FunctionDef):
            found[node.name] = [a.arg for a in node.args.posonlyargs + node.args.args]
    return found


def load_agent(directory):
    for name in list(sys.modules):
        if name == "dqn" or name.startswith("dqn."):
            del sys.modules[name]
    sys.path.insert(0, str(directory))
    try:
        return importlib.import_module("dqn.agent")
    finally:
        sys.path.pop(0)


def max_state_error(first, second):
    assert set(first) == set(second)
    return max(float((first[key] - second[key]).abs().max()) for key in first)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def cli(directory, output, extra, log_name, tensorboard=False):
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1",
           "PYTHONPATH": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy"}
    command = [sys.executable, str(directory / "run_dqn.py"), "--env_name", "CartPole-v1",
               "--device", "cpu", "--seed", "17", "--output_dir", str(output),
               "--eval_seed", "10000", "--eval_episodes", "10", "--eval_every", "100",
               "--max_ep_len", "25", "--memory_size", "64", "--learning_start", "8",
               "--batch_size", "8", "--target_update_freq", "10", "--hidden_size", "16",
               "--layer_num", "2", "--epsilon", "0.8", "--epsilon_min", "0.02",
               "--epsilon_decay", "0.9", "--tensorboard" if tensorboard else "--no-tensorboard", *extra]
    if output.exists():
        config = json.loads((output / "config.json").read_text(encoding="utf-8"))
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        assert summary["status"] == "complete", "Preserve incomplete QA evidence"
        digest = hashlib.sha256()
        for name in FILES[:3] if directory == FROZEN else FILES:
            digest.update(name.encode())
            digest.update((directory / name).read_bytes())
        assert config["code_hash"] == digest.hexdigest(), "QA checkpoint source changed"
        requested = {value[2:]: command[index + 1] for index, value in enumerate(command[:-1])
                     if value.startswith("--") and not command[index + 1].startswith("--")}
        for key, expected in requested.items():
            actual = config[key]
            assert (float(actual) == float(expected) if isinstance(actual, (int, float)) else str(actual) == expected), key
        assert config["tensorboard"] is tensorboard
        return command
    result = subprocess.run(command, cwd=QA, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace", timeout=180)
    (QA / log_name).write_text(result.stdout, encoding="utf-8")
    assert result.returncode == 0, result.stdout
    return command


def main():
    torch.set_num_threads(1)
    report = {"status": "running", "template_reference": str(EARLY),
              "original_teacher_DQN_zip_available": False,
              "template_claim": "Checked against existing early source, not an unavailable unmodified teacher archive.",
              "files": {}, "checks": {}}
    early_before = {str(EARLY / name): sha(EARLY / name) for name in FILES}
    frozen_before = {str(FROZEN / name): sha(FROZEN / name) for name in FILES[:3]}
    target_before = {str(path.relative_to(TARGET)): sha(path) for path in TARGET.rglob("*") if path.is_file()}
    for relative in FILES:
        original, submitted = EARLY / relative, TARGET / "DQN" / relative
        old_comments, new_comments = comments(original), comments(submitted)
        remaining = iter(new_comments)
        assert all(any(item == wanted for item in remaining) for wanted in old_comments), relative
        original_interfaces, submitted_interfaces = interfaces(original), interfaces(submitted)
        for name, arguments in original_interfaces.items():
            assert name in submitted_interfaces, (relative, name)
            if arguments is not None:
                assert submitted_interfaces[name][:len(arguments)] == arguments, (relative, name)
        for node in ast.walk(ast.parse(submitted.read_text(encoding="utf-8"))):
            assert not isinstance(node, ast.Pass)
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert not node.value.startswith(("D:\\", "D:/", "C:\\Users\\", "E:\\")), (relative, node.value)
        report["files"][relative] = {
            "early_sha256": sha(original), "submission_sha256": sha(submitted),
            "original_comments": len(old_comments), "retained_comments_in_order": True,
            "todo_comments_retained": sum("TODO" in text for text in old_comments),
            "original_interfaces": list(original_interfaces), "missing_interfaces": [],
            "added_interfaces": [name for name in submitted_interfaces if name not in original_interfaces]}
    assert all((EARLY / name).read_bytes() == (ROOT / "Workshop3-Code-Cao-Yang/DQN" / name).read_bytes() for name in FILES)
    with zipfile.ZipFile(ROOT / "Workshop3-Code-Cao-Yang.zip") as archive:
        report["early_submission_zip_agrees_with_five_available_python_files"] = all(
            archive.read("Workshop3-Code-Cao-Yang/DQN/" + name) == (EARLY / name).read_bytes() for name in FILES)

    reference = load_agent(FROZEN)
    submitted = load_agent(TARGET / "DQN")
    train_env = gym.make("CartPole-v1")
    settings = {"device": "cpu", "hidden_size": 64, "layer_num": 3, "gamma": .99,
                "epsilon": .8, "epsilon_min": .02, "epsilon_decay": .9,
                "memory_size": 64, "batch_size": 8, "lr": .001, "target_update_freq": 2, "seed": 11}
    torch.manual_seed(777)
    ref = reference.DQNAgent(train_env, **settings)
    torch.manual_seed(777)
    port = submitted.DQNAgent(train_env, **settings)
    assert max_state_error(ref.q_net.state_dict(), port.q_net.state_dict()) == 0
    generator = np.random.default_rng(31415)
    states = torch.tensor(generator.normal(size=(8, 4)), dtype=torch.float32)
    next_states = torch.tensor(generator.normal(size=(8, 4)), dtype=torch.float32)
    batch = (states, torch.tensor([0, 1, 1, 0, 1, 0, 1, 0]),
             torch.tensor([10000, -20000, 30000, -40000, 50000, -60000, 70000, -80000], dtype=torch.float32).reshape(-1, 1),
             next_states, torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.float32).reshape(-1, 1))
    initial_target = {key: value.clone() for key, value in port.target_net.state_dict().items()}
    first_reference, first_port = ref.update(batch), port.update(batch)
    scalar_errors = {key: abs(first_reference[key] - first_port[key]) for key in ("critic_loss", "q_values", "target_values", "gradient")}
    gradient_error = max(float((a.grad - b.grad).abs().max()) for a, b in zip(ref.q_net.parameters(), port.q_net.parameters()))
    parameter_error = max_state_error(ref.q_net.state_dict(), port.q_net.state_dict())
    assert max(scalar_errors.values()) <= 1e-7 and gradient_error <= 1e-7 and parameter_error <= 1e-7
    assert first_port["gradient"] > 10
    clipped_norm = float(torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in port.q_net.parameters()])))
    assert clipped_norm <= 10.00001
    assert max_state_error(initial_target, port.target_net.state_dict()) == 0
    assert max_state_error(port.q_net.state_dict(), port.target_net.state_dict()) > 0
    ref.update(batch)
    port.update(batch)
    assert max_state_error(port.q_net.state_dict(), port.target_net.state_dict()) == 0
    assert max_state_error(ref.q_net.state_dict(), port.q_net.state_dict()) <= 1e-7
    report["checks"]["fixed_batch_reference"] = {"status": "passed", "scalar_absolute_errors": scalar_errors,
        "post_clip_gradient_max_abs_error": gradient_error, "first_update_parameter_max_abs_error": parameter_error,
        "pre_clip_norm": first_port["gradient"], "post_clip_norm": clipped_norm,
        "target_unchanged_after_update_1": True, "target_equals_online_after_update_2": True}

    for index in range(90):
        state, next_state = generator.normal(size=(2, 4))
        transition = (state, index % 2, float(index), next_state, float(index % 3 == 0))
        ref.replay_buffer.add(*transition)
        port.replay_buffer.add(*transition)
    replay_error = 0.
    for _ in range(3):
        replay_error = max(replay_error, max(float((a - b).abs().max()) for a, b in zip(ref.replay_buffer.sample(8), port.replay_buffer.sample(8))))
    assert replay_error == 0
    assert len(ref.replay_buffer) == len(port.replay_buffer) == 64
    observation, _ = train_env.reset(seed=5)
    legal_actions = [port.sample_action(observation) for _ in range(200)]
    assert all(isinstance(a, int) and train_env.action_space.contains(a) for a in legal_actions)
    assert port.epsilon == settings["epsilon"]
    port.epsilon = 0
    assert all(port.sample_action(observation) == port.greedy_action(observation) for _ in range(20))
    port.epsilon, port.epsilon_decay, port.epsilon_min = .1, .5, .02
    epsilon_trace = []
    for _ in range(5):
        port.epsilon_step()
        epsilon_trace.append(port.epsilon)
    np.testing.assert_allclose(epsilon_trace, [.05, .025, .02, .02, .02], rtol=0, atol=0)
    report["checks"]["replay_and_exploration"] = {"status": "passed", "wrapped_replay_samples_max_abs_error": replay_error,
        "capacity": 64, "transitions_added": 90, "legal_actions_checked": len(legal_actions),
        "action_sampling_does_not_decay_epsilon": True, "epsilon_trace": epsilon_trace}

    sys.path.insert(0, str(TARGET / "DQN"))
    spec = importlib.util.spec_from_file_location("v3_runner", TARGET / "DQN/run_dqn.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    eval_env = gym.make("CartPole-v1", max_episode_steps=25)
    streams_before = (pickle.dumps(random.getstate()), pickle.dumps(np.random.get_state()), torch.get_rng_state().clone(),
                      pickle.dumps(port.action_rng.getstate()), pickle.dumps(port.replay_buffer.rng.getstate()),
                      pickle.dumps(train_env.unwrapped.np_random.bit_generator.state), train_env.unwrapped.state.copy())
    values1 = runner.evaluate(port, eval_env, list(range(10000, 10010)))
    values2 = runner.evaluate(port, eval_env, list(range(10000, 10010)))
    streams_after = (pickle.dumps(random.getstate()), pickle.dumps(np.random.get_state()), torch.get_rng_state().clone(),
                     pickle.dumps(port.action_rng.getstate()), pickle.dumps(port.replay_buffer.rng.getstate()),
                     pickle.dumps(train_env.unwrapped.np_random.bit_generator.state), train_env.unwrapped.state.copy())
    assert values1 == values2 and len(values1) == 10
    assert all(a == b for index, (a, b) in enumerate(zip(streams_before, streams_after)) if index not in (2, 6))
    assert torch.equal(streams_before[2], streams_after[2]) and np.array_equal(streams_before[6], streams_after[6])
    report["checks"]["evaluation_rng_isolation"] = {"status": "passed", "greedy_episodes": 10,
        "repeat_returns": values1, "global_and_agent_random_streams_unchanged": True,
        "training_environment_state_unchanged": True}
    train_env.close()
    eval_env.close()

    copy = QA / "independent_copy"
    if copy.exists():
        assert all(sha(copy / "DQN" / name) == sha(TARGET / "DQN" / name) for name in FILES)
        for name in ("README.md", "requirements.txt"):
            shutil.copyfile(TARGET / name, copy / name)
        assert {str(p.relative_to(copy)): sha(p) for p in copy.rglob("*") if p.is_file()} == target_before
    else:
        shutil.copytree(TARGET, copy)
    command = cli(copy / "DQN", QA / "cpu_copy_run", ["--total_steps", "200"], "cpu_copy_run.log")
    cli(FROZEN, QA / "cpu_frozen_reference_run", ["--total_steps", "200"], "cpu_frozen_reference_run.log")
    new_checkpoint = torch.load(QA / "cpu_copy_run/checkpoint.pt", map_location="cpu", weights_only=False)
    old_checkpoint = torch.load(QA / "cpu_frozen_reference_run/checkpoint.pt", map_location="cpu", weights_only=False)
    online_error = max_state_error(new_checkpoint["q_net"], old_checkpoint["q_net"])
    target_error = max_state_error(new_checkpoint["target_net"], old_checkpoint["target_net"])
    assert online_error <= 1e-7 and target_error <= 1e-7
    new_summary = json.loads((QA / "cpu_copy_run/summary.json").read_text(encoding="utf-8"))
    old_summary = json.loads((QA / "cpu_frozen_reference_run/summary.json").read_text(encoding="utf-8"))
    assert new_summary["status"] == "complete" and new_summary["actual_training_steps"] == 200
    assert new_summary["updates"] == old_summary["updates"] == 193
    assert new_summary["final_epsilon"] == old_summary["final_epsilon"]
    for filename in ("metrics.csv", "evaluations.csv", "evaluation_episodes.csv"):
        left, right = read_csv(QA / "cpu_copy_run" / filename), read_csv(QA / "cpu_frozen_reference_run" / filename)
        assert left == right, (filename, left, right)
    assert new_checkpoint["config"]["bootstrap_mask"] == "terminated_only"
    metrics = read_csv(QA / "cpu_copy_run/metrics.csv")
    truncations_in_200_step_run = sum(int(row["truncated"]) for row in metrics)
    report["checks"]["independent_copy_cpu_200_steps"] = {"status": "passed", "command": command,
        "working_directory": str(QA), "training_steps": 200, "gradient_updates": 193,
        "online_weights_max_abs_error_vs_frozen": online_error,
        "target_weights_max_abs_error_vs_frozen": target_error, "all_training_and_evaluation_csv_rows_equal": True,
        "time_limit_truncations_observed": truncations_in_200_step_run, "formal_training": False}
    cli(copy / "DQN", QA / "legacy_episode_run", ["--episodes", "2", "--max_ep_len", "5"], "legacy_episode_run.log")
    cli(FROZEN, QA / "truncation_frozen_reference_run", ["--total_steps", "10", "--max_ep_len", "5"], "truncation_frozen_reference_run.log")
    legacy = json.loads((QA / "legacy_episode_run/summary.json").read_text(encoding="utf-8"))
    assert legacy["actual_training_steps"] == 10 and legacy["completed_training_episodes"] == 2
    truncated_rows = read_csv(QA / "legacy_episode_run/metrics.csv")
    assert all(int(row["truncated"]) == 1 and int(row["terminated"]) == 0 for row in truncated_rows)
    legacy_state = torch.load(QA / "legacy_episode_run/checkpoint.pt", map_location="cpu", weights_only=False)
    truncation_reference = torch.load(QA / "truncation_frozen_reference_run/checkpoint.pt", map_location="cpu", weights_only=False)
    truncation_error = max_state_error(legacy_state["q_net"], truncation_reference["q_net"])
    assert truncation_error <= 1e-7
    report["checks"]["legacy_episode_interface_and_truncation_bootstrap"] = {
        "status": "passed", "episodes": 2, "steps": 10, "all_episodes_time_limit_truncated": True,
        "online_weights_max_abs_error_vs_frozen": truncation_error}

    tensorboard_copy = QA / "independent_copy_tensorboard"
    if tensorboard_copy.exists():
        assert {str(p.relative_to(tensorboard_copy)): sha(p) for p in tensorboard_copy.rglob("*") if p.is_file()} == target_before
    else:
        shutil.copytree(TARGET, tensorboard_copy)
    tb_command = cli(tensorboard_copy / "DQN", QA / "tensorboard_cpu_run", ["--total_steps", "200"],
                     "tensorboard_cpu_run.log", tensorboard=True)
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    accumulator = EventAccumulator(str(QA / "tensorboard_cpu_run/tensorboard"), size_guidance={"scalars": 0}).Reload()
    tb_metrics = read_csv(QA / "tensorboard_cpu_run/metrics.csv")
    tb_evaluations = read_csv(QA / "tensorboard_cpu_run/evaluations.csv")
    scalar_counts, event_error = {}, 0.
    for tag, field in (("Network_Loss", "critic_loss"), ("Q_Values", "q_values"), ("Target_Q_Values", "target_values")):
        records = accumulator.Scalars(tag)
        assert [event.step for event in records] == list(range(8, 201))
        scalar_counts[tag] = len(records)
        lookup = {event.step: event.value for event in records}
        for row in tb_metrics:
            if row[field]:
                error = abs(lookup[int(row["env_steps"])] - float(row[field]))
                assert error <= 1e-5
                event_error = max(event_error, error)
    records = accumulator.Scalars("Eval_Returns")
    assert [event.step for event in records] == [int(row["env_steps"]) for row in tb_evaluations]
    scalar_counts["Eval_Returns"] = len(records)
    for event, row in zip(records, tb_evaluations):
        error = abs(event.value - float(row["mean_return"]))
        assert error <= 1e-5
        event_error = max(event_error, error)
    records = accumulator.Scalars("_Epsilon")
    assert [event.step for event in records] == list(range(len(tb_metrics)))
    scalar_counts["_Epsilon"] = len(records)
    for event, row in zip(records, tb_metrics):
        error = abs(event.value - float(row["epsilon"]))
        assert error <= 1e-5
        event_error = max(event_error, error)
    tb_state = torch.load(QA / "tensorboard_cpu_run/checkpoint.pt", map_location="cpu", weights_only=False)
    tb_online_error = max_state_error(tb_state["q_net"], old_checkpoint["q_net"])
    assert tb_online_error <= 1e-7
    assert read_csv(QA / "tensorboard_cpu_run/metrics.csv") == read_csv(QA / "cpu_copy_run/metrics.csv")
    report["checks"]["independent_copy_tensorboard_cpu_run"] = {
        "status": "passed", "command": tb_command, "training_steps": 200,
        "event_scalar_counts": scalar_counts, "event_data_max_abs_error_vs_csv": event_error,
        "all_training_and_evaluation_scalar_steps_checked": True,
        "online_weights_max_abs_error_vs_frozen": tb_online_error,
        "all_training_rows_equal_to_tensorboard_disabled_run": True,
        "browser_ui_inspection_claimed": False}

    assert all(sha(Path(path)) == digest for path, digest in early_before.items())
    assert all(sha(Path(path)) == digest for path, digest in frozen_before.items())
    assert target_before == {str(path.relative_to(TARGET)): sha(path) for path in TARGET.rglob("*") if path.is_file()}
    submitted_files = sorted(path.relative_to(TARGET).as_posix() for path in TARGET.rglob("*") if path.is_file())
    assert len(submitted_files) == 7
    assert not any(path.suffix in {".pyc", ".pt", ".zip"} or "__pycache__" in path.parts for path in TARGET.rglob("*"))
    report.update(status="passed", submission_directory=str(TARGET), submission_files=submitted_files,
                  final_submission_file_sha256={name.replace("\\", "/"): digest for name, digest in target_before.items()},
                  original_comment_tokens_preserved=sum(row["original_comments"] for row in report["files"].values()),
                  original_todo_tokens_preserved=sum(row["todo_comments_retained"] for row in report["files"].values()),
                  original_class_count=sum(isinstance(node, ast.ClassDef) for name in FILES for node in ast.walk(ast.parse((EARLY / name).read_text(encoding="utf-8")))),
                  original_callable_count=sum(isinstance(node, ast.FunctionDef) for name in FILES for node in ast.walk(ast.parse((EARLY / name).read_text(encoding="utf-8")))),
                  early_and_frozen_sources_unchanged=True, submission_unchanged_by_verification=True,
                  no_cache_checkpoints_logs_or_archives_in_submission=True,
                  runtime={name: importlib.metadata.version(name) for name in ("torch", "numpy", "gymnasium", "tensorboardX", "imageio")},
                  maximum_verified_numerical_error=max(max(scalar_errors.values()), gradient_error, parameter_error, replay_error, online_error, target_error, truncation_error, tb_online_error),
                  semantics="Matched frozen DQN fixed-batch updates, clipped gradients, replay after wrap, target synchronization and all 200-step CPU training/evaluation rows; source bytes differ to retain template comments.")
    (QA / "verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

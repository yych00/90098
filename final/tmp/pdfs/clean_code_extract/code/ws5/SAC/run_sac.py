"""Teacher-layout SAC runner with reproducible, commented performance options."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import time
import gymnasium as gym
import numpy as np
import torch
from sac.agent import SACAgent
from sac.cuda_update import CudaUpdate, CudaPolicy
from utils.logger import Logger
from utils.evaluation import EvaluationPool


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def append_csv(path, row):
    exists = path.exists()
    with path.open('a', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def source_hashes():
    base = Path(__file__).resolve().parent
    return {p.relative_to(base).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(base.rglob('*.py'))}


@torch.no_grad()
def record_video(agent, args, output, step):
    import imageio.v2 as imageio
    video_env = gym.make(args.env_name, render_mode='rgb_array')
    noise = np.random.default_rng(args.eval_seed)
    obs, _ = video_env.reset(seed=args.eval_seed)
    length = 0
    try:
        # Stream video frames instead of retaining 1000 frames in RAM; use independent noise.
        with imageio.get_writer(str(output/f'policy_step{step}.mp4'),
                                fps=video_env.metadata.get('render_fps',20)) as writer:
            done = False
            while not done:
                normal = agent.actor(torch.as_tensor(obs,dtype=torch.float32,
                                                       device=agent.device)).base_dist.base_dist
                z = torch.as_tensor(noise.standard_normal(agent.act_dim),dtype=torch.float32,
                                    device=agent.device)
                action = agent._scale_to_env(torch.tanh(normal.loc+normal.scale*z)).squeeze(0).cpu().numpy()
                obs, _, terminated, truncated, _ = video_env.step(action)
                writer.append_data(video_env.render())
                length += 1
                done = terminated or truncated
    finally:
        video_env.close()
    write_json(output/f'policy_step{step}.json',{'env_steps':step,'seed':args.eval_seed,
               'policy':'stochastic','frames':length,'visually_reviewed':False})
    return length


def train(args):
    torch.set_num_threads(args.torch_threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    log_dir = Path(args.output_dir or f'logs/{args.env_name}/{args.exp_name or "SAC"}_{time.time_ns()}')
    if log_dir.exists() and any(log_dir.iterdir()):
        raise FileExistsError(f'Refusing to overwrite nonempty run: {log_dir}')
    log_dir.mkdir(parents=True, exist_ok=True)
    config = {**vars(args), 'python_source_sha256': source_hashes(),
              'torch': torch.__version__, 'gymnasium': gym.__version__,
              'evaluation_policy': 'stochastic tanh Gaussian; independent per-episode NumPy noise',
              'checkpoint_purpose': 'evaluation only; replay/environment state not saved',
              'update_to_transition_ratio': 1}
    write_json(log_dir/'config.json', config)
    optimized = not args.reference
    logger = Logger(log_dir/'tensorboard', optimized=optimized)
    env = gym.make(args.env_name, render_mode='human' if args.render else None)
    env.action_space.seed(args.seed)
    evaluation = EvaluationPool(gym, args.env_name, args.eval_episodes, args.eval_seed)
    agent = SACAgent(env, device=args.device, hidden_size=args.hidden_size, layer_num=args.layer_num,
                     actor_lr=args.actor_learning_rate, critic_lr=args.critic_learning_rate,
                     gamma=args.discount_gamma, tau=args.tau, temperature=args.temperature,
                     memory_size=args.memory_size, optimized=optimized, fused_optimizer=args.fused_optimizer,
                     capturable=args.cuda_graph)
    graph_update = None
    policy_graph = None
    total_envsteps = updates = episode = 0
    eval_steps = 0
    evaluation_seconds = 0.0
    video_steps = 0
    phase = dict(action_seconds=0.0, env_seconds=0.0, update_seconds=0.0, logging_seconds=0.0)
    next_eval = args.eval_every
    next_video = args.video_every
    window = None
    window_count = 0
    window_start = None
    learning_started_at = None
    metric_names = ['actor_loss','critic_loss1','critic_loss2','q_value','target_value','entropy','temperature']
    tags = ['Actor Loss','Critic Loss 1','Critic Loss 2','Q Value','Target Value','Entropy','Temperature']
    if args.start_gate:
        # Benchmark only: synchronize workers after initialization for fair concurrency timing.
        (log_dir/'ready').touch()
        deadline=time.perf_counter()+120
        while not Path(args.start_gate).exists():
            if time.perf_counter()>deadline:
                raise TimeoutError('Benchmark start gate timed out')
            time.sleep(0.05)
    started = time.perf_counter()
    started_unix=time.time()
    learning_started_unix=None
    status, error = 'completed', None
    state, _ = env.reset(seed=args.seed)
    episode_reward = 0.0

    def emit_metrics(step):
        nonlocal window, window_count, window_start
        if not window_count:
            return
        # Copy window means to the CPU once; record the window boundaries and update count.
        # These are window means, not per-step losses; update frequency, data and RNG are unchanged.
        values = (window/window_count).cpu().tolist()
        row = {'env_steps': step, 'window_start_step': window_start,
               'window_updates': window_count, **dict(zip(metric_names, values))}
        if not np.isfinite(values).all():
            raise FloatingPointError('Non-finite SAC metrics')
        append_csv(log_dir/'metrics.csv', row)
        for tag, value in zip(tags, values):
            logger.log_scalar(value, 'Training/'+tag, step)
        window, window_count, window_start = None, 0, None

    try:
        while total_envsteps < args.total_steps:
            t = time.perf_counter()
            ### (TODO) Before args.learning_start, random an action
            ### After args.learning_start, get the action from the agent with agent.sample_action()
            if total_envsteps < args.learning_start:
                action = env.action_space.sample()
            elif args.cuda_policy_graph:
                if policy_graph is None:
                    policy_graph=CudaPolicy(agent)
                action=policy_graph.sample_action(state)
            else:
                action=agent.sample_action(state)
            phase['action_seconds'] += time.perf_counter()-t
            t = time.perf_counter()
            ### (TODO) Implement the action
            step_result = env.step(action)
            next_state, reward, terminated, truncated, _ = step_result
            done = terminated or truncated
            ### Add the tuple to the replay buffer
            # Time-limit truncation retains bootstrapping; replay done stores true termination only.
            agent.replay_buffer.add(state, action, reward, next_state, float(terminated))
            state = next_state
            episode_reward += reward
            total_envsteps += 1
            if done:
                append_csv(log_dir/'episodes.csv', {'episode': episode, 'env_steps': total_envsteps,
                                                  'return': episode_reward})
                episode += 1
                state, _ = env.reset()
                episode_reward = 0.0
            phase['env_seconds'] += time.perf_counter()-t
            if total_envsteps < args.learning_start:
                continue
            if learning_started_at is None:
                learning_started_at = time.perf_counter()
                learning_started_unix=time.time()
            t = time.perf_counter()
            ### (TODO) Sample args.batch_size from the agent's replay buffer
            batch = agent.replay_buffer.sample(args.batch_size)
            ### (TODO) Update the agent with the batch
            if args.cuda_graph:
                # Combine CUDA dispatch within one update without reducing updates or changing sample order.
                if graph_update is None:
                    graph_update = CudaUpdate(agent,batch)
                update_info = graph_update.update(batch)
            else:
                update_info = agent.update(batch, return_tensors=optimized)
            updates += 1
            phase['update_seconds'] += time.perf_counter()-t
            t = time.perf_counter()
            if optimized:
                values = torch.stack([update_info[name] for name in metric_names])
            else:
                values = torch.tensor([update_info[name] for name in metric_names], device=agent.device)
            window = values.clone() if window is None else window.add_(values)
            window_count += 1
            window_start = total_envsteps if window_start is None else window_start
            if window_count >= args.log_every:
                emit_metrics(total_envsteps)
            phase['logging_seconds'] += time.perf_counter()-t
            if total_envsteps >= next_eval or total_envsteps == args.total_steps:
                while next_eval <= total_envsteps:
                    next_eval += args.eval_every
                result = evaluation.evaluate(agent, args.eval_batch_size, args.eval_device)
                eval_steps += result['env_steps']
                evaluation_seconds += result['seconds']
                append_csv(log_dir/'evaluations.csv', {'env_steps': total_envsteps,
                           'mean_return': result['mean_return'], 'std_return': result['std_return'],
                           'episodes': args.eval_episodes, 'evaluation_seconds': result['seconds']})
                for i, (ret, length) in enumerate(zip(result['returns'], result['lengths'])):
                    append_csv(log_dir/'evaluation_episodes.csv', {'env_steps': total_envsteps,
                               'episode': i, 'env_seed': args.eval_seed+i, 'policy_seed': args.eval_seed+i,
                               'return': ret, 'length': length})
                logger.log_scalar(result['mean_return'], 'Eval Returns', total_envsteps)
                print(f'Steps {total_envsteps}/{args.total_steps} | eval {result["mean_return"]:.2f}', flush=True)
            if args.log_video and total_envsteps >= next_video:
                while next_video <= total_envsteps:
                    next_video += args.video_every
                video_steps += record_video(agent,args,log_dir,total_envsteps)
        emit_metrics(total_envsteps)
    except BaseException as exc:
        status, error = 'failed', repr(exc)
        raise
    finally:
        if args.device == 'cuda':
            torch.cuda.synchronize()
        loop_seconds = time.perf_counter()-started
        loop_finished_unix=time.time()
        torch.save({'config': config, 'env_steps': total_envsteps,
                    'actor_state_dict': agent.actor.state_dict(),
                    'critic1_state_dict': agent.critic1.state_dict(),
                    'critic2_state_dict': agent.critic2.state_dict(),
                    'target_critic1_state_dict': agent.target_critic1.state_dict(),
                    'target_critic2_state_dict': agent.target_critic2.state_dict(),
                    'log_temperature': agent.log_temperature.detach().cpu()}, log_dir/'checkpoint.pt')
        logger.close()
        evaluation.close()
        env.close()
        wall_seconds = time.perf_counter()-started
        write_json(log_dir/'summary.json', {'status': status, 'error': error,
                   'requested_training_steps': args.total_steps, 'actual_training_steps': total_envsteps,
                   'updates': updates, 'evaluation_steps': eval_steps, 'wall_seconds': wall_seconds,
                   'video_steps': video_steps,
                   'loop_seconds': loop_seconds, 'evaluation_seconds': evaluation_seconds,
                   'started_unix':started_unix,'active_started_unix':learning_started_unix,
                   'loop_finished_unix':loop_finished_unix,
                   'warmup_seconds': (learning_started_at-started) if learning_started_at else loop_seconds,
                   'active_training_seconds': (started+loop_seconds-learning_started_at-evaluation_seconds)
                                              if learning_started_at else 0.0,
                   'training_steps_per_wall_second': total_envsteps/wall_seconds,
                   'phase_host_seconds': phase,
                   'timing_note': 'CUDA is asynchronous; host phase timings are not GPU kernel timings.'})
    print(f'Completed {log_dir}: {total_envsteps/wall_seconds:.1f} training steps/s', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Soft Actor Critic')
    parser.add_argument('--env_name', required=True)
    parser.add_argument('--total_steps', type=int, default=1000000)
    parser.add_argument('--eval_every', type=int, default=10000)
    parser.add_argument('--eval_episodes', type=int, default=10)
    parser.add_argument('--eval_batch_size', type=int, default=10)
    parser.add_argument('--eval_device', choices=['cpu','cuda'], default=None)
    parser.add_argument('--log_video', action='store_true')
    parser.add_argument('--video_every', type=int, default=100000)
    parser.add_argument('--memory_size', type=int, default=1000000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--learning_start', type=int, default=5000)
    parser.add_argument('--actor_learning_rate', type=float, default=1e-4)
    parser.add_argument('--critic_learning_rate', type=float, default=1e-4)
    parser.add_argument('--discount_gamma', type=float, default=0.99)
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--layer_num', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--eval_seed', type=int, default=10000)
    parser.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    parser.add_argument('--torch_threads', type=int, default=1)
    parser.add_argument('--log_every', type=int, default=1000)
    parser.add_argument('--output_dir')
    parser.add_argument('--start_gate',default=None,help='benchmark synchronization file')
    parser.add_argument('--exp_name')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--reference', action='store_true', help='completed unoptimized numerical path')
    parser.add_argument('--fused_optimizer', action='store_true')
    parser.add_argument('--cuda_graph', action='store_true')
    parser.add_argument('--cuda_policy_graph', action='store_true')
    args = parser.parse_args()
    if min(args.total_steps,args.eval_every,args.eval_episodes,args.eval_batch_size,args.memory_size,
           args.batch_size,args.learning_start,args.hidden_size,args.layer_num,args.log_every,args.torch_threads,
           args.video_every) <= 0:
        parser.error('Counts and sizes must be positive')
    if min(args.actor_learning_rate,args.critic_learning_rate,args.temperature) <= 0:
        parser.error('Learning rates and initial temperature must be positive')
    if not 0 <= args.discount_gamma <= 1 or not 0 <= args.tau <= 1:
        parser.error('gamma and tau must be in [0,1]')
    if args.cuda_graph and (args.device != 'cuda' or not args.fused_optimizer or args.reference):
        parser.error('cuda_graph requires optimized CUDA with fused_optimizer')
    if args.cuda_policy_graph and not args.cuda_graph:
        parser.error('cuda_policy_graph requires cuda_graph')
    train(args)

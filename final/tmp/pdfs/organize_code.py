from pathlib import Path
import ast
import hashlib
import json
import re
import shutil

ROOT = Path(__file__).resolve().parents[2]
COURSE = ROOT.parent
DEST = ROOT / 'code'
HAN = re.compile(r'[\u3400-\u4dbf\u4e00-\u9fff]')
TRANSLATIONS = {
    '优化：流式编码单帧，不把 1000 帧全存入 RAM；使用独立噪声流。': 'Stream video frames instead of retaining 1000 frames in RAM; use independent noise.',
    '性能测试专用：多个独立运行完成初始化后同时开始，避免启动先后污染并发比较。': 'Benchmark only: synchronize workers after initialization for fair concurrency timing.',
    '优化：一次 CPU 复制取回窗口均值；明确标记窗口起止与计数，': 'Copy window means to the CPU once; record the window boundaries and update count.',
    '不是每一步的瞬时 loss，也不会改变更新频率/训练数据/随机数。': 'These are window means, not per-step losses; update frequency, data and RNG are unchanged.',
    '时间截断仍需 bootstrap，因此 replay 的 done 只记录真正 terminated。': 'Time-limit truncation retains bootstrapping; replay done stores true termination only.',
    '只合并同一次 update 的 CUDA 调度，不减少更新数或改变采样顺序。': 'Combine CUDA dispatch within one update without reducing updates or changing sample order.',
    '优化：后台批量写入，避免每个 scalar 都受到单元素队列的背压。': 'Batch background writes to avoid single-element queue backpressure for every scalar.',
    'reference 路径保留原设置供对照；close() 保证末尾日志不会丢失。': 'The reference path keeps the original settings; close() flushes the remaining events.',
    '固定种子评估；复用环境并批量推理，不使用训练随机流。': 'Fixed-seed evaluation with reusable environments, batched inference and independent RNG streams.',
    '优化：小批次评估可使用 CPU，避免逐步 GPU 调度/同步。': 'Small evaluation batches can use the CPU to avoid per-step GPU dispatch and synchronization.',
    'deepcopy 不随机初始化网络，不消耗训练 RNG；复制时间计入评估墙钟。': 'deepcopy avoids random initialization and training RNG consumption; copying counts as evaluation time.',
    '优化：每回合独立噪声流；批量大小不改变各回合的随机数序列，': 'Each episode has an independent noise stream, unaffected by the evaluation batch size.',
    '同时不需要保存/恢复训练 RNG，也不会让评估消耗训练随机数。': 'Evaluation neither consumes training randomness nor requires saving and restoring training RNG.',
    '修复模板：保存损失函数本身，不能在没有输入时调用 F.mse_loss()。': 'Store the loss callable; calling F.mse_loss() without inputs would fail.',
    '优化：foreach/fused 只改变优化器实现，不改变 Adam 超参数及每步更新次数。': 'foreach/fused changes the optimizer implementation, not Adam settings or updates per step.',
    '保留 reference 路径，便于以相同数据、随机数和初始权重验证数值差异。': 'Keep the reference path to compare numerics with identical data, randomness and initial weights.',
    '优化：采样仅用于环境交互，不构建随后马上丢弃的 autograd 图。': 'Environment action sampling does not need an autograd graph.',
    '优化：Normal.sample 底层 torch.normal 会检查 std 并阻碍 CUDA Graph。': 'Normal.sample uses torch.normal, whose scale checks prevent CUDA Graph capture.',
    '在 no_grad 下 rsample 仍是同分布高斯样本，但使用可捕获的标准正态变换。': 'Under no_grad, rsample draws the same Gaussian distribution using a capturable normal transform.',
    'Independent 已汇总动作维；这里只增加列维，避免 [B] 与 [B,1] 广播。': 'Independent already sums action dimensions; add only a column dimension to avoid [B]/[B,1] broadcasting.',
    '优化：两个 Q 网络不共享参数，损失相加一次 backward 与分别反传等价。': 'The Q networks share no parameters, so one backward on their summed losses is equivalent.',
    '优化：actor 步仍需 dQ/da，但不需要 Q 参数的梯度；不能用 no_grad 包住 Q。': 'The actor step needs dQ/da but not Q-parameter gradients; do not wrap Q in no_grad.',
    '优化：多张量操作合并 target 更新的 Python 循环/小 CUDA kernel；': 'Multi-tensor operations combine target-update Python loops and small CUDA kernels.',
    '保留相同 tau 与更新频率，舍入差异通过 reference 数值测试约束。': 'Keep tau and update frequency unchanged; compare roundoff against the reference path.',
    '优化：设备上累计日志窗口，只在窗口结束取回 CPU，避免每步 7 次 .item()。': 'Accumulate log windows on the device and copy once, avoiding seven .item() calls per step.',
    '优化：同一条 transition 连续存储，一次索引和一次设备复制替代五次。': 'Store each transition contiguously for one indexing operation and device copy instead of five.',
    '各列仍通过视图暴露原模板字段，不改变抽样分布、容量或存储精度。': 'Column views preserve the original fields, sampling distribution, capacity and storage precision.',
    '修复模板接口：SACAgent 与 PGAgent 都使用 sample_action。': 'Use the sample_action interface shared by SACAgent and PGAgent.',
    '可选 CUDA Graph 加速；保持一次 transition 对应一次完整 SAC update。': 'Optional CUDA Graph acceleration with one complete SAC update per eligible transition.',
    '优化：策略推理也捕获为图，减少单状态推理的小 kernel 调度。': 'Capture policy inference to reduce small-kernel dispatch for individual states.',
    '预热不改变权重，恢复 RNG 后不会额外消耗训练随机数。': 'Warm-up leaves weights unchanged; restoring RNG avoids consuming extra training randomness.',
    '图预热/捕获会执行更新；必须原地恢复权重、温度和 Adam 全部状态，': 'Graph warm-up/capture executes updates; restore weights, temperature and all Adam state in place.',
    '否则会偷偷多训练几步。不能 load_state_dict 替换图已引用的张量地址。': 'Otherwise extra training occurs. Do not replace tensor addresses already referenced by the graph.',
    '优化：提前分配优化器状态和临时内存；初始化成本单独计入试跑墙钟。': 'Preallocate optimizer state and temporary memory; include initialization in benchmark timing.',
    '优化：一次 cat 写入固定输入缓冲，避免每步五次小张量复制与重新建图。': 'Write a fixed input buffer with one cat, avoiding five small copies and graph reconstruction.',
    '优化：GPU 上逐次参数检查会执行 valid.all() 的 Python 布尔判断并同步。': 'Repeated GPU parameter checks convert valid.all() to a Python boolean and synchronize.',
    'std 已由 clamp 后 exp 保证为正；优化路径关闭重复分布检查，': 'clamp followed by exp ensures positive std; the optimized path disables repeated distribution checks.',
    'runner 仍在每个日志窗口检查 NaN/Inf，reference 保留原始检查。': 'The runner checks NaN/Inf every log window; the reference path retains the original checks.',
}

def executable_tree(text):
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
                node.body.pop(0)
    return ast.dump(tree, include_attributes=False)

records = []
for n in (3, 4, 5):
    src = COURSE / f'ws{n}' / 'final' / f'Workshop{n}-Code-Cao-Yang'
    dst = DEST / f'ws{n}'
    dst.mkdir(parents=True, exist_ok=True)
    for path in sorted(src.rglob('*')):
        if not path.is_file() or not (path.suffix == '.py' or path.name == 'requirements.txt'):
            continue
        if '__pycache__' in path.parts:
            continue
        relative = path.relative_to(src)
        target = dst / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        original = path.read_text(encoding='utf-8-sig')
        translated = original
        for a, b in TRANSLATIONS.items():
            translated = translated.replace(a, b)
        assert not HAN.search(translated), target
        if path.suffix == '.py':
            assert executable_tree(original) == executable_tree(translated), target
        if translated == original:
            shutil.copy2(path, target)
        else:
            target.write_text(translated, encoding='utf-8', newline='\n')
        records.append({'source': path.relative_to(COURSE).as_posix(),
                        'destination': target.relative_to(ROOT).as_posix(),
                        'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                        'delivered_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                        'changes': 'English comments/docstrings only' if translated != original else 'Unchanged',
                        'executable_ast_identical': True if path.suffix == '.py' else None})

# Export portable commands from the actual formal-run configurations, without old queued status.
commands = json.loads((COURSE / 'ws5/evidence/execution/approved_commands.json').read_text(encoding='utf-8'))
portable = []
for entry in commands:
    command = entry['command']
    script_index = next(i for i, part in enumerate(command) if part.endswith('.py'))
    script = 'SAC/run_sac.py' if entry['algorithm'] == 'sac' else 'Policy_Gradient/run_policy_gradient.py'
    args = command[script_index+1:]
    args[args.index('--output_dir') + 1] = 'runs/' + entry['condition_id'] + '__seed0'
    summary_file = COURSE / 'ws5' / entry['output_dir'] / 'summary.json'
    summary = json.loads(summary_file.read_text(encoding='utf-8'))
    portable.append({'condition_id': entry['condition_id'], 'algorithm': entry['algorithm'],
                     'original_run_status': summary['status'],
                     'command': ['python', script] + args})
(DEST / 'ws5/experiment_commands.json').write_text(json.dumps(portable, indent=2) + '\n', encoding='utf-8')
(ROOT / 'tmp/pdfs/code_source_manifest.json').write_text(json.dumps(records, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'files_copied': len(records), 'translated_files': sum(r['changes'] != 'Unchanged' for r in records),
                  'portable_ws5_commands': len(portable)}, indent=2))

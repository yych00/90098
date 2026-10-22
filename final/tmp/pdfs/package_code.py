from pathlib import Path
import ast
import csv
import hashlib
import json
import math
import re
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / 'code'
PYTHON = Path(r'D:\code\miniconda3\envs\vla\python.exe')
TEMP = ROOT / 'tmp/pdfs'

shutil.copy2(TEMP / 'code_source_manifest.json', CODE / 'SOURCE_MANIFEST.json')
checks = []
for i in range(1, 5):
    run = TEMP / f'code_smoke_{i}'
    required = ['config.json', 'metrics.csv', 'evaluations.csv', 'evaluation_episodes.csv', 'summary.json', 'checkpoint.pt']
    assert all((run / name).is_file() for name in required), run
    summary = json.loads((run / 'summary.json').read_text(encoding='utf-8'))
    assert summary['status'] in ('complete', 'completed')
    assert summary['updates'] > 0
    count = 0
    for path in run.glob('*.csv'):
        with path.open(newline='', encoding='utf-8') as stream:
            for row in csv.DictReader(stream):
                count += 1
                for value in row.values():
                    try:
                        number = float(value)
                    except (ValueError, TypeError):
                        continue
                    assert math.isfinite(number), (path, row)
    checks.append({'check': i, 'status': summary['status'], 'actual_training_steps': summary['actual_training_steps'],
                    'updates': summary['updates'], 'finite_csv_rows': count})

files = sorted(p for p in CODE.rglob('*') if p.is_file())
for path in files:
    relative = path.relative_to(CODE)
    assert not any(x in relative.parts for x in ('__pycache__', '.venv', 'runs', 'logs'))
    assert not re.search(r'[\u3400-\u4dbf\u4e00-\u9fff]', relative.as_posix())
    text = path.read_text(encoding='utf-8-sig')
    assert not re.search(r'[\u3400-\u4dbf\u4e00-\u9fff]', text), path
    if path.suffix == '.py':
        compile(text, str(path), 'exec')

archive = ROOT / 'Code_WS3_WS4_WS5.zip'
with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as zipped:
    for path in files:
        zipped.write(path, Path('code') / path.relative_to(CODE))

extracted = TEMP / 'clean_code_extract'
extracted.mkdir(parents=True, exist_ok=False)
with zipfile.ZipFile(archive) as zipped:
    assert zipped.testzip() is None
    zipped.extractall(extracted)
for path in files:
    copy = extracted / 'code' / path.relative_to(CODE)
    assert hashlib.sha256(path.read_bytes()).digest() == hashlib.sha256(copy.read_bytes()).digest()

entries = [('ws3', 'DQN/run_dqn.py'), ('ws4', 'Policy_Gradient/run_policy_gradient.py'),
           ('ws5', 'Policy_Gradient/run_policy_gradient.py'), ('ws5', 'SAC/run_sac.py')]
cli = []
for workshop, script in entries:
    result = subprocess.run([str(PYTHON), '-B', script, '--help'],
                            cwd=extracted / 'code' / workshop,
                            capture_output=True, text=True, timeout=90)
    (TEMP / f'{workshop}_{Path(script).stem}_help.txt').write_text(result.stdout + result.stderr, encoding='utf-8')
    assert result.returncode == 0 and '--env_name' in result.stdout, (workshop, result.stderr)
    cli.append({'workshop': workshop, 'entry_point': script, 'exit_code': result.returncode})

qa = {'archive': archive.name, 'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
      'files': len(files), 'python_files': sum(p.suffix == '.py' for p in files),
      'english_only': True, 'clean_extraction_hashes_match': True, 'cli_checks': cli, 'smoke_checks': checks}
(TEMP / 'code_package_qa.json').write_text(json.dumps(qa, indent=2) + '\n', encoding='utf-8')
print(json.dumps(qa, indent=2))

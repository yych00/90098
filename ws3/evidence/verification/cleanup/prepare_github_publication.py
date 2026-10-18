"""Prepare complete cleaned WS3 and WS4 for public GitHub upload."""
from pathlib import Path
import hashlib
import json
import re
import shutil
import subprocess
import sys

WS3=next(p for p in Path(__file__).resolve().parents if p.name=='ws3')
COURSE=WS3.parent
TARGET=Path(r'C:\Users\25704\.codex\visualizations\2026\10\15\01a141e0-85bd-7c91-a9ed-28eff84022fc\github-90098')
TARGET.mkdir(parents=True,exist_ok=True)
record=WS3/'evidence/verification/cleanup/github_publication.json'
record.write_text(json.dumps(dict(status='prepared_not_uploaded',account='yych00',repository='90098',visibility='public',scope='complete cleaned ws3 and ws4',snapshot_path=str(TARGET),blocker='Repository creation API unavailable in connector; local GitHub credential access denied; no browser provider available.'),indent=2),encoding='utf-8')
sources=[]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

for number in (3,4):
    root=COURSE/f'ws{number}'
    local=root
    integrity=json.loads((root/'evidence/verification/delivery_manifest.json').read_text(encoding='utf-8'))
    expected={x['path']:x['sha256'] for x in integrity['inventory']}
    for source in sorted(local.rglob('*')):
        if not source.is_file(): continue
        relative=source.relative_to(root)
        assert relative.parts[0] in ('final','evidence')
        if relative.as_posix() in expected:
            assert sha(source)==expected[relative.as_posix()],source
        destination=TARGET/f'ws{number}'/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,destination)
        assert sha(source)==sha(destination)
        sources.append(dict(path=destination.relative_to(TARGET).as_posix(),source_sha256=sha(source)))
(TARGET/'README.md').write_text('''# 90098 - Reinforcement Learning Workshops

Complete cleaned ELEN90098 Workshop 3 and Workshop 4 directories: final reports, completed submission code and experimental evidence.

Authors: Zhengnan Cao and Yuchen Yang.

## Reports and code

| Workshop | English report | Chinese report | Submission code |
|---|---|---|---|
| WS3: Deep Q-Learning | [PDF](ws3/final/Workshop3_Final_Report_EN.pdf) / [Markdown](ws3/final/Workshop3_Final_Report_EN.md) | [PDF](ws3/final/Workshop3_Final_Report_ZH.pdf) / [Markdown](ws3/final/Workshop3_Final_Report_ZH.md) | [DQN](ws3/final/Workshop3-Code-Cao-Yang/) |
| WS4: Policy Gradient | [PDF](ws4/final/Workshop4_Final_Report_EN.pdf) / [Markdown](ws4/final/Workshop4_Final_Report_EN.md) | [PDF](ws4/final/Workshop4_Final_Report_ZH.pdf) / [Markdown](ws4/final/Workshop4_Final_Report_ZH.md) | [Policy Gradient](ws4/final/Workshop4-Code-Cao-Yang/) |

Each report combines Task 1 and Task 2. Experimental questions explicitly identify the question, submitted plots/results and discussion answers. Teaching comments and TODO annotations remain in the code, with the implementations filled in. Submission folders contain English READMEs and dependency lists; no code ZIP is required to browse this repository.

## Topics

- **WS3:** CartPole DQN, learning rate, epsilon-greedy exploration schedules, replay capacity and minibatch size.
- **WS4:** InvertedPendulum Vanilla PG, reward-to-go and a previous-batch historical-return baseline; learning rates and batch steps; CartPole PG versus DQN; a tuned HalfCheetah controller.

Reports use the verified corrected experiment set: 33 DQN formal runs, 51 corrected PG formal runs and six PG selection pilots. Corrected RTG uses episode-origin discount powers; the baseline is formed from earlier batches only. Results include seed variability and unsuccessful configurations. In particular, the HalfCheetah results do not establish uniformly reliable control.

## Running the code

Use Python 3.10 or newer. Follow the setup and command-line examples in the corresponding code README:

- [WS3 setup and commands](ws3/final/Workshop3-Code-Cao-Yang/README.md)
- [WS4 setup and commands](ws4/final/Workshop4-Code-Cao-Yang/README.md)

The recorded checks used Python 3.10.21 and PyTorch 2.5.1. MuJoCo is required for InvertedPendulum and HalfCheetah.

## Snapshot contents

Both workshop directories are included in full. Each contains `final/` and `evidence/`. The evidence includes original assignment materials, frozen experiment source, run configurations, metrics, evaluations, checkpoints, result analysis, available media, integrity manifests and verification records. No training or PDF regeneration was performed during publication preparation.

Historical records retain their original machine paths as provenance. Use the relative paths in each `evidence/run_manifest.json` to locate archived runs. Historical analysis scripts may need their paths adapted before rerunning. Operational preparation records describe the pre-upload state and remain historical snapshots.

From the repository root, verify the archived delivery without training:

```sh
python -B ws3/evidence/verification/verify_clean_delivery.py
```

`PUBLICATION_MANIFEST.json` records the size and SHA256 of every published payload file.
''',encoding='utf-8')
(TARGET/'.gitignore').write_text('''__pycache__/
*.py[cod]
.venv/
venv/
.env
.env.*
*.tmp
.DS_Store
Thumbs.db
''',encoding='utf-8')
(TARGET/'.gitattributes').write_text('''* -text
*.pdf binary
*.png binary
*.pt binary
*.npz binary
*.mp4 binary
*.zip binary
''',encoding='utf-8')

secret_patterns=[r'-----BEGIN [A-Z ]*PRIVATE KEY-----',r'gh[pousr]_[A-Za-z0-9]{25,}',r'github_pat_[A-Za-z0-9_]{20,}',r'AKIA[0-9A-Z]{16}',r'sk-[A-Za-z0-9_-]{30,}']
files=sorted(p for p in TARGET.rglob('*') if p.is_file() and p != TARGET/'PUBLICATION_MANIFEST.json')
for file in files:
    if file.suffix in ('.md','.py','.txt','.json','.csv','.log','.yaml','.yml','.toml','.ps1','.js','.html'):
        text=file.read_text(encoding='utf-8-sig')
        assert not any(re.search(pattern,text) for pattern in secret_patterns),file
        if 'final' in file.relative_to(TARGET).parts:
            for path in re.findall(r'!\[[^\]]*\]\(([^)]+)\)',text):
                assert (file.parent/path).is_file(),(file,path)
inventory=[dict(path=p.relative_to(TARGET).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in files]
manifest=dict(repository='yych00/90098',requested_visibility='public',status='prepared_not_uploaded',
    payload_files=len(inventory),bytes=sum(x['bytes'] for x in inventory),inventory=inventory,
    immutable_local_sources=sources,scope='complete cleaned ws3 and ws4',excluded=[])
(TARGET/'PUBLICATION_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
assert {x['path'] for x in sources} == {p.relative_to(TARGET).as_posix() for folder in ('ws3','ws4') for p in (TARGET/folder).rglob('*') if p.is_file()}
subprocess.run([sys.executable,'-B',str(TARGET/'ws3/evidence/verification/verify_clean_delivery.py')],check=True)
print(json.dumps(dict(status='prepared',target=str(TARGET),files=len(inventory)+1,MiB=round(manifest['bytes']/1048576,2))))

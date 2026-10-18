"""Verify relocated reports, submission source and 90 runs without training."""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import re

def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''): digest.update(block)
    return digest.hexdigest()

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def resolve(base,relative):
    assert not Path(relative).is_absolute(),relative
    path=(base/relative).resolve()
    assert path.is_relative_to(base.resolve()),relative
    assert path.is_file(),path
    return path

def verify(staged=False):
    ws3=next(p for p in Path(__file__).resolve().parents if p.name=='ws3')
    course=ws3.parent
    summary=[]
    for n in (3,4):
        base=course/f'ws{n}'
        if staged: base=base/'_cleanup_stage_20261018'
        evidence=base/'evidence'; final=base/'final'
        manifest=read(evidence/'verification/delivery_manifest.json')
        for item in manifest['inventory']:
            path=resolve(base,item['path'])
            assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256'],path
        actual={p.relative_to(base).as_posix() for folder in ('final','evidence')
                for p in (base/folder).rglob('*') if p.is_file()
                and p.relative_to(base).as_posix()!='evidence/verification/delivery_manifest.json'
                and ('cleanup' not in p.relative_to(base).parts or p.name=='relocation_manifest.json')}
        assert actual=={x['path'] for x in manifest['inventory']},actual^{x['path'] for x in manifest['inventory']}
        runs=read(evidence/'run_manifest.json')['runs']
        assert len(runs)==(33 if n==3 else 57)
        assert len({r['id'] for r in runs})==len(runs)
        source=evidence/'experiment_source'/('DQN' if n==3 else 'Policy_Gradient')
        if n==3:
            digest=hashlib.sha256()
            for relative in ('run_dqn.py','dqn/agent.py','dqn/network.py'):
                digest.update(relative.encode()); digest.update((source/relative).read_bytes())
            trainer_hash=digest.hexdigest()
        else:
            files=[source/'run_policy_gradient.py',*(source/'Policy_Gradient').glob('*.py')]
            hashes={p.relative_to(source).as_posix():sha(p) for p in sorted(files)}
            trainer_hash=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
        for run in runs:
            runpath=evidence/run['path']
            assert runpath.resolve().is_relative_to(evidence.resolve())
            assert trainer_hash==run['trainer_hash']
            for name,digest in run['core_files'].items():
                assert sha(resolve(runpath,name))==digest,(run['id'],name)
            config=read(runpath/'config.json'); result=read(runpath/'summary.json')
            assert config['code_hash']==trainer_hash and result['status'] in {'complete','completed','success'},run['id']
            if n==4: assert config['estimator_version']=='corrected-v2'
        original=read(evidence/'verification/final_report_v3_manifest_original.json')
        # All submitted bytes and figure assets remain identical; only DELIVERY changed.
        for item in original['inventory']:
            if item['path'].startswith('verification/') or item['path']=='DELIVERY.md': continue
            assert sha(resolve(final,item['path']))==item['sha256'],item['path']
        code=final/f'Workshop{n}-Code-Cao-Yang'
        codefiles=sorted(p for p in code.rglob('*') if p.is_file())
        assert len(codefiles)==(7 if n==3 else 8)
        assert not list(final.rglob('*.zip'))
        assert not any(p.name=='__pycache__' for p in code.rglob('*'))
        assert [p.name for p in codefiles if p.name.lower().startswith('readme')]==['README.md']
        assert not re.search('[\u4e00-\u9fff]',(code/'README.md').read_text(encoding='utf-8'))
        for path in codefiles:
            if path.suffix=='.py': ast.parse(path.read_text(encoding='utf-8-sig'))
        for md in final.glob('Workshop*_Final_Report_*.md'):
            for relative in re.findall(r'!\[[^\]]*\]\(([^)]+)\)',md.read_text(encoding='utf-8')):
                resolve(final,relative)
        qa=read(evidence/'verification/final_report_v3/pdf_qa.json')
        for pdf in qa['files']:
            assert sha(final/pdf['path'])==pdf['sha256'] and pdf['visually_checked']
        summary.append(dict(workshop=n,inventory_files=len(manifest['inventory']),runs=len(runs),
            code_files=len(codefiles),pdf_pages=sum(x['pages'] for x in qa['files']),trainer_hash=trainer_hash))
    base4=course/'ws4'
    base3=course/'ws3'
    if staged: base4=base4/'_cleanup_stage_20261018'; base3=base3/'_cleanup_stage_20261018'
    media=read(base4/'evidence/analysis/q3_q4/video_decode_check.json')
    assert sha(base4/'evidence/analysis/q3_q4/representative_seed0.mp4')==media['sha256']
    assert sha(base4/'final/figures/task2_q4_frames.png')==media['contact_sheet_sha256']
    assert sha(base4/'evidence/runs/formal/halfcheetah_final_s0/checkpoint.pt')==media['checkpoint_sha256']
    reload_record=base3/'evidence/verification/joint_original/reload_halfcheetah_pg_seed0.json'
    assert sha(reload_record)==media['render_record_sha256']
    result=dict(status='passed',staged=staged,workshops=summary,media_bindings='passed',formal_training=False)
    print(json.dumps(result,ensure_ascii=False))
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--staged',action='store_true')
    verify(parser.parse_args().staged)

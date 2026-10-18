"""External QA for the WS4 teacher-template submission; no formal training."""
from __future__ import annotations
import ast
import csv
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tokenize
import types
from zipfile import ZipFile

import gymnasium as gym
import numpy as np
import torch

COURSE = Path(r'D:\uni mel\Melbourne College\第四学期\90098强化学习\ws4')
QA = COURSE / 'tmp/final_v3_code_qa'
SUBMISSION = COURSE / 'outputs/final_report_v3/Workshop4-Code-Cao-Yang'
PROJECT = SUBMISSION / 'Policy_Gradient'
FROZEN = Path(r'D:\uni mel\Melbourne College\第四学期\90098强化学习\ws3\task2_workspace\corrected_v2\ws4\Policy_Gradient')
EXPECTED = '54d3600cf459dd85ee20b8af00f1aacf47ed022eb0714807e99e1f4ba6794341'
RESULT = {'status': 'running', 'formal_training_started': False, 'submission': str(SUBMISSION)}
torch.set_num_threads(1)
sys.dont_write_bytecode = True


def comments(source):
    return [t.string for t in tokenize.generate_tokens(io.StringIO(source).readline) if t.type == tokenize.COMMENT]


def definitions(tree):
    found = {}
    def visit(node, prefix=''):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                name = prefix + child.name
                found[name] = child
                visit(child, name + '.')
            else:
                visit(child, prefix)
    visit(tree)
    return found


coverage = []
with ZipFile(COURSE / 'Policy_Gradient.zip') as archive:
    for name in archive.namelist():
        if not name.endswith('.py') or '__pycache__' in name:
            continue
        old = archive.read(name).decode('utf-8-sig')
        path = SUBMISSION / name
        new = path.read_text(encoding='utf-8-sig')
        before, after = comments(old), comments(new)
        cursor = 0
        for text in before:
            while cursor < len(after) and after[cursor] != text:
                cursor += 1
            assert cursor < len(after), (name, 'missing or reordered original comment', repr(text))
            cursor += 1
        original, current = definitions(ast.parse(old)), definitions(ast.parse(new))
        for symbol, node in original.items():
            assert symbol in current, (name, 'original class/function missing', symbol)
            assert ast.get_docstring(node) == ast.get_docstring(current[symbol]), (name, 'docstring changed', symbol)
            if isinstance(node, ast.FunctionDef):
                args = [x.arg for x in node.args.args]
                assert [x.arg for x in current[symbol].args.args][:len(args)] == args, (name, symbol, 'original interface changed')
        assert not any(isinstance(x, ast.Constant) and x.value is Ellipsis for x in ast.walk(ast.parse(new))), name
        compile(new, str(path), 'exec')
        coverage.append({'file': name, 'original_comments': len(before), 'preserved_comments': len(before),
                         'original_todo_comments': sum('TODO' in text for text in before),
                         'original_w11_comments': sum('W11' in text for text in before),
                         'original_classes_functions': list(original), 'ordered_exact_comment_coverage': True,
                         'original_docstrings_and_argument_prefixes_preserved': True})
assert sum(x['original_todo_comments'] for x in coverage) == 21
RESULT['template_coverage'] = coverage
RESULT['total_original_comments'] = sum(x['original_comments'] for x in coverage)
RESULT['total_original_todo_comments'] = 21
RESULT['all_original_comments_preserved_in_order'] = True
RESULT['teacher_zip_sha256'] = hashlib.sha256((COURSE / 'Policy_Gradient.zip').read_bytes()).hexdigest()


def module_from_source(name, source, filename):
    module = types.ModuleType(name)
    module.__file__ = str(filename)
    sys.modules[name] = module
    exec(compile(source, str(filename), 'exec'), module.__dict__)
    return module


def load_agent(prefix, directory, package):
    network_path = directory / package / 'network.py'
    module_from_source(prefix+'_network', network_path.read_text(encoding='utf-8-sig'), network_path)
    agent_path = directory / package / 'agent.py'
    source = agent_path.read_text(encoding='utf-8-sig').replace('from '+package+'.network import', 'from '+prefix+'_network import')
    return module_from_source(prefix+'_agent', source, agent_path)


frozen = load_agent('reference', FROZEN, 'Policy_Gradient')
submission = load_agent('submitted', PROJECT, 'policy_gradient')
source_files = sorted([FROZEN/'run_policy_gradient.py', *(FROZEN/'Policy_Gradient').glob('*.py')])
source_hashes = {str(p.relative_to(FROZEN)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
actual_source = hashlib.sha256(json.dumps(source_hashes,sort_keys=True).encode()).hexdigest()
assert actual_source == EXPECTED
RESULT['frozen_trainer_hash'] = actual_source
maxima = dict(return_error=0., rtg_error=0., loss_error=0., gradient_error=0., updated_parameter_error=0., gae_error=0.)
comparisons = []


def dummy(kind):
    width = 4 if kind == 'categorical' else 17
    action = gym.spaces.Discrete(2) if kind == 'categorical' else gym.spaces.Box(-1,1,(1 if kind=='gaussian_1' else 6,),dtype=np.float32)
    return types.SimpleNamespace(observation_space=gym.spaces.Box(-np.inf,np.inf,(width,),dtype=np.float32), action_space=action)


def max_error(a,b):
    return float(np.max(np.abs(np.asarray(a)-np.asarray(b))))


rng = np.random.default_rng(65537)
for kind in ('gaussian_1','gaussian_6','categorical'):
    env = dummy(kind)
    for method in ('vanilla','rtg','baseline','gae'):
        for normalize in (False, True):
            options = dict(hidden_size=64,layer_num=3 if kind=='categorical' else 2,
                           gamma=.99,use_reward_to_go=method=='rtg',use_baseline=method=='baseline',
                           gae_lambda=.95 if method=='gae' else 1.,normalize_advantage=normalize,
                           actor_lr=.001,critic_lr=.001)
            torch.manual_seed(123)
            gold = frozen.PGAgent(env,**options)
            torch.manual_seed(124)
            test = submission.PGAgent(env,**options)
            test.actor.load_state_dict(gold.actor.state_dict())
            if gold.critic is not None:
                test.critic.load_state_dict(gold.critic.state_dict())
            for gamma in (.9,.99):
                gold.gamma=test.gamma=gamma
                r=np.array([1.,-.4,2.7,.2,1.3],dtype=np.float32)
                maxima['return_error']=max(maxima['return_error'],max_error(gold._discounted_return(r),test._discounted_return(r)))
                maxima['rtg_error']=max(maxima['rtg_error'],max_error(gold._discounted_reward_to_go(r),test._discounted_reward_to_go(r)))
            gold.gamma=test.gamma=.99
            per_case=[]
            for update in range(2):
                states=rng.normal(0,.3,(5,env.observation_space.shape[0])).astype(np.float32)
                next_states=rng.normal(0,.3,states.shape).astype(np.float32)
                actions=(np.array([0,1,1,0,1],dtype=np.int64) if kind=='categorical' else
                         rng.normal(0,1.7,(5,env.action_space.shape[0])).astype(np.float32))
                batch={'states':states,'next_states':next_states,'actions':actions,
                       'rewards':np.array([1.,.5,-.7,2.,.2],dtype=np.float32)*(update+1),
                       'terminateds':np.array([False,False,False,False,True]),
                       'dones':np.array([False,False,True,False,True])}
                a,b=gold.update(batch),test.update(batch)
                le=max(abs(a[key]-b[key]) for key in a if a[key] is not None)
                ge=max(float((x.grad-y.grad).abs().max()) for x,y in zip(gold.actor.parameters(),test.actor.parameters()))
                pe=max(float((x-y).detach().abs().max()) for x,y in zip(gold.actor.parameters(),test.actor.parameters()))
                if gold.critic is not None:
                    ge=max(ge,max(float((x.grad-y.grad).abs().max()) for x,y in zip(gold.critic.parameters(),test.critic.parameters())))
                    pe=max(pe,max(float((x-y).detach().abs().max()) for x,y in zip(gold.critic.parameters(),test.critic.parameters())))
                assert le <= 1e-7 and ge <= 1e-7 and pe <= 1e-7,(kind,method,normalize,le,ge,pe)
                assert gold.baseline_return_sum==test.baseline_return_sum and gold.baseline_episode_count==test.baseline_episode_count
                maxima['loss_error']=max(maxima['loss_error'],le)
                maxima['gradient_error']=max(maxima['gradient_error'],ge)
                maxima['updated_parameter_error']=max(maxima['updated_parameter_error'],pe)
                per_case.append({'update':update+1,'loss_metric_error':le,'gradient_error':ge,'parameter_error':pe,
                                 'baseline_episode_count':test.baseline_episode_count})
            comparisons.append({'actions':kind,'method':method,'normalize_advantage':normalize,'updates':per_case})
RESULT['fixed_input_comparisons']=comparisons

# An unseen single trajectory must not subtract itself away.
agent=submission.PGAgent(dummy('categorical'),use_baseline=True)
batch={'states':np.arange(12,dtype=np.float32).reshape(3,4)/10,
       'next_states':np.zeros((3,4),np.float32),'actions':np.array([0,1,1]),
       'rewards':np.array([1.,2.,3.],np.float32),'terminateds':np.array([False,False,True]),
       'dones':np.array([False,False,True])}
a=agent.update(batch)
assert a['gradient_norm']>0 and a['advantage_mean']>0 and agent.baseline_episode_count==1
prior=agent.baseline_return_sum
second={**batch,'rewards':np.array([4.,5.,6.],np.float32)}
b=agent.update(second)
expected=float(agent._discounted_return(second['rewards'])[0])-prior
assert np.isclose(b['advantage_mean'],expected,atol=1e-6) and agent.baseline_episode_count==2
RESULT['single_trajectory_baseline']={'first_gradient_norm':a['gradient_norm'],'first_advantage_mean':a['advantage_mean'],
                                    'second_advantage_expected':expected,'second_advantage_observed':b['advantage_mean']}

# A terminal state suppresses bootstrap; a time limit retains it, with no cross-episode carry.
agent=submission.PGAgent(dummy('categorical'),gamma=.9,gae_lambda=.95)
with torch.no_grad():
    for p in agent.critic.parameters():p.zero_()
    agent.critic.net[-1].bias.fill_(2.)
returns,advantages=agent.generalized_advantage_estimation(np.zeros((2,4)),np.zeros((2,4)),
                                                         [1.,1.],[True,False],[True,True])
maxima['gae_error']=max(max_error(returns,[1.,2.8]),max_error(advantages,[-1.,.8]))
assert maxima['gae_error']<1e-6
RESULT['termination_truncation_contract']={'returns':returns.tolist(),'advantages':advantages.tolist(),
                                         'bootstrap_on_truncation_only':True,'episode_recursion_reset':True}

# Raw Gaussian action samples retain their action dimension and can exceed execution bounds.
for kind in ('gaussian_1','gaussian_6'):
    agent=submission.PGAgent(dummy(kind))
    with torch.no_grad():
        for p in agent.actor.parameters():p.zero_()
    torch.manual_seed(0)
    action=agent.sample_action(np.zeros(17,np.float32))
    assert action.shape==dummy(kind).action_space.shape and np.any(np.abs(action)>1)
RESULT['raw_gaussian_sample_contract']=True

# Import the utilities in isolation and compare fixed evaluation episodes to the frozen runner.
util_path=PROJECT/'utils/util.py'
util=module_from_source('submitted_util',util_path.read_text(encoding='utf-8-sig'),util_path)
runner_path=FROZEN/'run_policy_gradient.py'
frozen_runner=module_from_source('reference_runner',runner_path.read_text(encoding='utf-8-sig').replace('from Policy_Gradient.agent import','from reference_agent import'),runner_path)
e1,e2=gym.make('CartPole-v1'),gym.make('CartPole-v1')
try:
    actor=submission.PGAgent(e1,hidden_size=64,layer_num=3)
    random.seed(812);np.random.seed(813);torch.manual_seed(814)
    py,ns,ts=random.getstate(),np.random.get_state(),torch.get_rng_state().clone()
    data=util.collect_trajectories(e1,actor,10,seeds=list(range(10000,10010)))
    assert random.getstate()==py and np.array_equal(np.random.get_state()[1],ns[1]) and torch.equal(torch.get_rng_state(),ts)
    again=util.collect_trajectories(e1,actor,10,seeds=list(range(10000,10010)))
    reference=frozen_runner.evaluate(e2,actor,10000,10)
    values=[float(np.sum(x)) for x in data['rews_list']]
    assert values==[float(np.sum(x)) for x in again['rews_list']]==[x['return'] for x in reference]
    RESULT['fixed_seed_evaluation']={'episodes':10,'seeds':list(range(10000,10010)),'returns':values,
                                     'matches_frozen_evaluator':True,'training_rng_unchanged':True}
finally:e1.close();e2.close()

log_path=PROJECT/'utils/log_utils.py'
log=module_from_source('submitted_log_utils',log_path.read_text(encoding='utf-8-sig'),log_path)
class BoundaryEnv:
    def __init__(self,mode):
        self.action_space=gym.spaces.Discrete(2);self.mode=mode;self.count=0
    def reset(self):self.count=0;return np.zeros(4,np.float32),{}
    def step(self,action):
        assert isinstance(action,int)
        self.count+=1
        return np.zeros(4,np.float32),1.,self.mode=='terminated',self.mode=='truncated',{}
class ZeroPolicy:
    def sample_action(self,ob):return 0
for mode in ('terminated','truncated','max_length'):
    traj=log.sample_trajectory(BoundaryEnv(mode),ZeroPolicy(),2)
    assert traj['terminal'][-1]==1
    assert traj['terminated'][-1]==(mode=='terminated')
    assert traj['truncated'][-1]==(mode!='terminated')
RESULT['helper_rollout_boundary_contract']=True
RESULT['maximum_absolute_errors']=maxima

# Independent code copy; subprocesses have no path to the frozen implementation.
copy_root=QA/'independent_copy'
assert not copy_root.exists(),'Preserve the existing independent-copy QA directory'
shutil.copytree(SUBMISSION,copy_root)
smokes=[]
for name,env,flags,steps,batch in [
    ('cartpole_rtg','CartPole-v1',['--use_reward_to_go','--layer_num','3','--learning_rate','.003'],64,32),
    ('halfcheetah_rtg_normalized','HalfCheetah-v5',['--use_reward_to_go','--normalize_advantage'],8,8),
    ('pendulum_gae','InvertedPendulum-v5',['--gae_lambda','.95','--normalize_advantage'],24,12)]:
    output=QA/('smoke_'+name)
    cmd=[sys.executable,'-B','run_policy_gradient.py','--env_name',env,'--device','cpu','--discount_gamma','.99',
         '--total_steps',str(steps),'--batch_steps',str(batch),'--eval_every',str(batch),'--eval_episodes','10',
         '--seed','7','--output_dir',str(output),*flags]
    clean={**os.environ,'PYTHONPATH':'','PYTHONDONTWRITEBYTECODE':'1'}
    run=subprocess.run(cmd,cwd=copy_root/'Policy_Gradient',env=clean,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                       text=True,encoding='utf-8',errors='replace',timeout=180)
    (QA/(name+'.log')).write_text(run.stdout,encoding='utf-8')
    assert run.returncode==0,(name,run.stdout)
    summary=json.loads((output/'summary.json').read_text(encoding='utf-8'))
    assert summary['status']=='completed' and summary['actual_training_steps']>=steps
    ev=list(csv.DictReader((output/'evaluations.csv').open(encoding='utf-8')))
    episodes=list(csv.DictReader((output/'evaluation_episodes.csv').open(encoding='utf-8')))
    assert int(ev[0]['env_steps'])==0 and int(ev[-1]['env_steps'])==summary['actual_training_steps']
    assert len(episodes)==len(ev)*10
    for row in ev:
        detail=[x for x in episodes if x['env_steps']==row['env_steps']]
        assert [int(x['seed']) for x in detail]==list(range(10000,10010))
        assert np.isclose(np.mean([float(x['return']) for x in detail]),float(row['mean_return']))
    checkpoint=torch.load(output/'checkpoint.pt',map_location='cpu',weights_only=False)
    assert all(torch.isfinite(v).all() for v in checkpoint['actor_state_dict'].values())
    smokes.append({'name':name,'command':cmd,'cwd':str(copy_root/'Policy_Gradient'),'returncode':run.returncode,
                   'requested_steps':steps,'actual_steps':summary['actual_training_steps'],'updates':summary['updates'],
                   'evaluation_checkpoints':len(ev),'evaluation_episodes_per_checkpoint':10,
                   'log':str(QA/(name+'.log')),'non_formal_smoke_only':True})
RESULT['independent_cpu_cli_smokes']=smokes
RESULT['status']='passed'
RESULT['files']=[{'path':str(p.relative_to(SUBMISSION)).replace('\\','/'),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(SUBMISSION.rglob('*')) if p.is_file()]
assert not list(SUBMISSION.rglob('__pycache__'))
RESULT['submission_contains_no_cache_logs_checkpoints']=all(p.suffix not in {'.pt','.pyc','.csv','.json'} for p in SUBMISSION.rglob('*') if p.is_file())
assert RESULT['submission_contains_no_cache_logs_checkpoints']
(QA/'ws4_code_verification.json').write_text(json.dumps(RESULT,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':'passed','original_comments':RESULT['total_original_comments'],'todos':21,
                  'maximum_absolute_errors':maxima,'smokes':[(x['name'],x['actual_steps']) for x in smokes],
                  'qa_json':str(QA/'ws4_code_verification.json')},indent=2))

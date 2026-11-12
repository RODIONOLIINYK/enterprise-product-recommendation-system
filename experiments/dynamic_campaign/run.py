"""P0–P7 campaign. Run from root: python experiments/dynamic_campaign/run.py.

No default aliases are written. Completed matching runs resume by digest;
incomplete runs fail explicitly. Results and predictions remain local until
aggregate reports are exported. This script never invokes held-out evaluation.
"""
import argparse
import gc
import json
from pathlib import Path
import runpy
import torch
from recommender.experiments.dynamic_selector.config import ARTIFACT_ROOT, EVIDENCE_ROOT, SOURCE_EVIDENCE
from recommender.artifacts import atomic_json, file_digest
from recommender.experiment_retrieval import encode_and_score, make_model, CONFIGS, metrics_from_scores
from recommender.experiment_retrieval_data import prepare_retrieval_data
from recommender.experiments.dynamic_selector.data import load_campaign_data, make_slates, UPSTREAM
from recommender.experiments.dynamic_selector.evaluate import ranking_metrics, paired_intervals
from recommender.experiments.dynamic_selector.train import fit_selector, fit_ranker, evaluated_checkpoint
from recommender.train_retriever import get_device

EVIDENCE=EVIDENCE_ROOT
ROOT=ARTIFACT_ROOT


def p0(device):
    path=EVIDENCE/'p0_replay.json'
    if path.exists(): return
    # Redirect the historical replay's sole output without changing its file.
    original=Path('experiments/e1_e5/reference_replay.json');digest=file_digest(original)
    module=runpy.run_path('experiments/e1_e5/replay_references.py')
    function=module['main']
    destination=ROOT/'legacy_reference_replay.json';destination.parent.mkdir(parents=True,exist_ok=True)
    function.__globals__['Path']=lambda value: destination if str(value)==str(original) else Path(value)
    function()
    assert json.loads(destination.read_text())==json.loads(original.read_text()) and digest==file_digest(original)
    train,val,info,ts,vs=prepare_retrieval_data()
    clean={}
    for seed in (42,17,2026):
        directory=UPSTREAM.parent/f'S0-seed{seed}'
        checkpoint=torch.load(directory/'best.pt',map_location='cpu')
        model=make_model(CONFIGS['S0'],info).to(device)
        model.load_state_dict(checkpoint['model_state_dict'],strict=True)
        vectors,customers,scores=encode_and_score(model,val,info,device)
        metrics=metrics_from_scores(scores,val,vs)
        assert metrics==checkpoint['metrics']
        cached=torch.load(directory/'val_purchase_cache.pt',map_location='cpu')
        torch.testing.assert_close(customers,cached['customer_vectors'],rtol=1e-5,atol=2e-6)
        clean[str(seed)]={'sha256':file_digest(directory/'best.pt'),'epoch':checkpoint['epoch'],'metrics':metrics}
    atomic_json({'historical_replay_exact':True,'preserved_report_sha256':digest,'demand_free':clean,'heldout_access':False},path,overwrite=False)


def load_scores(mode,seed,split,data):
    if mode=='dot': return data['dot_scores']
    directory=ROOT/'runs'/f'selector-{mode}-seed{seed}'
    payload=torch.load(directory/f'{split}_scores.pt',map_location='cpu')
    if payload['model_digest']!=file_digest(directory/'best.pt'): raise ValueError('Selector score provenance')
    return payload['scores']


def promising_ranker(result,control):
    a,b=result['metrics'],control['metrics']
    return result['qualified'] and a['new_hits@10']>=b['new_hits@10']+5 and a['recall@10']>=b['recall@10']-.002 and a['ndcg@10']>=b['ndcg@10']-.002


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',default=None)
    parser.add_argument('--phase',choices=('all','p0','selectors','rankers'),default='all')
    args=parser.parse_args();torch.set_num_threads(2);device=get_device(args.device)
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((SOURCE_EVIDENCE/'protocol.json').read_text())
    for filename in ('protocol.json','supplementary_protocol.json'):
        if not (EVIDENCE/filename).exists():atomic_json(json.loads((SOURCE_EVIDENCE/filename).read_text()),EVIDENCE/filename,overwrite=False)
    if args.phase in ('all','p0'): p0(device)
    if args.phase=='p0': return
    train,val,info,manifest=load_campaign_data()
    results={'selectors':{},'rankers':{}}
    dot=ranking_metrics(val['dot_scores'],val)
    results['selectors']['dot']={'metrics':dot,'parameters':0,'training_seconds':0,'frozen_upstream':True}
    for mode in ('static','dynamic','relations'):
        results['selectors'][f'{mode}-42']=fit_selector(mode,42,train,val,manifest,ROOT,device,protocol['selector_epochs'])
        atomic_json(results,EVIDENCE/'progress.json')
    promising=[mode for mode in ('static','dynamic','relations') if any(
        results['selectors'][f'{mode}-42']['metrics'][key]>=dot[key]+10 for key in ('new_hits@25','new_hits@50'))]
    repeated=set(promising)
    if 'dynamic' in repeated or 'relations' in repeated: repeated.add('static')
    for seed in (17,2026):
        for mode in ('static','dynamic','relations'):
            if mode in repeated:
                results['selectors'][f'{mode}-{seed}']=fit_selector(mode,seed,train,val,manifest,ROOT,device,protocol['selector_epochs'])
    # Coverage winner is a screening choice, never an architecture promotion.
    selector=max(('dot','static','dynamic','relations'),key=lambda m:(
        (dot if m=='dot' else results['selectors'][f'{m}-42']['metrics'])['new_hits@25'],
        (dot if m=='dot' else results['selectors'][f'{m}-42']['metrics'])['new_hits@50']))
    decision={'selector_screen_promising':promising,'selector_repeated':sorted(repeated),'selector_screen_winner':selector}
    atomic_json(decision,EVIDENCE/'decisions.json');atomic_json(results,EVIDENCE/'progress.json')
    if args.phase=='selectors': return
    # P3–P5 retain the original personal-repeat allocation, independently of P2.
    from experiments.dynamic_campaign.production_control import slates as personal_slates
    dot_slates=[personal_slates(data,split,info['product_ids']) for split,data in (('train',train),('val',val))]
    policy_slates=[make_slates(data,data['dot_scores'],info['product_ids'],'quota50_25') for data in (train,val)]
    def fit(mode,seed,name='production50',slates=dot_slates):
        result=fit_ranker(mode,seed,*slates,train,val,manifest,ROOT,device,name,protocol['ranker_epochs'])
        results['rankers'][f'{name}-{mode}-{seed}']=result
        atomic_json(results,EVIDENCE/'progress.json')
        return result
    control=fit('control',42)
    policy_control=fit('control',42,'fixed50',policy_slates)
    modes=('no_repeat_flag','no_repeat_group','experts','group_gate','recent_sku','hierarchy','complements',
           'dynamic_features','values_only','bias_and_values','context_direct','context_dot','context_residual')
    screened={mode:fit(mode,42) for mode in modes}
    repeat_rankers=[mode for mode in modes if promising_ranker(screened[mode],control)]
    for seed in (17,2026):
        fit('control',seed)
        for mode in repeat_rankers: fit(mode,seed)
    decision['ranker_promising_repeated']=repeat_rankers
    atomic_json(decision,EVIDENCE/'decisions.json')
    # P2 downstream: matched static / dynamic / relationship candidates, using
    # paired independently trained selector seeds when available.
    policies=('quota50_25','quota50_10','top50','quota100_50')
    p6={}
    for mode in ('static','dynamic','relations'):
        slates=[make_slates(data,load_scores(mode,42,split,data),info['product_ids'],'quota50_25')
                for split,data in (('train',train),('val',val))]
        p6[f'{mode}-quota50_25']=fit('control',42,f'{mode}-quota50_25',slates)
    for policy in policies:
        if selector!='dot' and policy=='quota50_25':
            p6[f'{selector}-{policy}']=p6[f'{selector}-quota50_25'];continue
        if selector=='dot' and policy=='quota50_25':
            p6[f'dot-{policy}']=policy_control;continue
        slates=[make_slates(data,load_scores(selector,42,split,data),info['product_ids'],policy)
                for split,data in (('train',train),('val',val))]
        p6[f'{selector}-{policy}']=fit('control',42,f'{selector}-{policy}',slates)
    # P7 force the proposed combined attention architecture as well as the
    # simpler matched control; retain independent component evidence above.
    best_policy=max(policies,key=lambda policy:(p6[f'{selector}-{policy}']['qualified'],
                    p6[f'{selector}-{policy}']['metrics']['new_hits@10'],p6[f'{selector}-{policy}']['metrics']['ndcg@10']))
    decision['combined_selector']=selector;decision['combined_policy']=best_policy
    for seed in (42,17,2026):
        if selector!='dot' and f'{selector}-{seed}' not in results['selectors']:
            results['selectors'][f'{selector}-{seed}']=fit_selector(selector,seed,train,val,manifest,ROOT,device,protocol['selector_epochs'])
        slates=[make_slates(data,load_scores(selector,seed,split,data),info['product_ids'],best_policy)
                for split,data in (('train',train),('val',val))]
        fit('control',seed,'combined',slates)
        fit('combined_direct',seed,'combined',slates)
    # Repeated P6 configurations that pass screening, with paired selector seeds.
    for name,result in p6.items():
        if not promising_ranker(result,control): continue
        mode,policy=name.split('-',1)
        for seed in (17,2026):
            if mode!='dot' and f'{mode}-{seed}' not in results['selectors']:
                results['selectors'][f'{mode}-{seed}']=fit_selector(mode,seed,train,val,manifest,ROOT,device,protocol['selector_epochs'])
            slates=[make_slates(data,load_scores(mode,seed,split,data),info['product_ids'],policy)
                    for split,data in (('train',train),('val',val))]
            fit('control',seed,name,slates)
    comparisons={}
    for key,result in results['rankers'].items():
        seed=result['settings']['seed']
        baseline=ROOT/'runs'/f'production50-control-seed{seed}'/'predictions.npz'
        predicted=evaluated_checkpoint(result).parent/'predictions.npz'
        comparisons[key]=paired_intervals(baseline,predicted)
    atomic_json(comparisons,EVIDENCE/'paired_intervals.json')
    decision.update(status='complete',promotion=False,test_access=False)
    atomic_json(decision,EVIDENCE/'decisions.json');atomic_json(results,EVIDENCE/'results.json')
    (EVIDENCE/'progress.json').unlink(missing_ok=True)
    gc.collect()
    print('P0–P7 fits complete. Generate telemetry and final report next.',flush=True)


if __name__=='__main__':main()

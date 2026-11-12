"""Aggregate every selected seed, paired intervals and transparent selection rules."""
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
import torch
from recommender.experiments.dynamic_selector.config import ARTIFACT_ROOT, EVIDENCE_ROOT
from recommender.artifacts import atomic_json, file_digest
from recommender.experiments.dynamic_selector.data import load_campaign_data
from recommender.experiments.dynamic_selector.train import evaluated_checkpoint
from recommender.experiments.dynamic_selector.evaluate import ranking_metrics, paired_intervals

EVIDENCE=EVIDENCE_ROOT
ROOT=ARTIFACT_ROOT


def aggregate(records):
    output={}
    names=('recall@10','ndcg@10','new_hits@10','new_recall@10','conditional_new_recall@10',
           'repeat_recall@10_micro','repeat_recall@10_macro','new_hits@25','new_hits@50','new_hits@100',
           'mixed_coverage_micro','new_targets_in_slate')
    for name in names:
        values=[r.get('metrics_final',r['metrics'])[name] for r in records if name in r.get('metrics_final',r['metrics'])]
        if values:output[name]={'mean':float(np.mean(values)),'std':float(np.std(values,ddof=1)) if len(values)>1 else None}
    output['seeds']=[r.get('settings',{}).get('seed') for r in records]
    output['n']=len(records)
    for name in ('training_seconds','parameters','trainable_parameters'):
        values=[r[name] for r in records if name in r]
        if values:output[name]={'mean':float(np.mean(values)),'std':float(np.std(values,ddof=1)) if len(values)>1 else None}
    return output


def nested_intervals(controls,treatments,repeats=2000):
    a,b=[np.load(p) for p in controls],[np.load(p) for p in treatments]
    for row in (*a,*b):
        if not np.array_equal(row['customers'],a[0]['customers']) or not np.array_equal(row['new_targets'],a[0]['new_targets']):raise ValueError('Paired row order/targets')
    _,groups=np.unique(a[0]['customers'],return_inverse=True);n=groups.max()+1
    row_counts=np.bincount(groups);target_counts=np.bincount(groups,weights=a[0]['new_targets'])
    deltas={key:np.stack([np.bincount(groups,weights=y[key]-x[key]) for x,y in zip(a,b)]) for key in ('recall','ndcg','new_hits')}
    samples={name:[] for name in deltas};rng=np.random.default_rng(123)
    for _ in range(repeats):
        seeds=rng.integers(len(a),size=len(a));customers=rng.integers(n,size=n)
        for name,values in deltas.items():
            denominator=(target_counts if name=='new_hits' else row_counts)[customers].sum()
            samples[name].append(values[seeds][:,customers].mean(0).sum()/max(denominator,1))
    return {name:np.quantile(values,[.025,.5,.975]).tolist() for name,values in samples.items()}


def main():
    torch.set_num_threads(2)
    results=json.loads((EVIDENCE/'results_enriched.json').read_text())
    decisions=json.loads((EVIDENCE/'decisions.json').read_text())
    train,val,info,manifest=load_campaign_data(build=False)
    selectors=defaultdict(list);rankers=defaultdict(list)
    for key,result in results['selectors'].items():
        if key=='dot':selectors['dot'].append(result)
        else:selectors[result['settings']['mode']].append(result)
    for key,result in results['rankers'].items():
        rankers[key.rsplit('-',1)[0]].append(result)
    for values in (*selectors.values(),*rankers.values()):values.sort(key=lambda r:r.get('settings',{}).get('seed',0))
    summary={'selectors':{key:aggregate(value) for key,value in selectors.items()},
             'rankers':{key:aggregate(value) for key,value in rankers.items()},'nested_paired':{},
             'identity':manifest['identity'],'promoted':False,'heldout_access':False}
    for key,records in rankers.items():
        seeds=[r['settings']['seed'] for r in records]
        control_name='production50-control'
        by_seed={r['settings']['seed']:r for r in rankers[control_name]}
        if any(seed not in by_seed for seed in seeds):continue
        controls=[evaluated_checkpoint(by_seed[seed]).parent/'predictions.npz' for seed in seeds]
        treatments=[evaluated_checkpoint(r).parent/'predictions.npz' for r in records]
        summary['nested_paired'][key]={'control':control_name,'seeds':seeds,
                                     'intervals':nested_intervals(controls,treatments)}
    stage1_paths={}
    for key,result in results['selectors'].items():
        if key=='dot':scores=val['dot_scores'];directory=ROOT
        else:
            directory=evaluated_checkpoint(result).parent
            scores=torch.load(directory/'val_scores.pt',map_location='cpu')['scores']
        # Dedicated Top-25/50 paired target outcomes, all 1,857 targets retained.
        ranked=scores.masked_fill(~val['unseen'],-torch.inf).argsort(dim=-1,descending=True,stable=True)
        outcome=val['new_target'].gather(1,ranked)&val['unseen'].gather(1,ranked)
        path=directory/'stage1_predictions.npz'
        np.savez_compressed(path,customers=np.array(val['source']['customer_ids']),
            recall=outcome[:,:25].sum(1).numpy(),ndcg=outcome[:,:50].sum(1).numpy(),
            new_hits=outcome[:,:25].sum(1).numpy(),
            new_targets=np.array([len(x) for x in val['source']['new_positive_pids']]))
        stage1_paths[key]=path
        result['metrics_final']=ranking_metrics(scores,val)
    summary['stage1_paired_vs_dot']={}
    for key,path in stage1_paths.items():
        if key=='dot':continue
        values=paired_intervals(stage1_paths['dot'],path)
        summary['stage1_paired_vs_dot'][key]={'new_recall@25_delta':values['new_recall@10_delta'],
            'macro_new_hits@25_delta':values['recall@10_delta'],'macro_new_hits@50_delta':values['ndcg@10_delta']}
    if 'combined-combined_direct' in rankers:
        controls={r['settings']['seed']:evaluated_checkpoint(r).parent/'predictions.npz' for r in rankers['combined-control']}
        treatments=rankers['combined-combined_direct']
        summary['context_contribution_paired']=nested_intervals([controls[r['settings']['seed']] for r in treatments],
            [evaluated_checkpoint(r).parent/'predictions.npz' for r in treatments])
    if 'finetune50-context_residual' in rankers:
        controls={r['settings']['seed']:evaluated_checkpoint(r).parent/'predictions.npz' for r in rankers['finetune50-control']}
        treatments=rankers['finetune50-context_residual']
        summary['pretrained_residual_contribution_paired']=nested_intervals([controls[r['settings']['seed']] for r in treatments],
            [evaluated_checkpoint(r).parent/'predictions.npz' for r in treatments])
    summary['depth_matched_factorial_paired']={}
    summary['incremental_acceptance']={}
    for name,treatment,control_name in (
        ('MLP_selector_at100','dynamic-quota100_50-control','dot100-control'),
        ('ranker_signals_with_dot100','dot100-dynamic_features','dot100-control'),
        ('ranker_signals_with_MLP100','dynamic100-dynamic_features','dynamic-quota100_50-control'),
        ('MLP_selector_with_ranker_signals','dynamic100-dynamic_features','dot100-dynamic_features')):
        controls={r['settings']['seed']:evaluated_checkpoint(r).parent/'predictions.npz' for r in rankers[control_name]}
        records=rankers[treatment]
        summary['depth_matched_factorial_paired'][name]=nested_intervals([controls[r['settings']['seed']] for r in records],
            [evaluated_checkpoint(r).parent/'predictions.npz' for r in records])
        treatment_stats=summary['rankers'][treatment];control_stats=summary['rankers'][control_name]
        mean_gain=treatment_stats['new_hits@10']['mean']-control_stats['new_hits@10']['mean']
        summary['incremental_acceptance'][name]={'control':control_name,'treatment':treatment,
            'mean_new_hit_gain':mean_gain,'passes':bool(all(r['qualified'] for r in records) and mean_gain>=10
                and treatment_stats['recall@10']['mean']>=control_stats['recall@10']['mean']
                and treatment_stats['ndcg@10']['mean']>=control_stats['ndcg@10']['mean']
                and summary['depth_matched_factorial_paired'][name]['new_hits'][0]>0)}
    control=summary['rankers']['production50-control']
    final=[]
    for key,values in summary['rankers'].items():
        if values['n']<3 or key=='production50-control':continue
        ci=summary['nested_paired'][key]['intervals']['new_hits']
        # Architecture selection compares every repeated configuration against
        # the common fixed-slate control; context contribution is separate above.
        meets=(all(r.get('qualified',True) for r in rankers[key]) and values['new_hits@10']['mean']>=control['new_hits@10']['mean']+10 and
               values['recall@10']['mean']>=control['recall@10']['mean'] and
               values['ndcg@10']['mean']>=control['ndcg@10']['mean'] and ci[0]>0)
        final.append({'configuration':key,'passes_exploratory_guardrails':bool(meets),
                      'mean_new_hit_gain':values['new_hits@10']['mean']-control['new_hits@10']['mean'],
                      'mean_exceeds_preserved_discovery_on_three_metrics':bool(values['new_hits@10']['mean']>209 and
                          values['recall@10']['mean']>=.7615051078117159 and values['ndcg@10']['mean']>=.5918928743676547),
                      'interval_control':summary['nested_paired'][key]['control']})
    summary['acceptance_audit']=final
    summary['source_sha256']={str(path):file_digest(path) for base in ('recommender/experiments/dynamic_selector',
        'recommender/experiments/candidate_context','experiments/dynamic_campaign') for path in Path(base).glob('*.py')}
    summary['source_sha256']['recommender/ranker.py']=file_digest('recommender/ranker.py')
    summary['recommendation']={'conservative_architecture':'demand-disabled single-vector dot retrieval, 100 mixed candidates, original UPS-CANR',
        'optional_candidate':'added scalar ranker features; highest-mean configurations remain experimental',
        'avoid_automatic_additions':['full-pool MLP selector','candidate self-attention'],'promotion':False}
    atomic_json(summary,EVIDENCE/'summary.json')
    atomic_json(results,EVIDENCE/'results_enriched.json')
    lines=['# Selected-checkpoint comparison','',
           'Mean ± sample standard deviation across all repeated seeds. A dash means single-seed screening, not zero uncertainty. No promotion.', '',
           '| Configuration | n | New hits@10 | Recall@10 | NDCG@10 | Conditional new recall | Repeat recall (micro) | New targets in slate |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    def format_value(record,key):
        value=record[key];return f"{value['mean']:.4f}"+(f" ± {value['std']:.4f}" if value['std'] is not None else ' (screen)')
    for key,value in summary['rankers'].items():
        lines.append('| '+key+' | '+str(value['n'])+' | '+' | '.join(format_value(value,name) for name in
            ('new_hits@10','recall@10','ndcg@10','conditional_new_recall@10','repeat_recall@10_micro','new_targets_in_slate'))+' |')
    lines.extend(['','## Full-pool selectors','', '| Selector | n | Unseen hits@25 | @50 | @100 | Recall@10 | NDCG@10 |', '|---|---:|---:|---:|---:|---:|---:|'])
    for key,value in summary['selectors'].items():
        lines.append('| '+key+' | '+str(value['n'])+' | '+' | '.join(format_value(value,name) for name in
            ('new_hits@25','new_hits@50','new_hits@100','recall@10','ndcg@10'))+' |')
    (EVIDENCE/'COMPARISON.md').write_text('\n'.join(lines)+'\n')
    lines=['# Fit time, parameter allocation and forward latency','',
        'Time is mean ± sample SD in seconds per completed fit, including per-epoch validation. Parameters exclude the shared frozen S0 encoder. Active means a gradient tensor was present on one diagnostic batch; it does not prove every entry has nonzero gradient. Latencies average per-seed batch-1 medians, in milliseconds; full-request timings are in telemetry.json.', '',
        '| Configuration | n | Fit seconds | Allocated parameters | Gradient-active parameters | CPU forward median | MPS forward median |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for key,records in rankers.items():
        values=summary['rankers'][key];active=np.mean([r['active_gradient_parameters'] for r in records])
        def forward(device):
            samples=[r['forward_latency_batch1'][device]['median_ms'] for r in records if device in r['forward_latency_batch1']]
            return f'{np.mean(samples):.3f}' if samples else '—'
        lines.append(f'| {key} | {len(records)} | {format_value(values,"training_seconds")} | {values["parameters"]["mean"]:.0f} | {active:.0f} | {forward("cpu")} | {forward("mps")} |')
    lines.extend(['','| Full-pool selector | n | Fit seconds | Parameters |','|---|---:|---:|---:|'])
    for key,values in summary['selectors'].items():
        lines.append(f'| {key} | {values["n"]} | {format_value(values,"training_seconds")} | {values["parameters"]["mean"]:.0f} |')
    (EVIDENCE/'RESOURCES.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'fits':len(results['selectors'])-1+len(results['rankers']),'acceptance':final,'decisions':decisions},indent=2))


if __name__=='__main__':main()

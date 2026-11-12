"""Additional matched-slate diagnostic requested by the later review.

Run after the main campaign. Does not replace or modify any existing fit.
"""
import json
import torch
from recommender.experiments.dynamic_selector.config import ARTIFACT_ROOT, EVIDENCE_ROOT
from recommender.artifacts import atomic_json
from recommender.experiments.dynamic_selector.data import load_campaign_data
from recommender.experiments.dynamic_selector.train import fit_ranker
from recommender.train_retriever import get_device
from experiments.dynamic_campaign.production_control import slates
from experiments.dynamic_campaign.run import promising_ranker


def main():
    torch.set_num_threads(2);device=get_device()
    evidence=EVIDENCE_ROOT;root=ARTIFACT_ROOT
    results=json.loads((evidence/'results.json').read_text())
    train,val,info,manifest=load_campaign_data(build=False)
    from experiments.dynamic_campaign.depth_control import run as depth
    atomic_json(depth(results,train,val,info,manifest,root,device),evidence/'depth_decision.json')
    ts,vs=slates(train,'train',info['product_ids']),slates(val,'val',info['product_ids'])
    control=results['rankers']['production50-control-42']['metrics']
    supplementary=[]
    for key,candidate in list(results['rankers'].items()):
        if not key.startswith('production50-') or candidate['settings']['seed']!=42 or candidate['settings']['mode']=='control':continue
        metrics=candidate['metrics']
        if (candidate.get('qualified') and metrics['new_hits@10']>=control['new_hits@10']+10
                and metrics['recall@10']>=control['recall@10']-.003 and metrics['ndcg@10']>=control['ndcg@10']-.002):
            mode=candidate['settings']['mode'];supplementary.append(mode)
            for seed in (17,2026):
                results['rankers'][f'production50-{mode}-{seed}']=fit_ranker(mode,seed,ts,vs,train,val,manifest,root,device,'production50',10)
            atomic_json(results,evidence/'results.json')
    result=fit_ranker('pointwise',42,ts,vs,train,val,manifest,root,device,'production50',10)
    results['rankers']['production50-pointwise-42']=result
    reference=results['rankers']['production50-control-42']['metrics'];metric=result['metrics']
    lower_cost_noninferior=(result['qualified'] and metric['new_hits@10']>=reference['new_hits@10']-5
        and metric['recall@10']>=reference['recall@10']-.001 and metric['ndcg@10']>=reference['ndcg@10']-.001)
    promising=promising_ranker(result,results['rankers']['production50-control-42']) or lower_cost_noninferior
    if promising:
        for seed in (17,2026):
            results['rankers'][f'production50-pointwise-{seed}']=fit_ranker('pointwise',seed,ts,vs,train,val,manifest,root,device,'production50',10)
    atomic_json(results,evidence/'results.json')
    atomic_json({'reason':'later review requested a cheap matched-50-item scoring control',
                 'screen_seed':42,'promising_repeated':promising,'lower_cost_noninferior_screen':lower_cost_noninferior,
                 'supplementary_near_miss_repeats':supplementary},evidence/'pointwise_decision.json')
    from experiments.dynamic_campaign.finetune_control import run as finetune
    decision=finetune(results,train,val,manifest,ts,vs,root,device)
    atomic_json(results,evidence/'results.json')
    atomic_json(decision,evidence/'finetune_decision.json')


if __name__=='__main__':main()

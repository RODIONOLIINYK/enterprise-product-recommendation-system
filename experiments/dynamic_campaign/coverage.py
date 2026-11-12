"""All selector/policy coverage ceilings before downstream fitting."""
import torch
from recommender.experiments.dynamic_selector.config import ARTIFACT_ROOT, EVIDENCE_ROOT
from recommender.artifacts import atomic_json,file_digest
from recommender.experiments.dynamic_selector.data import load_campaign_data,make_slates
from recommender.experiments.dynamic_selector.evaluate import ranking_metrics
from experiments.dynamic_campaign.production_control import slates as personal_slates


def main():
    torch.set_num_threads(2)
    train,val,info,manifest=load_campaign_data(build=False)
    results={}
    original=personal_slates(val,'val',info['product_ids'])
    results['dot-personal-repeat']=ranking_metrics(val['dot_scores'].gather(1,original['candidate_indices']),val,original['candidate_indices'])
    for mode in ('dot','static','dynamic','relations'):
        if mode=='dot':scores=val['dot_scores']
        else:
            directory=ARTIFACT_ROOT/'runs'/f'selector-{mode}-seed42'
            payload=torch.load(directory/'val_scores.pt',map_location='cpu')
            assert payload['identity']==manifest['identity'] and payload['model_digest']==file_digest(directory/'best.pt')
            scores=payload['scores']
        for policy in ('quota50_25','quota50_10','top50','quota100_50'):
            slate=make_slates(val,scores,info['product_ids'],policy)
            results[f'{mode}-{policy}']=ranking_metrics(scores.gather(1,slate['candidate_indices']),val,slate['candidate_indices'])
    atomic_json({'identity':manifest['identity'],'direct_selector_metrics_before_UPSCANR':results,
                 'heldout_access':False},EVIDENCE_ROOT/'coverage.json',overwrite=False)
    print('Full selector/policy candidate coverage and direct-score metrics recorded.',flush=True)


if __name__=='__main__':main()

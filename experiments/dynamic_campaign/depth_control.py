"""Depth-matched P7 factorial: selector versus scalar ranker information."""
from pathlib import Path
import torch
from recommender.experiments.dynamic_selector.config import EVIDENCE_ROOT
from recommender.artifacts import atomic_json,file_digest
from recommender.experiments.dynamic_selector.data import make_slates
from recommender.experiments.dynamic_selector.train import fit_ranker


def run(results,train,val,info,manifest,root,device):
    for selector,mode,name in (('dot','control','dot100'),('dot','dynamic_features','dot100'),
                               ('dynamic','dynamic_features','dynamic100')):
        for seed in (42,17,2026):
            slates=[]
            for split,data in (('train',train),('val',val)):
                if selector=='dot':scores=data['dot_scores']
                else:
                    directory=Path(root)/'runs'/f'selector-{selector}-seed{seed}'
                    payload=torch.load(directory/f'{split}_scores.pt',map_location='cpu')
                    if payload['model_digest']!=file_digest(directory/'best.pt') or payload['identity']!=manifest['identity']:
                        raise ValueError('Depth control selector provenance')
                    scores=payload['scores']
                slates.append(make_slates(data,scores,info['product_ids'],'quota100_50'))
            result=fit_ranker(mode,seed,*slates,train,val,manifest,root,device,name,10)
            results['rankers'][f'{name}-{mode}-{seed}']=result
            atomic_json(results,EVIDENCE_ROOT/'results.json')
    return {'purpose':'separate wider slates, learned selection, and added scalar information',
            'depth':100,'repeat_quota':50,'seeds':[42,17,2026],'selection':'unchanged simultaneous quality floors'}

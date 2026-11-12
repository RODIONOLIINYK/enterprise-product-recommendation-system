"""Additional matched control retaining the original personal-repeat policy.

Run after run.py. Original S0 slates supply candidate IDs only; new feature
normalization, train cohort, initialization and budgets match the campaign.
This quantifies effects of repeat ordering separately from selector scoring.
"""
import json
import numpy as np
import torch
from recommender.experiments.dynamic_selector.config import ARTIFACT_ROOT, EVIDENCE_ROOT
from recommender.artifacts import atomic_json, json_digest
from recommender.experiments.dynamic_selector.data import load_campaign_data, UPSTREAM
from recommender.experiments.dynamic_selector.train import fit_ranker
from recommender.train_retriever import get_device


def slates(data, split, product_ids):
    original=torch.load(UPSTREAM/f'stage2_{split}.pt',map_location='cpu')
    lookup={key:i for i,key in enumerate(original['transition_keys'])}
    index={sku:i for i,sku in enumerate(product_ids)}
    rows=[lookup[key] for key in data['source']['transition_keys']]
    ids=torch.tensor([[index[sku] for sku in original['candidate_pids'][row]] for row in rows])
    if not bool(data['allowed'].gather(1,ids).all()):raise ValueError('Historical policy eligibility mismatch')
    result=dict(data['source'])
    result.update(candidate_indices=ids,candidate_vectors=None,
                  candidate_pids=[[product_ids[i] for i in row] for row in ids.tolist()],
                  labels=data['target'].gather(1,ids).float(),stage1_customer_vectors=data['u'],
                  pairwise_features=torch.from_numpy(np.array(data['features'][np.arange(len(ids))[:,None],ids.numpy()],dtype=np.float32)),
                  cache_identity=json_digest({'policy':'original_personal_repeat','ids':ids.tolist()}))
    return result


def main():
    torch.set_num_threads(2);device=get_device()
    root=ARTIFACT_ROOT;evidence=EVIDENCE_ROOT
    results=json.loads((evidence/'results.json').read_text())
    train,val,info,manifest=load_campaign_data(build=False)
    ts,vs=slates(train,'train',info['product_ids']),slates(val,'val',info['product_ids'])
    for seed in (42,17,2026):
        result=fit_ranker('control',seed,ts,vs,train,val,manifest,root,device,'personal-repeat',10)
        results['rankers'][f'personal-repeat-control-{seed}']=result
        atomic_json(results,evidence/'results.json')


if __name__=='__main__':main()

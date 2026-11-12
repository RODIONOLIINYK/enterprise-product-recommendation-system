"""Final read-only source/artifact/cache audit; never open held-out inputs."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from recommender.experiments.dynamic_selector.config import EVIDENCE_ROOT
from recommender.artifacts import atomic_json,file_digest
from recommender.catalogue import load_product_catalogue
from recommender.company import load_companies
from recommender.development import load_development_purchases
from recommender.experiment_retrieval import make_model,CONFIGS,encode_and_score
from recommender.experiment_retrieval_data import prepare_retrieval_data
from recommender.experiments.dynamic_selector.data import load_campaign_data,UPSTREAM,CACHE
from recommender.experiments.dynamic_selector.feature_builder import DynamicFeatureBuilder
from recommender.experiments.dynamic_selector.feature_schema import quantize_features
from recommender.experiments.dynamic_selector.train import evaluated_checkpoint
from recommender.scope import get_datasets_locs


def main():
    torch.set_num_threads(2);root=EVIDENCE_ROOT
    train,val,info,manifest=load_campaign_data(build=False)
    assert not info['product_inputs']['demand_features'][:,:6].any()
    inventory=json.loads(Path('audit/artifacts.json').read_text());preserved={}
    for name,value in inventory.items():
        if 'sha256' not in value:continue
        assert name.endswith('.pt') and ('retriever' in name or '/ranker/' in name or 'slates_v3' in name)
        assert file_digest(name)==value['sha256'];preserved[name]=value['sha256']
    assert len(preserved)==7
    p0=json.loads((root/'p0_replay.json').read_text())
    assert file_digest('experiments/e1_e5/reference_replay.json')==p0['preserved_report_sha256']
    raw_train,raw_val,raw_info,ts,vs=prepare_retrieval_data()
    device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model=make_model(CONFIGS['S0'],raw_info).to(device)
    model.load_state_dict(torch.load(UPSTREAM/'best.pt',map_location='cpu')['model_state_dict'])
    replay={}
    for split,data in (('train',raw_train),('val',raw_val)):
        v,u,scores=encode_and_score(model,data,raw_info,device)
        cached=torch.load(UPSTREAM/f'{split}_purchase_cache.pt',map_location='cpu')
        torch.testing.assert_close(u,cached['customer_vectors'],atol=2e-6,rtol=1e-5)
        torch.testing.assert_close(v,cached['product_vectors'],atol=2e-6,rtol=1e-5)
        torch.testing.assert_close(scores,cached['scores'],atol=2e-4,rtol=1e-5)
        replay[split]={'max_customer_difference':float((u-cached['customer_vectors']).abs().max()),
                       'max_score_difference':float((scores-cached['scores']).abs().max()),
                       'cache_sha256':file_digest(UPSTREAM/f'{split}_purchase_cache.pt')}
    purchases=load_development_purchases('data/interim/cleaned_purchases.csv')
    builder=DynamicFeatureBuilder(purchases,load_product_catalogue(get_datasets_locs('items')),
        load_companies(purchases=purchases,minimum_purchases=1),info['product_ids'])
    queries=sorted((int(data['source']['prediction_days'][row]),split,row) for split,data in enumerate((train,val))
                    for row in [0,1,100,len(data['u'])-1])
    for day,split,row in queries:
        data=(train,val)[split];date=pd.Timestamp('2025-01-01')+pd.Timedelta(days=day)
        np.testing.assert_array_equal(quantize_features(builder.build(data['source']['customer_ids'][row],date)),data['features'][row].astype(np.float32))
    results=json.loads((root/'results_enriched.json').read_text());runs={}
    for group in ('selectors','rankers'):
        for name,result in results[group].items():
            if name=='dot':continue
            path=evaluated_checkpoint(result)
            run=json.loads((path.parent/'run.json').read_text());payload=torch.load(path,map_location='cpu')
            assert payload['settings']==result['settings'] and payload['epoch']==result['epoch']
            assert payload['settings']['identity']==manifest['identity']
            assert payload['settings']['upstream_digest']==manifest['upstream_digest']
            key='selected_sha256' if result.get('qualified',True) else 'diagnostic_sha256'
            assert file_digest(path)==run[key]
            if not result.get('qualified',True):assert run['selected_checkpoint'] is None
            runs[f'{group}/{name}']={'sha256':run[key],'qualified':result.get('qualified',True)}
    output={'heldout_access':False,'default_promoted':False,'original_artifact_hashes_unchanged':preserved,
            'reference_report_unchanged':True,'upstream_full_cache_replay':replay,'request_feature_exact_rows':len(queries),
            'feature_payload_sha256':manifest['feature_sha256'],'feature_manifest_sha256':file_digest(CACHE/'manifest.json'),
            'run_checkpoint_checks':runs,'input_artifact_sha256':{str(path):file_digest(path) for path in
                [UPSTREAM/'best.pt',UPSTREAM/'stage2_train.pt',UPSTREAM/'stage2_val.pt',Path('data/interim/shared_stage1_v1/train.pt'),Path('data/interim/shared_stage1_v1/val.pt')]}}
    atomic_json(output,root/'verification.json')
    print(f'Verified {len(runs)} fit artifacts, complete upstream caches and eight exact request rows; original artifacts unchanged.')


if __name__=='__main__':main()

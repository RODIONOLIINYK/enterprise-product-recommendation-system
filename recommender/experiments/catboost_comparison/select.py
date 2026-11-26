"""Compare CatBoost prefixes under the same discovery/NDCG selection as neural fits.

Evaluate at 50-tree intervals (at most ten checkpoints), independently of historical
Logloss selection. Both views are retained; no original model artifact is overwritten.
"""

import json
from pathlib import Path
import numpy as np
import torch

from recommender.artifacts import atomic_json,file_digest
from recommender.catalogue import load_product_catalogue
from recommender.development import load_development_purchases
from recommender.experiments.rolling_candidate_v2.data import (
    prepare_or_load_rolling_feature_bundle,prepare_or_load_rolling_selector_scores,prepare_or_load_rolling_slate_cache,
)
from recommender.experiments.rolling_candidate_v2.evaluate import evaluate_rolling_slate_metrics,paired_rolling_intervals
from recommender.scope import get_datasets_locs
from .features import CATEGORICAL,build_legacy_numeric,categorical_frame,legacy_frame
from .run import EVIDENCE,modern_frame,validate_dates


def run():
    from catboost import CatBoostClassifier,Pool
    summary=json.loads((EVIDENCE/'summary.json').read_text())
    data,info,_,meta=prepare_or_load_rolling_feature_bundle(build=False)
    prepare_or_load_rolling_selector_scores(data,info,meta)
    slates,_=prepare_or_load_rolling_slate_cache(data,info,meta);validate_dates(slates)
    slate=slates['val'];indices=slate['candidate_indices'];pids=info['product_ids']
    catalogue=load_product_catalogue(get_datasets_locs('items'))
    purchases=load_development_purchases('data/interim/cleaned_purchases.csv')
    baseline=summary['neural_controls']['arm_a_control_pointwise_425d']
    guard_r=0.7558833119851873;guard_n=0.5918928743676547
    selected={}
    for name,entry in summary['models'].items():
        if name=='historical17':
            dates=np.datetime64('2025-01-01')+slate['prediction_days'].numpy().astype('timedelta64[D]')
            values=build_legacy_numeric(purchases,catalogue,pids,slate['customer_ids'],dates,indices.numpy())
            frame=legacy_frame(values,categorical_frame(pids,catalogue,indices.numpy()))
        else:frame=modern_frame(slate,data['val'],catalogue,pids)
        pool=Pool(frame,cat_features=list(CATEGORICAL))
        model=CatBoostClassifier();model.load_model(entry['model_path'])
        root=Path(entry['model_path']).parent
        best={'ndcg':None,'discovery':None};history=[]
        predictions=model.staged_predict(pool,prediction_type='RawFormulaVal',eval_period=50,thread_count=4)
        for step,raw in enumerate(predictions,1):
            trees=min(step*50,model.tree_count_)
            scores=torch.tensor(np.asarray(raw).reshape(indices.shape),dtype=torch.float32)
            metrics=evaluate_rolling_slate_metrics(scores,data['val'],indices)
            qualified=metrics['recall@10']>=guard_r and metrics['ndcg@10']>=guard_n
            keys={'ndcg':(metrics['ndcg@10'],metrics['recall@10'],metrics['new_hits@10']),
                  'discovery':(int(qualified),metrics['new_hits@10'],metrics['ndcg@10'])}
            history.append({'trees':trees,'metrics':metrics,'qualified':qualified})
            for selection,key in keys.items():
                if best[selection] is None or key>best[selection]['key']:
                    best[selection]={'key':key,'trees':trees,'metrics':metrics,'scores':scores,'qualified':qualified}
        selected[name]={'history':history}
        for selection,result in best.items():
            model_path=root/f'best_{selection}.cbm';npz_path=root/f'{selection}_predictions.npz'
            copy=model.copy();copy.shrink(ntree_end=result['trees']);copy.save_model(str(model_path))
            metrics=evaluate_rolling_slate_metrics(result['scores'],data['val'],indices,save_path=npz_path)
            loaded=CatBoostClassifier();loaded.load_model(str(model_path))
            replay=loaded.predict(pool,prediction_type='RawFormulaVal',thread_count=4).reshape(indices.shape)
            np.testing.assert_allclose(replay,result['scores'].numpy(),rtol=0,atol=1e-5)
            selected[name][selection]={'trees':result['trees'],'metrics':metrics,'qualified':result['qualified'],
                                       'checkpoint':str(model_path),'checkpoint_sha256':file_digest(model_path),
                                       'predictions_npz':str(npz_path),'predictions_sha256':file_digest(npz_path),
                                       'A_to_tree_95ci':paired_rolling_intervals(baseline[selection]['predictions_npz'],npz_path)}
    atomic_json({'selection_protocol':'50-tree prefixes; identical neural discovery/NDCG keys and guardrails',
                 'heldout_access':False,'promoted':False,'models':selected},EVIDENCE/'selected_checkpoints.json',overwrite=False)
    print(json.dumps({n:{s:{'trees':r[s]['trees'],'new':r[s]['metrics']['new_hits@10'],'recall':r[s]['metrics']['recall@10'],'ndcg':r[s]['metrics']['ndcg@10']} for s in ('ndcg','discovery')} for n,r in selected.items()},indent=2))


if __name__=='__main__':run()

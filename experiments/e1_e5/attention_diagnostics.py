"""Quantify whether E3's learned older-event pooling differs from mean pooling."""
import json
from pathlib import Path
import math
import torch
from recommender.experiment_retrieval import OUTPUT
from recommender.experiment_retrieval_data import CACHE


def main():
    torch.set_num_threads(2)
    data=torch.load(CACHE/'val.pt',map_location='cpu')
    ids,counts=data['older_ids'],data['older_counts']
    mask=counts>0; active=mask.any(1)
    baseline=counts/counts.sum(1,keepdim=True).clamp_min(1)
    results={}
    with torch.no_grad():
        for path in sorted(OUTPUT.glob('E3attention-*/best.pt')):
            state=torch.load(path,map_location='cpu')['model_state_dict']
            cache=torch.load(path.parent/'val_purchase_cache.pt',map_location='cpu')
            v=cache['product_vectors']
            score=(v@state['long_key.weight'].T@state['long_query'].T).squeeze(-1)/math.sqrt(128)
            score=torch.nn.functional.pad(score,(1,0))[ids]
            weight=(score+counts.clamp_min(1).log()).masked_fill(~mask,-1e9).softmax(-1)*mask
            weight/=weight.sum(1,keepdim=True).clamp_min(1e-9)
            total_variation=(weight-baseline).abs().sum(1)*.5
            kl=(weight*(weight.clamp_min(1e-12).log()-baseline.clamp_min(1e-12).log())).sum(1)
            results[path.parent.name]={'validation_rows_with_older_history':int(active.sum()),
                'query_norm':float(state['long_query'].norm()),'key_weight_norm':float(state['long_key.weight'].norm()),
                'mean_total_variation_from_frequency_pooling':float(total_variation[active].mean()),
                'max_total_variation_from_frequency_pooling':float(total_variation[active].max()),
                'mean_kl_to_frequency_pooling_nats':float(kl[active].mean()),
                'global_sku_attention_logit_std':float(score[mask].std())}
    Path('experiments/e1_e5/attention_diagnostics.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))


if __name__=='__main__': main()

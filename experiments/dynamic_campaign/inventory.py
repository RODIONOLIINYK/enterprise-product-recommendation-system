"""Export aggregate development feature availability and temporal contracts."""
import numpy as np
import pandas as pd
import torch
from recommender.experiments.dynamic_selector.config import EVIDENCE_ROOT
from recommender.artifacts import atomic_json
from recommender.candidates import PointInTimeFeatureStore
from recommender.catalogue import load_product_catalogue
from recommender.company import load_companies
from recommender.development import load_development_purchases
from recommender.scope import get_datasets_locs
from recommender.split import TRAIN_CUTOFF
from recommender.experiments.dynamic_selector.data import load_campaign_data
from recommender.experiments.dynamic_selector.feature_builder import DynamicFeatureBuilder
from recommender.experiments.dynamic_selector.feature_schema import NAMES, contract


def main():
    torch.set_num_threads(2)
    train,val,info,manifest=load_campaign_data(build=False)
    purchases=load_development_purchases('data/interim/cleaned_purchases.csv')
    catalogue=load_product_catalogue(get_datasets_locs('items'))
    companies=load_companies(purchases=purchases,minimum_purchases=1)
    policies={name:PointInTimeFeatureStore(purchases,catalogue,companies,metadata_policy=name)
              for name in ('snapshot_unverified','observed_only_v1')}
    availability={}
    for name,store in policies.items():
        collected=[]
        for row in range(min(256,len(val['u']))):
            date=pd.Timestamp('2025-01-01')+pd.Timedelta(days=int(val['source']['prediction_days'][row]))
            ids=torch.where(val['allowed'][row])[0].tolist()
            collected.append(store.compute_slate_pairwise_features(val['source']['customer_ids'][row],date,
                [info['product_ids'][i] for i in ids]).numpy())
        values=np.concatenate(collected)
        availability[name]={'rows_sampled':len(collected),'eligible_pairs':len(values),
                            'nonzero_fraction':{key:float((values[:,i]!=0).mean()) for i,key in enumerate(NAMES[:13])}}
    counts=np.zeros(len(NAMES));eligible=0
    for start in range(0,len(val['u']),128):
        allowed=val['allowed'][start:start+128].numpy()
        counts+=(val['features'][start:start+128][allowed]!=0).sum(0)
        eligible+=allowed.sum()
    builder=DynamicFeatureBuilder(purchases,catalogue,companies,info['product_ids'])
    builder.tables['sku'].advance(TRAIN_CUTOFF)
    pairs=builder.tables['sku'].pairs
    observed=pairs[pairs>0]
    result={'identity':manifest['identity'],'feature_contract':contract(), 'heldout_access':False,
            'purchase_columns':list(purchases.columns),'training_events':int((purchases.purchase_date<TRAIN_CUTOFF).sum()),
            'validation_events':int((purchases.purchase_date>=TRAIN_CUTOFF).sum()),
            'train_rows':len(train['u']),'validation_rows':len(val['u']),'products':len(info['product_ids']),
            'eligible_validation_min_max':[int(val['allowed'].sum(1).min()),int(val['allowed'].sum(1).max())],
            'missing_purchase_metadata':[name for name in ('brand','product_subcategory','product_family','industry_branch','price','inventory','promotion') if name not in purchases],
            'policy_availability':availability,'new_feature_nonzero_eligible':{key:float(value/eligible) for key,value in zip(NAMES,counts)},
            'train_sku_transition_types':len(observed),'singleton_pair_fraction':float((observed==1).mean()),
            'metadata_assumption':'current snapshot for product/family/industry; no effective-date claim',
            'global_policy':'training strict earlier; validation frozen at training cutoff (including global sale age at aggregate refresh)',
            'unit_parsing':'not inferred; price/stock/promotion excluded; package SKUs remain distinct'}
    atomic_json(result,EVIDENCE_ROOT/'p1_inventory.json',overwrite=False)
    print('P1 feature availability and transition sparsity recorded.',flush=True)


if __name__=='__main__':main()

"""Versioned 64-event memory preserving the original 32-event suffix exactly."""

from pathlib import Path

import pandas as pd
import torch

from recommender.artifacts import file_digest, json_digest
from recommender.catalogue import load_product_catalogue
from recommender.development import load_development_purchases
from recommender.scope import get_datasets_locs
from recommender.sequences import build_customer_sequence, DAYS_PER_YEAR
from recommender.shared_data import DEFAULT_RETRIEVER, atomic_torch_save
from recommender.split import TEST_START

FIELDS=('sequence','attention_mask','seq_product_ids','seq_purchase_dates',
        'seq_category_ids','seq_subcategory_ids','seq_business_line_ids')
CACHE=Path('data/interim/separate_profile_v1')


def expand_history(slates, source_digest):
    identity=json_digest({'observed_source_digest':source_digest,'length':64,
                          'original_slates':{s:v['cache_identity'] for s,v in slates.items()},
                          'schema':'legacy_event9_quantity60_strict_prior_v1'})
    path=CACHE/'history64.pt'
    if path.exists():
        payload=torch.load(path,map_location='cpu',weights_only=False)
        if payload['identity']!=identity:raise ValueError('64-event history cache identity changed.')
        return {s:{**v,**payload['fields'][s]} for s,v in slates.items()},payload['diagnostics'],file_digest(path)
    purchases=load_development_purchases('data/interim/cleaned_purchases.csv')
    if (purchases.purchase_date>=TEST_START).any():raise ValueError('Development-only history required.')
    catalogue=load_product_catalogue(get_datasets_locs('items'))
    metadata=catalogue.set_index('product_id').to_dict(orient='index')
    reference=torch.load(DEFAULT_RETRIEVER,map_location='cpu',weights_only=False)
    vocabs=reference['vocabs'];quantity=reference['config'].get('quantity_reference')
    if quantity!=60.:raise ValueError('Unexpected source history quantity reference.')
    histories={str(cid):v for cid,v in purchases.groupby('customer_id',sort=False)}
    empty=purchases.iloc[:0]
    output={};diagnostics={}
    for split,slate in slates.items():
        rows={name:[] for name in FIELDS}
        for i,(cid,day) in enumerate(zip(slate['customer_ids'],slate['prediction_days'].tolist())):
            date=pd.Timestamp('2025-01-01')+pd.Timedelta(days=int(day))
            if date>=TEST_START:raise ValueError('Held-out query forbidden.')
            result=build_customer_sequence(histories.get(str(cid),empty),str(cid),date,max_len=64,
                                           feature_dim=9,vocabularies=vocabs,item_metadata=metadata,
                                           enriched=True,quantity_reference=quantity)
            count=int(result['attention_mask'].sum());start=max(0,count-32)
            for name in FIELDS:
                suffix=result[name][start:start+32]
                if not torch.equal(suffix,slate[name][i]):
                    difference=(suffix.float()-slate[name][i].float()).abs().max().item()
                    raise AssertionError(f'History suffix mismatch: {split} row{i} {name} diff={difference}.')
                rows[name].append(result[name])
            if i%2000==0:print(f'64-event history {split}: {i}/{len(slate["customer_ids"])}; exact32 suffix verified.',flush=True)
        fields={k:torch.stack(v) for k,v in rows.items()}
        valid=fields['attention_mask'].bool()
        if ((fields['seq_purchase_dates']*DAYS_PER_YEAR).round()>=slate['prediction_days'][:,None])[valid].any():
            raise AssertionError('History is not strictly prior to query date.')
        output[split]=fields
        diagnostics[split]={'queries':len(valid),'queries_with_more_than32_events':int((valid.sum(1)>32).sum()),
                            'mean_events':float(valid.sum(1).float().mean()),'exact32_suffix_parity':True}
    CACHE.mkdir(parents=True,exist_ok=True)
    atomic_torch_save({'identity':identity,'fields':output,'diagnostics':diagnostics,
                       'heldout_access':False,'quantity_reference':quantity},path)
    return {s:{**v,**output[s]} for s,v in slates.items()},diagnostics,file_digest(path)

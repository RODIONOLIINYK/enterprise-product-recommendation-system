"""Resolve frozen references, align real transitions, and export only blind past views."""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from recommender.catalogue import load_product_catalogue,RollingCandidatePool
from recommender.company import load_companies
from recommender.data import select_paid_purchases,aggregate_purchases
from recommender.development import load_development_purchases
from recommender.scope import get_datasets_locs
from recommender.shared_data import validate_shared_cache
from recommender.shared_ranker import tensor_digest
from recommender.ranker import UPSCANRRanker
from recommender.train_ranker import RankerDataset,collate_ranker_batch,ranker_batch_inputs
from recommender.evaluate import build_global_popularity_ranking,build_personal_repeat_ranking
from recommender.split import TEST_START,TRAIN_CUTOFF
from recommender.experiments.candidate_token_enrichment.train import snapshot_preserved_hashes
from recommender.experiments.candidate_token_enrichment.model import load_enrichment_ranker_checkpoint
from recommender.experiments.dynamic_selector.model import select_candidates
from recommender.experiments.rolling_candidate_v2.data import (
    prepare_or_load_rolling_feature_bundle,prepare_or_load_rolling_selector_scores,
    prepare_or_load_rolling_slate_cache,make_rolling_slate_loader,rolling_ranker_batch_inputs,
)
from .protocol import ROOT,ARMS,PRODUCT_BASIC,PRODUCT_TEXT,COMPANY_FIELDS,alias,canonical,digest,file_sha,private_dir,read_json,save_json
from .history_tools import json_value
from .sampling import select_cases
from .agent import worker_prompt


def align_transition_rows(expected,actual):
    expected=[tuple(k) for k in expected];actual=[tuple(k) for k in actual]
    if len(set(expected))!=len(expected) or len(set(actual))!=len(actual) or set(expected)!=set(actual):
        raise ValueError('Transition IDs/dates must match exactly and uniquely.')
    positions={k:i for i,k in enumerate(actual)}
    return np.array([positions[k] for k in expected],dtype=np.int64)


def subset(slate,rows):
    n=len(slate['customer_ids']);rows=np.array(rows,dtype=np.int64,copy=True)
    return {k:v[rows] if isinstance(v,torch.Tensor) and v.ndim and len(v)==n
            else [v[int(i)] for i in rows] if isinstance(v,list) and len(v)==n else v for k,v in slate.items()}


def ranked(candidates,scores):
    return [candidates[int(i)] for i in np.argsort(-np.asarray(scores),kind='stable')]


def check_case(purchases,customer_id,date,targets,new_targets,candidate_sets,pool):
    date=pd.Timestamp(date)
    if pd.isna(date) or not TRAIN_CUTOFF<=date<TEST_START:raise ValueError('Only validation transitions are permitted.')
    history=purchases.loc[purchases.customer_id.eq(customer_id)&purchases.purchase_date.lt(date)]
    actual=set(purchases.loc[purchases.customer_id.eq(customer_id)&purchases.purchase_date.eq(date),'product_id'])
    if not actual or set(targets)!=actual or set(new_targets)!=actual-set(history.product_id):
        raise ValueError('Cached labels do not match the exact raw transition/date.')
    eligible=pool.eligible_on(date)
    for name,ids in candidate_sets.items():
        depth=50 if name=='historical50' else 100
        if len(ids)!=depth or len(set(ids))!=depth or not set(ids)<=eligible:
            raise ValueError('Slate has invalid size/duplicates/eligibility.')
    return history


def write_blind_session(path,*,case_id,date,benchmark,arm,replicate,company_id,candidate_ids,
                        events,catalogue,companies,presentation_seed=42):
    path=private_dir(path)
    as_of=pd.Timestamp(date)
    if pd.isna(as_of) or as_of>=TEST_START:raise ValueError('Development snapshot only.')
    past=events.loc[pd.to_datetime(events.purchase_date).lt(as_of)].copy()
    if arm=='customer_only':past=past.loc[past.customer_id.eq(company_id)]
    event_rows=[{'customer_id':str(r.customer_id),'purchase_date':str(pd.Timestamp(r.purchase_date).date()),
                 'product_id':str(r.product_id),'quantity':float(r.quantity)} for r in past.itertuples()]
    present_companies=set(past.customer_id)|{company_id}
    present_products=set(past.product_id)|set(candidate_ids)
    basic=(*PRODUCT_BASIC,*PRODUCT_TEXT) if arm!='reduced_semantic' else PRODUCT_BASIC
    cat={p:{k:json_value(catalogue[p].get(k)) for k in basic if k in catalogue[p]} for p in sorted(present_products) if p in catalogue}
    comp={c:{k:json_value(companies.get(c,{}).get(k)) for k in COMPANY_FIELDS} for c in sorted(present_companies)}
    data_path=path/'events.json.gz'
    with data_path.open('xb') as out:out.write(gzip.compress(canonical(event_rows).encode(),mtime=0))
    data_path.chmod(0o600)
    save_json(path/'catalogue.json',cat);save_json(path/'companies.json',comp)
    # Same presentation for all information ablations and stability repeats.
    rng=np.random.default_rng(int(digest({'case_id':case_id,'benchmark':benchmark,'seed':presentation_seed})[:16],16))
    shuffled=[candidate_ids[int(i)] for i in rng.permutation(len(candidate_ids))]
    session={'protocol_version':1,'case_id':case_id,'prediction_date':str(as_of.date()),'benchmark':benchmark,
             'arm':arm,'replicate':replicate,'company_id':company_id,'candidate_ids':shuffled,
             'metadata_policy':'snapshot_unverified','requested_provider':'antigravity','requested_model':'gemini-4-hydrogen',
             'requested_options':{'thinking_level':'high'},'temperature':None,'decoding_seed':None,
             'decoding_limit':'Connected runtime exposes thinking level, not temperature or decoding seed.',
             'physical_snapshot_events':len(event_rows),'files_sha256':{p.name:file_sha(p) for p in
                (data_path,path/'catalogue.json',path/'companies.json')}}
    save_json(path/'session.json',session)
    return session


@torch.no_grad()
def legacy_scores(slate,manifest,checkpoint):
    ckpt=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if ckpt['retriever_digest']!=manifest['stage1_digest'] or list(ckpt['product_ids'])!=manifest['product_ids']:
        raise ValueError('Historical ranker/catalogue/retriever compatibility failure.')
    model=UPSCANRRanker.from_retriever_checkpoint('checkpoints/retriever-discovery-reference/best_retriever.pt',
             statistics_attention=ckpt['statistics_attention'],excluded_statistics=tuple(ckpt['excluded_statistics']))
    model.load_state_dict(ckpt['model_state_dict'],strict=True);model.eval()
    result=[]
    loader=torch.utils.data.DataLoader(RankerDataset(slate),batch_size=64,collate_fn=collate_ranker_batch)
    for batch in loader:
        result.append(model(**ranker_batch_inputs(batch,manifest['product_vectors'],model,torch.device('cpu'))).logits.cpu())
    return torch.cat(result),{'checkpoint':checkpoint,'checkpoint_sha256':file_sha(checkpoint),'epoch':ckpt['epoch'],
                              'retriever_sha256':ckpt['retriever_digest'],'input_contract':'legacy_unverified_input_contract',
                              'statistics_policy':'historical frozen_train_cutoff_v1','slate_width':50}


def prepare(root=ROOT,*,size=50,seed=42):
    root=Path(root)
    if root.exists():raise FileExistsError('Use a fresh private output root; never overwrite oracle predictions.')
    private_dir(root);private_dir(root/'vault');private_dir(root/'sessions')
    before=snapshot_preserved_hashes()
    purchases=load_development_purchases('data/interim/cleaned_purchases.csv')
    catalogue=load_product_catalogue(get_datasets_locs('items'))
    companies=load_companies(get_datasets_locs('companies'),purchases=purchases,minimum_purchases=1)
    paid=aggregate_purchases(select_paid_purchases(purchases,products=catalogue))
    if len(paid)!=len(purchases) or not np.isclose(paid.quantity.sum(),purchases.quantity.sum()):
        raise ValueError('Existing cleaned paid-sale scope/merging differs; rebuild upstream coherently first.')
    for name in ('features_manifest.json','selector_scores_manifest.json','slates_manifest.json','val_selector_scores.pt','val_slates.pt'):
        if not (Path('data/interim/rolling_candidate_v2/v3')/name).is_file():
            raise FileNotFoundError('Required corrected rolling caches are absent; no reference cache generation is performed.')
    data,info,features_manifest,meta=prepare_or_load_rolling_feature_bundle(build=False)
    _,selector_manifest=prepare_or_load_rolling_selector_scores(data,info,meta)
    slates,slate_manifest=prepare_or_load_rolling_slate_cache(data,info,meta)
    current=slates['val'];source=data['val']['source']
    if len(source['customer_ids'])!=6191:raise ValueError('Unexpected validation population.')
    rebuilt=select_candidates(data['val']['rolling_selector_scores'],data['val']['allowed'],data['val']['unseen'],'quota100_50')
    if not torch.equal(rebuilt,current['candidate_indices']):raise ValueError('Current candidates are not corrected rolling selector slates.')
    legacy_path=Path('data/interim/ranker_val_slates_v3_repeat_0.5_rolling_v1.pt')
    old=torch.load(legacy_path,map_location='cpu',weights_only=False)
    shared=torch.load('data/interim/shared_stage1_v1/val.pt',map_location='cpu',weights_only=False)
    legacy_manifest=torch.load('data/interim/shared_stage1_v1/manifest.pt',map_location='cpu',weights_only=False)
    validate_shared_cache(shared,shared['cache_identity'],legacy_manifest['product_ids'],legacy_manifest['stage1_digest'])
    if tensor_digest(legacy_manifest['product_vectors'])!=legacy_manifest['product_vector_digest']:
        raise ValueError('Historical product-vector checksum failure.')
    if shared['candidate_pids']!=old['candidate_pids']:raise ValueError('Shared reconstruction is not the exact historical50 slate.')
    source_keys=[(str(c),int(day)) for c,day in source['transition_keys']]
    old_rows=align_transition_rows(source_keys,shared['transition_keys'])
    old={**old,'prediction_days':shared['prediction_days'],'new_positive_pids':shared['new_positive_pids']}
    pool=RollingCandidatePool(catalogue,purchases)
    groups={str(c):g.sort_values(['purchase_date','product_id'],kind='stable') for c,g in purchases.groupby('customer_id')}
    master=catalogue.set_index('product_id').to_dict('index')
    company_master=companies.set_index('customer_id').to_dict('index')
    population=[];truth={}
    for index,(cid,day) in enumerate(source_keys):
        date=pd.Timestamp('2025-01-01')+pd.Timedelta(days=day);old_row=int(old_rows[index])
        if old['customer_ids'][old_row]!=cid or set(old['positive_pids'][old_row])!=set(source['positive_pids'][index]):
            raise ValueError('Historical labels/customer alignment failure.')
        candidate_sets={'historical50':list(old['candidate_pids'][old_row]),'rolling100':list(current['candidate_pids'][index])}
        history=check_case(purchases,cid,date,source['positive_pids'][index],source['new_positive_pids'][index],candidate_sets,pool)
        new=set(source['new_positive_pids'][index]);old_new=new&set(candidate_sets['historical50']);cur_new=new&set(candidate_sets['rolling100'])
        case_class='repeat_only' if not new else 'new_absent100' if not cur_new else 'new_absent50_present100' if not old_new else 'new_present_both'
        case_id='V-'+digest({'customer':cid,'date':str(date.date()),'seed':seed})[:16]
        n=len(history);span=max((date-history.purchase_date.min()).days,1) if n else 1;rate=history.purchase_date.nunique()*30/span
        row={'case_id':case_id,'customer_id':cid,'month':str(date.to_period('M')),'prediction_date':str(date.date()),
             'source_row':index,'legacy_row':old_row,'case_class':case_class,'history_events':n,
             'history_bin':'short_1_8' if n<=8 else 'medium_9_31' if n<32 else 'long_32plus',
             'industry':str(json_value(company_master.get(cid,{}).get('industry_branch')) or 'unknown'),
             'frequency_bin':'low' if rate<.5 else 'medium' if rate<2 else 'high'}
        population.append(row)
        truth[case_id]={'case_id':case_id,'company_id':alias(cid),'prediction_date':str(date.date()),
                        'targets':sorted(set(source['positive_pids'][index])),'new_targets':sorted(new),
                        'candidates':candidate_sets,'history_events':n,'baseline_rankings':{}}
    if sum(len(t['targets']) for t in truth.values())!=10276 or sum(len(t['new_targets']) for t in truth.values())!=1857:
        raise ValueError('Complete validation target denominators mismatch.')
    selected,sampling=select_cases(pd.DataFrame(population),size=size,seed=seed)
    save_json(root/'vault'/'sampling.json',{'metadata':sampling,'selected_cases':selected.to_dict('records')})
    selected_truth={r.case_id:truth[r.case_id] for r in selected.itertuples()}
    rows=selected.source_row.to_numpy();legacy_selected=selected.legacy_row.to_numpy()
    baseline_provenance={}
    for tag,path in [('upscanr_ndcg','checkpoints/ranker/best_ranker.pt'),('upscanr_discovery','checkpoints/ranker/best_discovery_ranker.pt')]:
        scores,provenance=legacy_scores(subset(old,legacy_selected),legacy_manifest,path)
        baseline_provenance[tag]=provenance
        for i,r in enumerate(selected.itertuples()):selected_truth[r.case_id]['baseline_rankings'].setdefault('historical50',{})[tag]=ranked(truth[r.case_id]['candidates']['historical50'],scores[i].numpy())
    # Current Stage2 weights explicitly bind this exact rolling100 slate.
    arm=read_json('experiments/candidate_token_enrichment_v1/summary.json')['arms']['arm_a_control_pointwise_425d']
    record=arm['ndcg']
    if file_sha(record['checkpoint'])!=record['checkpoint_sha256'] or file_sha(record['logits_pt'])!=record['logits_sha256']:
        raise ValueError('Current ranker/score checksum failure.')
    if arm['provenance']['val_slate_identity']!=slate_manifest['split_cache_identity']['val']:
        raise ValueError('Current ranker was trained/evaluated with another slate.')
    model,_=load_enrichment_ranker_checkpoint(record['checkpoint'],expected_provenance=arm['provenance'],expected_feature_contract_digest=meta['feature_schema_digest'])
    model.eval();sample_slate=subset(current,rows)
    inputs=rolling_ranker_batch_inputs(next(iter(make_rolling_slate_loader(sample_slate,batch_size=size))),data['train']['v'],torch.device('cpu'))
    saved=torch.load(record['logits_pt'],map_location='cpu',weights_only=False)['logits'][rows]
    with torch.no_grad():difference=float((model(**inputs).logits-saved).abs().max())
    if difference>1e-4:raise ValueError('Current checkpoint cannot reproduce its saved slate scores.')
    baseline_provenance['pointwise425_ndcg']={**arm['provenance'],'checkpoint_sha256':record['checkpoint_sha256'],'checkpoint':record['checkpoint'],
                                             'epoch':record['epoch'],'cpu_replay_max_abs_diff':difference}
    selector=data['val']['rolling_selector_scores'][rows].gather(1,current['candidate_indices'][rows])
    for i,r in enumerate(selected.itertuples()):
        entry=selected_truth[r.case_id];entry['baseline_rankings']['rolling100']={
            'pointwise425_ndcg':ranked(entry['candidates']['rolling100'],saved[i].numpy()),
            'selector301_rolling':ranked(entry['candidates']['rolling100'],selector[i].numpy())}
        for benchmark,candidates in entry['candidates'].items():
            hist=groups[r.customer_id].loc[groups[r.customer_id].purchase_date.lt(pd.Timestamp(r.prediction_date))]
            popularity=build_global_popularity_ranking(purchases,candidates,cutoff=r.prediction_date)
            repeat=build_personal_repeat_ranking(hist,r.prediction_date,popularity,max_k=10000)
            repeat=[p for p in repeat if p in set(candidates)]
            if set(repeat)!=set(candidates):raise ValueError('Personal-repeat heuristic must cover the exact slate.')
            entry['baseline_rankings'][benchmark].update(popularity_prior=popularity,personal_repeat_prior=repeat)
    # Original17-field recipe was reconstructed causally on rolling100, unlike the invalid old binary.
    try:
        from catboost import CatBoostClassifier,Pool
        from recommender.experiments.catboost_comparison.features import build_legacy_numeric,categorical_frame,legacy_frame,LEGACY_FEATURES,CATEGORICAL
        tree_summary=read_json('experiments/catboost_current_v1/summary.json');tree=tree_summary['models']['historical17']
        if tree['provenance']['slate_cache_identities']!=slate_manifest['split_cache_identity'] or tree['provenance']['observed_source_digest']!=meta['observed_source_digest']:
            raise ValueError('Original-recipe reconstruction provenance mismatch.')
        if file_sha(tree['model_path'])!=tree['model_sha256']:raise ValueError('Tree checkpoint mismatch.')
        cat=CatBoostClassifier();cat.load_model(tree['model_path'])
        if tuple(cat.feature_names_)!=LEGACY_FEATURES:raise ValueError('Tree preprocessing feature names changed.')
        indices=current['candidate_indices'][rows].numpy();dates=pd.to_datetime(selected.prediction_date)
        values=build_legacy_numeric(purchases,catalogue,info['product_ids'],selected.customer_id.tolist(),dates,indices)
        frame=legacy_frame(values,categorical_frame(info['product_ids'],catalogue,indices))
        scores=cat.predict(Pool(frame,cat_features=list(CATEGORICAL)),prediction_type='RawFormulaVal',thread_count=2).reshape(size,100)
        original=torch.load(tree['logits_pt'],map_location='cpu',weights_only=False)['logits'][rows].numpy()
        if not np.allclose(scores,original,rtol=0,atol=1e-5):raise ValueError('Original-recipe preprocessing replay failure.')
        for i,r in enumerate(selected.itertuples()):selected_truth[r.case_id]['baseline_rankings']['rolling100']['catboost_original17_causal_retrain']=ranked(truth[r.case_id]['candidates']['rolling100'],scores[i])
        baseline_provenance['catboost_original17_causal_retrain']={'status':'available','feature_names':list(LEGACY_FEATURES),
            'checkpoint_sha256':tree['model_sha256'],'checkpoint':tree['model_path'],'source_commit':tree_summary['historical']['source_commit_full'],
            'selection':tree['selection'],'provenance':tree['provenance'],
            'historical_binary_status':'Not evaluated: old binary training extends beyond current validation/test boundary.'}
    except (FileNotFoundError,ImportError) as error:
        baseline_provenance['catboost_original17_causal_retrain']={'status':'unavailable','reason':str(error)}
    # All agent-visible companies are pseudonyms and fields exclude names/contact/counts.
    anonymized=purchases[['customer_id','purchase_date','product_id','quantity']].copy()
    anonymized['customer_id']=anonymized.customer_id.map(alias)
    catalog_dict={str(p):{k:json_value(v) for k,v in meta_row.items() if k in (*PRODUCT_BASIC,*PRODUCT_TEXT)} for p,meta_row in master.items()}
    for pid in catalog_dict:catalog_dict[pid]['product_id']=pid
    company_dict={alias(str(cid)):{k:json_value(row.get(k)) for k in COMPANY_FIELDS} for cid,row in company_master.items()}
    jobs=[];repeat_ids=set(selected.case_id.iloc[:5])
    for r in selected.itertuples():
        entry=selected_truth[r.case_id]
        for benchmark,candidates in entry['candidates'].items():
            for arm in ARMS:
                repeats=(0,1) if arm=='global_history' and benchmark=='rolling100' and r.case_id in repeat_ids else (0,)
                for replicate in repeats:
                    job_id=r.case_id+'-'+benchmark+'-'+arm+f'-r{replicate}'
                    session_path=root/'sessions'/job_id
                    write_blind_session(session_path,case_id=r.case_id,date=r.prediction_date,benchmark=benchmark,arm=arm,
                        replicate=replicate,company_id=entry['company_id'],candidate_ids=candidates,events=anonymized,
                        catalogue=catalog_dict,companies=company_dict,presentation_seed=seed)
                    prompt=worker_prompt(str(session_path.resolve()),str(Path.cwd()))
                    (session_path/'worker_prompt.txt').write_text(prompt);(session_path/'worker_prompt.txt').chmod(0o600)
                    jobs.append({'job_id':job_id,'case_id':r.case_id,'benchmark':benchmark,'arm':arm,'replicate':replicate,
                                 'session_path':str(session_path.resolve()),'prompt_path':str((session_path/'worker_prompt.txt').resolve())})
    save_json(root/'vault'/'truth.json',selected_truth)
    save_json(root/'jobs.json',{'jobs':jobs,'requested_provider':'antigravity','requested_model':'gemini-4-hydrogen','requested_options':{'thinking_level':'high'}})
    if snapshot_preserved_hashes()!=before:raise AssertionError('Frozen reference/cache hashes changed.')
    manifest={'protocol_version':1,'training_executed':False,'heldout_access':False,'promoted':False,'seed':seed,
        'sample_size':size,'population_queries':6191,'population_targets':10276,'population_new_targets':1857,
        'historical_train_rows':legacy_manifest['train_rows'],'current_train_rows':int(features_manifest['train_rows']),
        'source_fingerprint':meta['observed_source_digest'],'metadata_policy':'snapshot_unverified',
        'historical_slate_sha256':file_sha(legacy_path),'historical_slate_identity':old['cache_identity'],
        'current_slate_manifest':slate_manifest,'selector_scores_manifest':selector_manifest,
        'historical_vector_manifest_sha256':file_sha('data/interim/shared_stage1_v1/manifest.pt'),
        'baselines':baseline_provenance,'sampling':sampling,'expected_sessions':len(jobs),
        'jobs_sha256':file_sha(root/'jobs.json'),'truth_sha256':file_sha(root/'vault'/'truth.json'),
        'preserved_hashes':before,'status':'prepared','private_artifacts_git_ignored':True,
        'isolation':'Fresh model context and physically filtered blind bundles; shared filesystem is not an OS security sandbox. Runtime access audits are required.',
        'oracle_limit':'LLM expert performance is not a Bayes-optimal or mathematical predictability ceiling.'}
    save_json(root/'manifest.json',manifest)
    print(json.dumps({'prepared_cases':size,'prediction_sessions':len(jobs),'root':str(root),'heldout_access':False}))
    return manifest


if __name__=='__main__':
    torch.set_num_threads(2)
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',default=str(ROOT));parser.add_argument('--size',type=int,default=50);parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args();prepare(args.root,size=args.size,seed=args.seed)

"""Complete-denominator matched evaluation, candidate ceilings and a global reveal barrier."""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from recommender.evaluate import compute_single_transition_metrics
from .protocol import ROOT,read_json,save_json,file_sha


def case_metrics(ranking,targets,new_targets,candidates,k=10):
    ranking=list(ranking);targets=set(targets);new=set(new_targets);candidates=set(candidates)
    if not targets or not new<=targets or len(set(ranking))!=len(ranking) or not set(ranking)<=candidates:
        raise ValueError('Invalid complete-target/ranking contract.')
    base=compute_single_transition_metrics(ranking,targets,k_values=(k,))
    hits=set(ranking[:k])&targets;repeat=targets-new;present=targets&candidates;new_present=new&candidates
    discount=1/np.log2(np.arange(k)+2)
    ideal=float(discount[:min(k,len(targets))].sum())
    return {'recall@10':base[f'recall@{k}'],'ndcg@10':base['ndcg@10'],'targets':len(targets),'new_targets':len(new),
        'repeat_targets':len(repeat),'new_hits@10':len(hits&new),'repeat_hits@10':len(hits&repeat),
        'targets_in_slate':len(present),'new_targets_in_slate':len(new_present),
        'repeat_targets_in_slate':len(repeat&candidates),'coverage_macro':len(present)/len(targets),
        'oracle_recall@10':min(k,len(present))/len(targets),
        'oracle_ndcg@10':float(discount[:min(k,len(present))].sum())/ideal,
        'oracle_new_hits@10':min(k,len(new_present)), 'oracle_repeat_hits@10':min(k,len(repeat&candidates))}


def aggregate(rows):
    if not rows:return None
    def total(name):return sum(row[name] for row in rows)
    def ratio(n,d):return total(n)/total(d) if total(d) else None
    return {'queries':len(rows),'recall@10':float(np.mean([r['recall@10'] for r in rows])),
        'ndcg@10':float(np.mean([r['ndcg@10'] for r in rows])),
        'new_recall@10':ratio('new_hits@10','new_targets'),'repeat_recall@10':ratio('repeat_hits@10','repeat_targets'),
        'new_hits@10':total('new_hits@10'),'new_targets':total('new_targets'),'targets':total('targets'),
        'candidate_coverage_macro':float(np.mean([r['coverage_macro'] for r in rows])),
        'candidate_coverage_micro':ratio('targets_in_slate','targets'),'new_slate_coverage':ratio('new_targets_in_slate','new_targets'),
        'conditional_new_recall@10':ratio('new_hits@10','new_targets_in_slate'),
        'candidate_constrained_recall_ceiling':float(np.mean([r['oracle_recall@10'] for r in rows])),
        'candidate_constrained_ndcg_ceiling':float(np.mean([r['oracle_ndcg@10'] for r in rows])),
        'candidate_constrained_new_recall_ceiling':ratio('oracle_new_hits@10','new_targets'),
        'candidate_constrained_repeat_recall_ceiling':ratio('oracle_repeat_hits@10','repeat_targets'),
        'candidate_constrained_conditional_new_recall_ceiling':ratio('oracle_new_hits@10','new_targets_in_slate'),
        'ceiling_note':'New/repeat ceilings independently prioritize their respective targets; they need not be simultaneously attainable.'}


def paired_bootstrap(a,b,customers,*,seed=42,draws=2000):
    if len(a)!=len(b) or len(a)!=len(customers):raise ValueError('Matched case/customer alignment required.')
    groups=defaultdict(list)
    for i,cid in enumerate(customers):groups[cid].append(i)
    unique=sorted(groups);rng=np.random.default_rng(seed)
    names=('recall@10','ndcg@10','new_recall@10','repeat_recall@10')
    aa,bb=aggregate(a),aggregate(b);values={k:[] for k in names}
    for _ in range(draws):
        ids=[i for cid in rng.choice(unique,len(unique),replace=True) for i in groups[cid]]
        x,y=aggregate([a[i] for i in ids]),aggregate([b[i] for i in ids])
        for name in names:
            if x[name] is not None and y[name] is not None:values[name].append(y[name]-x[name])
    return {name:{'delta':bb[name]-aa[name] if aa[name] is not None and bb[name] is not None else None,
                  'ci95':np.quantile(values[name],[.025,.975]).tolist() if values[name] else None,
                  'absolute_percentage_points':100*(bb[name]-aa[name]) if name!='ndcg@10' and aa[name] is not None and bb[name] is not None else None,
                  'relative_change':(bb[name]-aa[name])/aa[name] if aa[name] else None}
            for name in names}


def validate_seals(root):
    root=Path(root);manifest=read_json(root/'manifest.json')
    if file_sha(root/'jobs.json')!=manifest['jobs_sha256']:raise ValueError('Job manifest changed.')
    jobs=read_json(root/'jobs.json')['jobs'];missing=[];records=[]
    for job in jobs:
        path=Path(job['session_path'])
        if not (path/'seal.json').exists():missing.append(job['job_id']);continue
        seal=read_json(path/'seal.json')
        if (seal['prediction_sha256']!=file_sha(path/'prediction.json') or seal['session_sha256']!=file_sha(path/'session.json')
                or seal['tool_log_sha256']!=file_sha(path/'tool_log.jsonl') or not seal['explanations_frozen_before_reveal']):
            raise ValueError('Prediction/context/explanation log changed after sealing.')
        records.append((job,seal,read_json(path/'prediction.json')))
    if missing:raise ValueError(f'Reveal blocked: {len(missing)} unsealed oracle sessions remain.')
    if file_sha(root/'vault'/'truth.json')!=manifest['truth_sha256']:raise ValueError('Evaluation targets changed.')
    return manifest,records


def baseline_report(root):
    """Ground-truth evaluator only; never a prediction-agent tool or context."""
    root=Path(root);truth=read_json(root/'vault'/'truth.json');per_case=[];grouped=defaultdict(list)
    for case_id,case in truth.items():
        for benchmark,candidates in case['candidates'].items():
            for name,ranking in case['baseline_rankings'][benchmark].items():
                metrics=case_metrics(ranking,case['targets'],case['new_targets'],candidates)
                per_case.append({'case_id':case_id,'company_id':case['company_id'],'benchmark':benchmark,'model':name,**metrics})
                grouped[(benchmark,name)].append(metrics)
    return {'aggregate':{benchmark:{name:aggregate(rows) for (b,name),rows in grouped.items() if b==benchmark}
                         for benchmark in ('historical50','rolling100')},'per_case':per_case}


def evaluate(root=ROOT):
    root=Path(root);manifest,seals=validate_seals(root)
    # Runtime audit must be supplied before we treat hosted prediction sessions as clean.
    runtime=read_json(root/'runtime_receipts.json')
    if runtime.get('status')!='complete' or runtime.get('access_audit_status')!='passed':
        raise ValueError('Runtime completion and clean access audit are required before reveal.')
    if not (root/'reveal.json').exists():save_json(root/'reveal.json',{'all_explanations_frozen':True,'sealed_sessions':len(seals)})
    truth=read_json(root/'vault'/'truth.json');base=baseline_report(root)
    rows=base['per_case'][:];by_group=defaultdict(dict);repeat_predictions={};errors=[];llm_only=[]
    for job,seal,prediction in seals:
        case=truth[job['case_id']];benchmark=job['benchmark'];ranking=[r['product_id'] for r in prediction['ranked_products']]
        if job['replicate']:
            repeat_predictions[(job['case_id'],benchmark,job['arm'])]=ranking;continue
        metrics=case_metrics(ranking,case['targets'],case['new_targets'],case['candidates'][benchmark])
        record={'case_id':job['case_id'],'company_id':case['company_id'],'benchmark':benchmark,'model':'llm_'+job['arm'],**metrics}
        rows.append(record);by_group[(benchmark,record['model'])][job['case_id']]=record
        primary='upscanr_ndcg' if benchmark=='historical50' else 'pointwise425_ndcg'
        neural=set(case['baseline_rankings'][benchmark][primary][:10]);hits=set(ranking)&set(case['targets'])
        for pid in case['targets']:
            present=pid in case['candidates'][benchmark];hit=pid in ranking
            errors.append({'case_id':job['case_id'],'benchmark':benchmark,'arm':job['arm'],'product_id':pid,
                           'is_new':pid in case['new_targets'],'candidate_generation_failure':not present,
                           'ranking_failure':present and not hit,'llm_hit':hit,'neural_hit':pid in neural,
                           'semantic_inference_failure':'unassessed; requires relationship evidence review',
                           'weak_historical_evidence':'unassessed','intrinsic_unpredictability':'not established by failure'})
        for row in prediction['ranked_products']:
            if row['product_id'] in hits-neural:
                llm_only.append({'case_id':job['case_id'],'benchmark':benchmark,'arm':job['arm'],'is_new':row['product_id'] in case['new_targets'],
                                 'prediction':row,'support_status':'Pre-reveal evidence cited; factual/causal interpretation requires review.',
                                 'augmentation_rule':'Revalidate on training-only data; never distill validation labels into training.'})
    for row in base['per_case']:by_group[(row['benchmark'],row['model'])][row['case_id']]=row
    aggregate_metrics={};comparisons={}
    for benchmark in ('historical50','rolling100'):
        aggregate_metrics[benchmark]={name:aggregate(list(records.values())) for (b,name),records in by_group.items() if b==benchmark}
        names=sorted(aggregate_metrics[benchmark]);case_ids=sorted(truth)
        primary='upscanr_ndcg' if benchmark=='historical50' else 'pointwise425_ndcg'
        pairs=[(primary,'llm_'+arm) for arm in ('customer_only','global_history','reduced_semantic')]
        pairs += [('llm_customer_only','llm_global_history'),('llm_reduced_semantic','llm_global_history')]
        comparisons[benchmark]={}
        for a,b in pairs:
            first=[by_group[(benchmark,a)][cid] for cid in case_ids];second=[by_group[(benchmark,b)][cid] for cid in case_ids]
            comparisons[benchmark][a+'__to__'+b]=paired_bootstrap(first,second,[truth[c]['company_id'] for c in case_ids])
    stability=[]
    primary_lookup={(job['case_id'],job['benchmark'],job['arm']):prediction for job,_,prediction in seals if not job['replicate']}
    for key,ranking in repeat_predictions.items():
        first=[r['product_id'] for r in primary_lookup[key]['ranked_products']]
        stability.append({'case_id':key[0],'benchmark':key[1],'arm':key[2],'top10_overlap':len(set(first)&set(ranking))/10,
                          'identical_order':first==ranking})
    out={'heldout_access':False,'training_executed':False,'promoted':False,'sample_type':'unweighted stratified exploratory sample, not full-population estimate',
         'sample_size':manifest['sample_size'],'aggregate':aggregate_metrics,'paired_customer_bootstrap95':comparisons,
         'stability':stability,'per_case':rows,'error_decomposition':errors,'llm_only_successes':llm_only,
         'resource_usage':{'sessions':len(seals),'tool_queries':sum(s['query_calls'] for _,s,_ in seals),
                           'tool_response_characters':sum(s['query_response_characters'] for _,s,_ in seals),
                           'actual_llm_tokens':None,'token_accounting_status':'not exposed by connected runtime'}}
    save_json(root/'evaluation.json',out)
    print(f'Evaluated {len(seals)} sealed sessions on {manifest["sample_size"]} matched cases; no held-out access.')
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default=str(ROOT));args=p.parse_args();evaluate(args.root)

"""Temporal, blinding, alignment, denominator and pre-reveal protocol regressions."""

import gzip
import json

import pandas as pd
import pytest

from recommender.development import load_development_purchases
from recommender.experiments.semantic_oracle.history_tools import HistoryTools
from recommender.experiments.semantic_oracle.protocol import save_json,file_sha,no_gold_fields,validate_prediction
from recommender.experiments.semantic_oracle.data import align_transition_rows,write_blind_session,check_case
from recommender.experiments.semantic_oracle.gateway import SessionGateway
from recommender.experiments.semantic_oracle.sampling import select_cases
from recommender.experiments.semantic_oracle.evaluate import case_metrics,aggregate,paired_bootstrap,validate_seals


def fixtures():
    rows=[('C0','2025-01-01','P00',1),('C0','2025-02-01','P00',2),
          ('C1','2025-01-02','P00',1),('C1','2025-02-02','P01',1)]
    rows += [('C2','2025-01-01',f'P{i:02d}',1) for i in range(20)]
    rows += [('C0','2025-10-05','SAME_DAY_FOCAL',1),('C1','2025-10-05','SAME_DAY_OTHER',1),
             ('C0','2025-11-01','FUTURE_FOCAL',1),('C3','2025-11-01','FUTURE_OTHER',1)]
    events=pd.DataFrame(rows,columns=['customer_id','purchase_date','product_id','quantity'])
    events['purchase_date']=pd.to_datetime(events.purchase_date)
    cat={f'P{i:02d}':{'product_id':f'P{i:02d}','product_name':f'Description{i}','product_category':'CAT',
                      'brand':'BRAND','product_family':'FAM','container_type':'BOX','unit':'unit','future_sales':999} for i in range(20)}
    companies={c:{'industry_branch':'IND','description':'Company description','purchase_count':999} for c in ('C0','C1','C2','C3')}
    return events,cat,companies


@pytest.mark.parametrize('date',['2025-03-01','2025-10-05','2025-02-01'])
def test_all_history_tools_are_past_only_and_order_independent(date):
    events,cat,companies=fixtures()
    tools=HistoryTools(events,cat,companies,as_of=date,customer_id='C0')
    prefix=events.loc[events.purchase_date.lt(pd.Timestamp(date))]
    truncated=HistoryTools(prefix,cat,companies,as_of=date,customer_id='C0')
    requests=[('company_history',{}),('product_history',{'product_id':'P00'}),('similar_companies',{}),
              ('relations',{'anchor_ids':['P00']}),('relations',{'anchor_ids':['P00'],'relation':'same_basket'}),
              ('trajectories',{'anchor_ids':['P00'],'destination_id':'P01'}),('intervals',{}),('market_trends',{})]
    for name,args in requests:
        assert getattr(tools,name)(**args)==getattr(truncated,name)(**args)
        text=json.dumps(getattr(tools,name)(**args))
        assert 'FUTURE_' not in text and 'SAME_DAY_' not in text


def test_global_search_has_no_company_or_32_event_cutoff():
    rows=[('FOCAL','2025-01-01',f'P{i}',1) for i in range(64)]
    rows += [(f'C{i:03d}','2025-01-01','P0',1) for i in range(90)]
    events=pd.DataFrame(rows,columns=['customer_id','purchase_date','product_id','quantity'])
    tools=HistoryTools(events,{}, {},as_of='2025-02-01',customer_id='FOCAL')
    assert tools.context([])['customer_history_events']==64
    assert tools.company_history(limit=10)['total_rows']==64
    assert tools.similar_companies(limit=10)['total_rows']==90
    assert len(tools.similar_companies(offset=80,limit=20)['rows'])==10
    assert tools.product_history('P0',offset=80,limit=20)['total_rows']==91


def test_heldout_reader_date_gates_before_dataframe_materialization(tmp_path,monkeypatch):
    path=tmp_path/'events.csv'
    path.write_text('customer_id,product_id,purchase_date,quantity\nA,P,2025-12-01,1\nSECRET_HELDOUT,SECRET_SKU,2026-02-19,1\n')
    original=pd.read_csv
    def checked(source,*args,**kwargs):
        text=source.getvalue();assert 'SECRET_HELDOUT' not in text and 'SECRET_SKU' not in text
        return original(source,*args,**kwargs)
    monkeypatch.setattr(pd,'read_csv',checked)
    result=load_development_purchases(path)
    assert len(result)==1
    events,cat,companies=fixtures();events.loc[len(events)]=['C0',pd.Timestamp('2026-02-19'),'FORBIDDEN',1]
    with pytest.raises(ValueError,match='Held-out'):HistoryTools(events,cat,companies,as_of='2025-10-05',customer_id='C0')


@pytest.mark.parametrize('arm',['customer_only','global_history','reduced_semantic'])
def test_physical_blind_bundles_and_scope_enforcement(tmp_path,arm):
    events,cat,companies=fixtures();session=tmp_path/'sessions'/'case'
    source=tmp_path/'frozen.pt';source.write_bytes(b'preserved checkpoint');before=file_sha(source)
    write_blind_session(session,case_id='case',date='2025-10-05',benchmark='synthetic20',arm=arm,replicate=0,
                        company_id='C0',candidate_ids=list(cat),events=events,catalogue=cat,companies=companies)
    gateway=SessionGateway(session);context=gateway.context();no_gold_fields(context)
    assert context['customer_history_events']==2
    raw=json.loads(gzip.decompress((session/'events.json.gz').read_bytes()))
    assert all(row['purchase_date']<'2025-10-05' for row in raw)
    assert context['candidate_ids']!=list(cat)
    assert 'purchase_count' not in json.dumps(context)
    if arm=='customer_only':
        assert {r['customer_id'] for r in raw}=={'C0'}
        with pytest.raises(PermissionError):gateway.query('similar_companies',{})
        with pytest.raises(PermissionError):gateway.query('company_history',{'company_id':'C1'})
    elif arm=='reduced_semantic':assert 'Description' not in (session/'catalogue.json').read_text()
    assert file_sha(source)==before
    with pytest.raises(TypeError):gateway.query('company_history',{'as_of':'2026-01-01'})
    with pytest.raises(ValueError):no_gold_fields({'labels':[1]})


def test_candidate_identity_and_date_alignment_are_exact():
    expected=[('A',1),('B',2)];actual=[('B',2),('A',1)]
    assert align_transition_rows(expected,actual).tolist()==[1,0]
    with pytest.raises(ValueError):align_transition_rows(expected,[('A',2),('B',1)])
    with pytest.raises(ValueError):align_transition_rows(expected,[('A',1),('A',1)])
    events,_,_=fixtures();events=events.assign(customer_id=events.customer_id.astype(str))
    class Pool:
        def eligible_on(self,date):return {f'P{i:03d}' for i in range(100)}
    targets={'SAME_DAY_FOCAL'}
    ids=[f'P{i:03d}' for i in range(100)]
    sets={'historical50':ids[:50],'rolling100':ids}
    check_case(events,'C0','2025-10-05',targets,targets,sets,Pool())
    with pytest.raises(ValueError,match='duplicates'):check_case(events,'C0','2025-10-05',targets,targets,{'historical50':[ids[0]]*50},Pool())
    with pytest.raises(ValueError,match='raw transition'):check_case(events,'C0','2025-10-05',{'WRONG'},targets,sets,Pool())


def test_seal_freezes_evidence_and_rejects_invalid_predictions(tmp_path):
    events,cat,companies=fixtures();path=tmp_path/'sessions'/'case'
    session=write_blind_session(path,case_id='case',date='2025-10-05',benchmark='synthetic20',arm='global_history',replicate=0,
             company_id='C0',candidate_ids=list(cat),events=events,catalogue=cat,companies=companies)
    gateway=SessionGateway(path);ctx=gateway.context()
    prediction={'case_id':'case','prediction_date':'2025-10-05','benchmark':'synthetic20','candidate_pool_size':20,
                'ranked_products':[{'rank':i+1,'product_id':f'P{i:02d}','relationship_types':['weak_evidence_speculation'],
                    'confidence':.1,'evidence_basis':'weak','evidence_ids':[],'evidence':['No strong evidence.'],
                    'explanation':'Weak uncalibrated example.'} for i in range(10)]}
    with pytest.raises(ValueError,match='Global arms'):gateway.submit(prediction)
    gateway.query('similar_companies',{});gateway.query('relations',{'anchor_ids':['P00']});gateway.query('market_trends',{})
    bad=json.loads(json.dumps(prediction));bad['ranked_products'][0]['evidence_basis']='empirical';bad['ranked_products'][0]['evidence_ids']=['FAKE']
    with pytest.raises(ValueError,match='not observed'):gateway.submit(bad)
    result=gateway.submit(prediction)
    assert result['sealed_prediction_sha256']==file_sha(path/'prediction.json')
    with pytest.raises(PermissionError,match='sealed'):gateway.context()
    bad=json.loads(json.dumps(prediction));bad['ranked_products'][1]['product_id']='P00'
    with pytest.raises(ValueError,match='Duplicate'):validate_prediction(bad,session)
    bad=json.loads(json.dumps(prediction));bad['ranked_products'][1]['relationship_types']=['repeat_replenishment']
    with pytest.raises(ValueError,match='never-purchased'):validate_prediction(bad,session,prior_products={'P00'})
    assert ctx['case_id']=='case'


def test_complete_denominators_and_independent_candidate_ceilings():
    metrics=case_metrics(['A','B'],{'A','B','C','OUT'},{'B','C','OUT'},{'A','B','C'})
    assert metrics['recall@10']==.5
    summary=aggregate([metrics])
    assert summary['new_recall@10']==1/3
    assert summary['conditional_new_recall@10']==1/2
    assert summary['candidate_coverage_micro']==3/4
    assert summary['candidate_constrained_recall_ceiling']==3/4
    assert summary['candidate_constrained_new_recall_ceiling']==2/3
    assert metrics['oracle_ndcg@10']<1
    long=case_metrics([str(i) for i in range(10)],set(map(str,range(12))),set(map(str,range(12))),set(map(str,range(11))))
    assert long['oracle_recall@10']==10/12
    first=paired_bootstrap([metrics,long],[metrics,long],['A','B'],draws=20)
    assert first['recall@10']['delta']==0 and first['recall@10']['ci95']==[0,0]
    assert first==paired_bootstrap([metrics,long],[metrics,long],['A','B'],draws=20)


def test_sampling_reproducible_diverse_and_reports_oversampling():
    frame=pd.DataFrame([{'case_id':f'v{i}','customer_id':f'c{i%60}','month':f'2025-{10+i%3:02d}',
        'case_class':'repeat_only' if i%2 else 'new_present_both','history_bin':'short' if i%3 else 'long',
        'industry':f'I{i%7}','frequency_bin':str(i%3)} for i in range(120)])
    a,meta=select_cases(frame,size=50);b,again=select_cases(frame,size=50)
    assert a.equals(b) and meta==again and a.customer_id.nunique()==50
    assert meta['population_size']==120 and 'not population estimates' in meta['oversampling']


def test_reveal_blocked_until_every_prediction_is_sealed(tmp_path):
    save_json(tmp_path/'jobs.json',{'jobs':[{'job_id':'missing','session_path':str(tmp_path/'missing')}]})
    save_json(tmp_path/'manifest.json',{'jobs_sha256':file_sha(tmp_path/'jobs.json')})
    with pytest.raises(ValueError,match='unsealed'):validate_seals(tmp_path)


def test_compact_gateway_preserves_blinding_evidence_and_denominators(tmp_path):
    from recommender.experiments.semantic_oracle.gateway import format_context_compact,format_query_compact
    events,cat,companies=fixtures()
    path=tmp_path/'sessions'/'case_rs'
    write_blind_session(path,case_id='case_rs',date='2025-10-05',benchmark='synthetic20',arm='reduced_semantic',replicate=0,
                        company_id='C0',candidate_ids=list(cat),events=events,catalogue=cat,companies=companies)
    gw=SessionGateway(path)
    ctx=gw.context()
    text,_=format_context_compact(ctx)
    assert ctx['customer_history_evidence_id'] in text
    assert 'Description' not in text
    rel_all=gw.query('relations',{'anchor_ids':['P00'],'relation':'next_basket'})
    rel_cand=gw.query('relations',{'anchor_ids':['P00'],'relation':'next_basket','candidate_only':True})
    assert rel_all['source_baskets']==rel_cand['source_baskets']
    assert rel_all['all_observed_pairs']==rel_cand['all_observed_pairs']
    assert rel_cand['filtered_to_candidates'] is True
    qtext=format_query_compact('relations',rel_cand)
    assert 'candidate_only=True' in qtext
    for row in rel_cand['rows']:
        assert row['evidence_id'] in qtext


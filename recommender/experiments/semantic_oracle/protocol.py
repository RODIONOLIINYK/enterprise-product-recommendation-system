"""Private artifacts, immutable session contracts and pre-reveal prediction validation."""

import hashlib
import json
import math
from pathlib import Path

ROOT = Path('data/interim/semantic_oracle_v1')
ARMS = ('customer_only', 'global_history', 'reduced_semantic')
RELATIONSHIPS = ('repeat_replenishment','assortment_completion','semantic_complementarity',
                 'cross_company_transition','substitution_variant','seasonality_market_trend','weak_evidence_speculation')
PRODUCT_TEXT = ('short_name','ukrainian_name','product_name','base_name')
PRODUCT_BASIC = ('product_id','brand','product_category','product_subcategory','business_line','product_family','container_type','unit')
COMPANY_FIELDS = ('industry_branch','description','city','region_code','sales_branch')


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',',':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def private_dir(path):
    path=Path(path);path.mkdir(parents=True,exist_ok=True);path.chmod(0o700);return path


def save_json(path,value,*,exclusive=True):
    path=Path(path);private_dir(path.parent)
    with path.open('x' if exclusive else 'w',encoding='utf-8') as out:
        out.write(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    path.chmod(0o600)


def read_json(path):
    return json.loads(Path(path).read_text())


def alias(value, *, kind='company'):
    return kind[:1].upper()+'-'+hashlib.sha256(f'semantic-oracle-v1:{kind}:{value}'.encode()).hexdigest()[:16]


def no_gold_fields(value):
    """Keys, not string substrings: 'target company' is legitimate; purchase labels are not."""
    forbidden={'positive_pids','new_positive_pids','labels','target_products','true_basket','baseline_scores',
               'baseline_rank','sampling_stratum','target_count','new_target_count','gold','ground_truth'}
    if isinstance(value,dict):
        if forbidden.intersection(value):raise ValueError('Prediction context contains target/baseline fields.')
        for v in value.values():no_gold_fields(v)
    elif isinstance(value,list):
        for v in value:no_gold_fields(v)


def validate_prediction(prediction,session,*,evidence_ids=(),prior_products=()):
    if prediction.get('case_id')!=session['case_id'] or prediction.get('prediction_date')!=session['prediction_date']:
        raise ValueError('Prediction case/date mismatch.')
    if prediction.get('benchmark')!=session['benchmark']:
        raise ValueError('Prediction benchmark mismatch.')
    candidates=session['candidate_ids']
    if len(set(candidates))!=len(candidates) or len(candidates)<10:
        raise ValueError('Invalid candidate pool.')
    if prediction.get('candidate_pool_size')!=len(candidates):raise ValueError('Candidate count mismatch.')
    rows=prediction.get('ranked_products',[])
    if len(rows)!=10 or [r.get('rank') for r in rows]!=list(range(1,11)):
        raise ValueError('Exactly ten products ranked 1..10 required.')
    ids=[r.get('product_id') for r in rows]
    if len(set(ids))!=10 or not set(ids)<=set(candidates):raise ValueError('Duplicate/ineligible recommendation.')
    for row in rows:
        confidence=row.get('confidence')
        if isinstance(confidence,bool) or not isinstance(confidence,(float,int)) or not math.isfinite(confidence) or not 0<=confidence<=1:
            raise ValueError('Subjective confidence must be finite in [0,1].')
        labels=row.get('relationship_types',[])
        if not labels or len(set(labels))!=len(labels) or not set(labels)<=set(RELATIONSHIPS):
            raise ValueError('Unknown/missing relationship classification.')
        if not isinstance(row.get('explanation'),str) or not row['explanation'].strip():raise ValueError('Explanation required.')
        evidence=row.get('evidence',[])
        if not evidence or not all(isinstance(e,str) and e.strip() for e in evidence):raise ValueError('Evidence statements required.')
        refs=row.get('evidence_ids',[])
        if not isinstance(refs,list) or not set(refs)<=set(evidence_ids):raise ValueError('Evidence ID was not observed before prediction.')
        basis=row.get('evidence_basis')
        if basis not in ('empirical','semantic','mixed','weak'):raise ValueError('Evidence basis required.')
        if basis in ('empirical','mixed') and not refs:raise ValueError('Empirical claims require historical evidence references.')
        if 'repeat_replenishment' in labels and row['product_id'] not in prior_products:
            raise ValueError('A never-purchased product cannot be labelled repeat/replenishment.')
        if session['arm']=='customer_only' and 'cross_company_transition' in labels:
            raise ValueError('Customer-only arm has no cross-company empirical evidence.')
    return prediction


def file_sha(path):
    with Path(path).open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()

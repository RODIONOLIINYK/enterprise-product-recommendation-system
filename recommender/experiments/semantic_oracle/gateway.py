"""The only permitted data interface for blind connected-runtime prediction agents."""

import argparse
import gzip
import json
from pathlib import Path
import sys

if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]))

from recommender.experiments.semantic_oracle.history_tools import HistoryTools
from recommender.experiments.semantic_oracle.protocol import (
    canonical,digest,file_sha,no_gold_fields,read_json,save_json,validate_prediction,
)

GLOBAL_TOOLS={'similar_companies','relations','trajectories','product_history','market_trends'}
TOOLS=GLOBAL_TOOLS|{'company_history','catalogue','intervals'}


def evidence_ids(value):
    found=set()
    if isinstance(value,dict):
        for k,v in value.items():
            if k.endswith('evidence_id') and isinstance(v,str):found.add(v)
            else:found.update(evidence_ids(v))
    elif isinstance(value,list):
        for item in value:found.update(evidence_ids(item))
    return found


def append_log(path,record):
    import fcntl
    with path.open('a',encoding='utf-8') as out:
        fcntl.flock(out,fcntl.LOCK_EX)
        out.write(canonical(record)+'\n');out.flush()
        fcntl.flock(out,fcntl.LOCK_UN)
    path.chmod(0o600)


class SessionGateway:
    def __init__(self,path):
        self.path=Path(path).resolve();self.session=read_json(self.path/'session.json')
        expected=self.session['files_sha256']
        for name,sha in expected.items():
            if name not in ('events.json.gz','catalogue.json','companies.json') or file_sha(self.path/name)!=sha:
                raise ValueError('Blind session data checksum mismatch.')
        events=json.loads(gzip.decompress((self.path/'events.json.gz').read_bytes()))
        catalogue=read_json(self.path/'catalogue.json');companies=read_json(self.path/'companies.json')
        arm=self.session['arm']
        if arm=='customer_only' and any(e['customer_id']!=self.session['company_id'] for e in events):
            raise ValueError('Customer-only bundle contains another company.')
        if arm=='reduced_semantic':
            forbidden={'short_name','ukrainian_name','product_name','base_name'}
            if any(forbidden.intersection(row) for row in catalogue.values()):raise ValueError('Reduced-semantic text leak.')
        self.tools=HistoryTools(events,catalogue,companies,as_of=self.session['prediction_date'],
                                customer_id=self.session['company_id'],allow_global=arm!='customer_only',semantic=arm!='reduced_semantic')
        self.log=self.path/'tool_log.jsonl'

    def _open(self):
        if (self.path/'seal.json').exists():raise PermissionError('Prediction sealed; historical queries are closed.')

    def records(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []

    def context(self):
        self._open()
        out={k:self.session[k] for k in ('case_id','prediction_date','benchmark','arm','candidate_ids')}
        out.update(candidate_pool_size=len(self.session['candidate_ids']),**self.tools.context(self.session['candidate_ids']))
        no_gold_fields(out)
        self._record('context',{},out)
        return out

    def _record(self,name,args,result):
        append_log(self.log,{'tool':name,'arguments':args,'response':result,'response_sha256':digest(result),
                             'response_characters':len(canonical(result))})

    def query(self,tool,arguments):
        self._open()
        if tool not in TOOLS or not isinstance(arguments,dict):raise ValueError('Unknown tool/argument object.')
        calls=[r for r in self.records() if r['tool']!='context']
        if len(calls)>=16:raise ValueError('The predeclared16-query budget is exhausted.')
        call_args=dict(arguments)
        if tool=='relations' and 'candidate_only' in call_args:
            if call_args.pop('candidate_only'):call_args['candidate_ids']=self.session['candidate_ids']
        out=getattr(self.tools,tool)(**call_args)
        no_gold_fields(out)
        self._record(tool,arguments,out)
        return out

    def submit(self,prediction):
        self._open();records=self.records()
        if (self.path.parent.parent/'reveal.json').exists():raise PermissionError('Gold has already been revealed; no new predictions permitted.')
        if not any(r['tool']=='context' for r in records):raise ValueError('Read the prediction context before submitting.')
        if self.session['arm']!='customer_only':
            observed={r['tool'] for r in records}
            if not {'similar_companies','relations'}<=observed or not observed.intersection({'trajectories','product_history','market_trends'}):
                raise ValueError('Global arms require similar-company, relationship and additional global evidence queries.')
        refs=evidence_ids([r['response'] for r in records])
        prior=set(self.tools._past(company_id=self.tools.customer_id).product_id)
        validate_prediction(prediction,self.session,evidence_ids=refs,prior_products=prior)
        save_json(self.path/'prediction.json',prediction)
        seal={'case_id':self.session['case_id'],'benchmark':self.session['benchmark'],'arm':self.session['arm'],
              'replicate':self.session.get('replicate',0),'prediction_sha256':file_sha(self.path/'prediction.json'),
              'session_sha256':file_sha(self.path/'session.json'),'tool_log_sha256':file_sha(self.log),
              'query_calls':sum(r['tool']!='context' for r in records),
              'query_response_characters':sum(r['response_characters'] for r in records),
              'llm_token_usage':None,'usage_limitation':'Connected runtime does not expose API token accounting to this gateway.',
              'explanations_frozen_before_reveal':True}
        save_json(self.path/'seal.json',seal)
        return {'case_id':seal['case_id'],'sealed_prediction_sha256':seal['prediction_sha256']}


def _write_private_text(path,text):
    path=Path(path)
    path.write_text(text,encoding='utf-8')
    path.chmod(0o600)
    return path


def _fmt_num(val):
    if val is None:return '-'
    if isinstance(val,float):return f'{val:.1f}' if abs(val)>=10 else f'{val:.2f}'
    return str(val)


def format_context_compact(ctx):
    cand_set=set(ctx['candidate_ids'])
    vm=ctx['visible_market']
    cp=ctx['company_profile']
    lines=[
        f"CASE: {ctx['case_id']} | DATE: {ctx['prediction_date']} | BENCHMARK: {ctx['benchmark']} | ARM: {ctx['arm']} | POOL: {ctx['candidate_pool_size']}",
        f"COMPANY: {ctx['company_id']} | INDUSTRY: {cp.get('industry_branch') or '-'} | CITY: {cp.get('city') or '-'} | REGION: {cp.get('region_code') or '-'} | BRANCH: {cp.get('sales_branch') or '-'}",
        f"COMPANY_DESC: {cp.get('description') or '-'}",
        f"FLAGS: global_history={ctx['global_history_available']} product_text={ctx['product_text_available']} metadata={ctx['metadata_policy']} | MARKET: events={vm['events']} companies={vm['companies']} products={vm['products']} last_date={vm['last_observed_date']}",
        f"HISTORY_EVIDENCE_ID: {ctx['customer_history_evidence_id']} | TOTAL_EVENTS: {ctx['customer_history_events']}",
        "",
        "## COMPLETE CUSTOMER BASKETS (strictly before prediction_date)",
    ]
    for b in ctx['complete_customer_baskets']:
        items=', '.join(f"{pid}({_fmt_num(qty)})" for pid,qty in b['items'])
        lines.append(f"{b['purchase_date']}: {items}")
    lines.extend(["", "## CUSTOMER SKU INTERVALS (sku | n | sum_q | first..last | days_ago | med_gap | mean_gap | std | last_q | evidence_id)"])
    for r in ctx['customer_intervals']['rows']:
        tag='CAND' if r['product_id'] in cand_set else 'HIST'
        lines.append(f"{r['product_id']}[{tag}] | n={r['purchase_events']} | sum={_fmt_num(r['quantity_sum'])} | {r['first_date']}..{r['last_date']} | ago={r['days_since_last']}d | med={_fmt_num(r['median_gap_days'])} | mean={_fmt_num(r['mean_gap_days'])} | std={_fmt_num(r['gap_std_days'])} | lastq={_fmt_num(r['last_quantity'])} | {r['evidence_id']}")
    prior_skus={r['product_id'] for r in ctx['customer_intervals']['rows']}
    header_fields='idx | pid | role | brand | cat/subcat | line | family | container/unit | name | evidence_id' if ctx['product_text_available'] else 'idx | pid | role | brand | cat/subcat | line | family | container/unit | evidence_id'
    lines.extend(["", f"## CATALOGUE ({header_fields})"])
    cat_start=len(lines)
    for idx,r in enumerate(ctx['catalogue']['rows'],1):
        pid=r['product_id']
        role=('CAND+PREV' if pid in prior_skus else 'CAND+NEW') if pid in cand_set else 'HIST_ONLY'
        base=f"{idx:03d} | {pid} | {role} | {r.get('brand') or '-'} | {r.get('product_category') or '-'}/{r.get('product_subcategory') or '-'} | {r.get('business_line') or '-'} | {r.get('product_family') or '-'} | {r.get('container_type') or '-'}/{r.get('unit') or '-'}"
        if ctx['product_text_available']:
            name=r.get('product_name') or r.get('ukrainian_name') or r.get('short_name') or r.get('base_name') or '-'
            lines.append(f"{base} | {name} | {r['evidence_id']}")
        else:
            lines.append(f"{base} | {r['evidence_id']}")
    return '\n'.join(lines),cat_start


def format_query_compact(tool,out):
    lines=[f"## TOOL: {tool} | total_rows={out['total_rows']} offset={out['offset']} limit={out['limit']} next_offset={out['next_offset']}"]
    if tool=='relations':
        lines.append(f"relation={out['relation']} match={out['match']} anchors={out['anchor_ids']} source_baskets={out['source_baskets']} all_pairs={out.get('all_observed_pairs')} candidate_only={out.get('filtered_to_candidates')}")
        for r in out['rows']:
            ex=','.join(f"{e['company_id']}:{e['source_date']}->{e['destination_date']}" for e in r['examples'][:2])
            lines.append(f"  {r['destination']} | sup={r['support_baskets']}/{r['source_baskets']} ({r['observed_conditional_frequency']:.3f}) | comp={r['distinct_companies']} | bg={r['background_baskets']} | ex=[{ex}] | {r['evidence_id']}")
    elif tool=='similar_companies':
        for r in out['rows']:
            other=','.join(r['other_products'][:12])+(f"(+{len(r['other_products'])-12})" if len(r['other_products'])>12 else '')
            shared=','.join(r['shared_products'][:10])+(f"(+{len(r['shared_products'])-10})" if len(r['shared_products'])>10 else '')
            lines.append(f"  {r['company_id']} | jacc={r['jaccard']:.3f} | same_ind={r['same_industry']} ({r['industry_branch'] or '-'}) | skus={r['observed_skus']} | shared=[{shared}] | other=[{other}] | {r['evidence_id']}")
    elif tool=='market_trends':
        for r in out['rows']:
            recent_m=','.join(f"{k}:{v}" for k,v in list(r['monthly_events'].items())[-6:])
            lines.append(f"  {r['product_id']} | life={r['lifetime_events']}({r['lifetime_buyers']}c) | 7d={r['events_7d']}({r['buyers_7d']}c) | 30d={r['events_30d']}({r['buyers_30d']}c) | 90d={r['events_90d']}({r['buyers_90d']}c) | months=[{recent_m}] | {r['evidence_id']}")
    elif tool=='trajectories':
        lines.append(f"qualifying={out['qualifying_companies']} adopters={out['later_adopting_companies']}")
        for r in out['rows']:
            traj='; '.join(f"{b['purchase_date']}:{'+'.join(b['product_ids'][:6])}" for b in r['observed_trajectory'][:4])
            lines.append(f"  {r['company_id']} ({r['industry_branch'] or '-'}) | {r['anchor_completion_date']}->{r['destination_first_date']} ({r['elapsed_days']}d) | [{traj}] | {r['evidence_id']}")
    elif tool in ('company_history','product_history'):
        if tool=='product_history':
            lines.append(f"distinct_companies={out['distinct_companies']} observed_events={out['observed_events']}")
        for r in out['rows']:
            lines.append(f"  {r['purchase_date']} | {r['company_id']} | {r['product_id']} | q={_fmt_num(r['quantity'])} | {r['evidence_id']}")
    elif tool=='intervals':
        for r in out['rows']:
            lines.append(f"  {r['product_id']} | n={r['purchase_events']} sum={_fmt_num(r['quantity_sum'])} {r['first_date']}..{r['last_date']} ago={r['days_since_last']}d med={_fmt_num(r['median_gap_days'])} lastq={_fmt_num(r['last_quantity'])} | {r['evidence_id']}")
    else:
        lines.append(json.dumps(out,ensure_ascii=False,allow_nan=False))
    return '\n'.join(lines)


def _emit_bounded(text,view_path,command_prefix,*,max_chars=9000):
    all_lines=text.splitlines()
    if len(text)<=max_chars:
        print(text)
        return
    _write_private_text(view_path,text+'\n')
    chunk=[];total=0
    for line in all_lines:
        if total+len(line)+1>max_chars and chunk:break
        chunk.append(line);total+=len(line)+1
    shown=len(chunk)
    print('\n'.join(chunk))
    print(f"\n[OUTPUT CONTINUES: lines {shown+1}..{len(all_lines)} of {len(all_lines)} saved in {view_path.name}. Run: {command_prefix} view --file {view_path.name} --start {shown+1} --end {len(all_lines)}]")


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--session',required=True);parser.add_argument('--json',action='store_true',dest='as_json')
    sub=parser.add_subparsers(dest='action',required=True)
    sub.add_parser('context')
    query=sub.add_parser('query');query.add_argument('--tool',required=True);query.add_argument('--arguments',default='{}')
    batch=sub.add_parser('batch');batch.add_argument('--requests',required=True)
    view=sub.add_parser('view');view.add_argument('--file',default='context_view.txt');view.add_argument('--start',type=int,default=1);view.add_argument('--end',type=int,default=200)
    submit=sub.add_parser('submit');submit.add_argument('--file');submit.add_argument('--stdin',action='store_true')
    args=parser.parse_args()
    try:
        session=SessionGateway(args.session)
        cmd_prefix=f'{sys.executable} {Path(__file__).resolve()} --session {session.path}'
        if args.action=='context':
            result=session.context()
            if args.as_json:print(json.dumps(result,ensure_ascii=False,allow_nan=False))
            else:
                text,_=format_context_compact(result)
                _write_private_text(session.path/'context_view.txt',text+'\n')
                _emit_bounded(text,session.path/'context_view.txt',cmd_prefix)
        elif args.action=='view':
            if args.file not in ('context_view.txt','last_query_view.txt'):
                raise ValueError('Only session view files may be read via view.')
            vpath=session.path/args.file
            if not vpath.is_file():raise ValueError(f'{args.file} does not exist yet; run context/query first.')
            all_lines=vpath.read_text(encoding='utf-8').splitlines()
            start=max(1,args.start);end=min(len(all_lines),max(start,args.end))
            selected=all_lines[start-1:end]
            chunk=[];total=0
            for line in selected:
                if total+len(line)+1>9000 and chunk:break
                chunk.append(line);total+=len(line)+1
            actual_end=start-1+len(chunk)
            print('\n'.join(chunk))
            if actual_end<len(all_lines):
                print(f"\n[OUTPUT CONTINUES: lines {actual_end+1}..{len(all_lines)} of {len(all_lines)}. Run: {cmd_prefix} view --file {args.file} --start {actual_end+1} --end {len(all_lines)}]")
        elif args.action=='query':
            result=session.query(args.tool,json.loads(args.arguments))
            if args.as_json:print(json.dumps(result,ensure_ascii=False,allow_nan=False))
            else:
                text=format_query_compact(args.tool,result)
                _emit_bounded(text,session.path/'last_query_view.txt',cmd_prefix)
        elif args.action=='batch':
            requests=json.loads(args.requests)
            if not isinstance(requests,list) or not 1<=len(requests)<=8:raise ValueError('Batch size must be1..8.')
            results=[(request['tool'],session.query(request['tool'],request.get('arguments',{}))) for request in requests]
            if args.as_json:print(json.dumps([r for _,r in results],ensure_ascii=False,allow_nan=False))
            else:
                text='\n\n'.join(format_query_compact(t,r) for t,r in results)
                _emit_bounded(text,session.path/'last_query_view.txt',cmd_prefix)
        else:
            if args.stdin:
                payload=json.loads(sys.stdin.read())
                save_json(session.path/'answer.json',payload,exclusive=False)
                result=session.submit(payload)
            else:
                if not args.file:raise ValueError('Provide --file or --stdin.')
                path=Path(args.file).resolve()
                if path!=session.path/'answer.json':raise ValueError('Write only the prescribed answer.json file.')
                result=session.submit(read_json(path))
            print(json.dumps(result,ensure_ascii=False,allow_nan=False))
    except (ValueError,PermissionError,TypeError,KeyError,json.JSONDecodeError) as error:
        print(canonical({'error':type(error).__name__,'message':str(error)}));return 2
    return 0


if __name__=='__main__':raise SystemExit(main())

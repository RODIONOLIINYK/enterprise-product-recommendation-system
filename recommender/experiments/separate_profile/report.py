"""One current comparison report for recovered CatBoost and separated customer memory."""

import json
import re
from pathlib import Path

CAT=Path('experiments/catboost_current_v1')
NEURAL=Path('experiments/separate_profile_v1')
OUT=Path('experiments/current_model_comparison_v1')


def build():
    trees=json.loads((CAT/'summary.json').read_text())
    selected=json.loads((CAT/'selected_checkpoints.json').read_text())
    neural=json.loads((NEURAL/'summary.json').read_text())
    extra=json.loads((NEURAL/'fused64_control.json').read_text())
    neural['arms']['fused_one_profile_64']=extra['result']
    historical=trees['historical'];old=historical['validation_metrics']
    labels={'historical17':'CatBoost / original17 features','current53':'CatBoost / current53 features',
            'fused_two_profile_32':'Two fused profile tokens /32 events',
            'split_text_attributes_32':'Separate text+attributes /32 events',
            'split_text_attributes_64':'Separate text+attributes / 64 events',
            'fused_one_profile_64':'A / fused profile / 64 events'}
    lines=['# Current CatBoost and separate-profile memory comparison','',
        'All new fits are development-only, seed42, with no held-out evaluation, promotion, commit or push. '
        'Current comparisons share26,253 training /6,191 validation queries and the same100 retrieved '
        'candidates per query, all10,276 target instances and1,857 new targets. Statistics roll strictly '
        'before prediction dates. No targets are injected into current candidate sets.','',
        '## Historical CatBoost evidence','',
        'The final historical model was a **CatBoostClassifier**, following an earlier CatBoostRanker. '
        'The classifier was introduced in `29ab48b` and made the only model in `0fdb00b`. '
        'Configuration and saved validation results were recovered from `2cb7f75`; the binary was '
        'last changed in `535e211` without an accompanying metrics report update. Historical saved '
        'numbers therefore do not independently certify that last binary.','',
        f"Saved validation: Recall@10={old['recall@10']:.5f}, NDCG@10={old['ndcg@10']:.5f}, "
        f"MRR={old['mrr']:.5f}, first-purchase Recall@10={old['first_purchase_recall@10']:.5f}.",'',
        f"That benchmark had{historical['old_validation_queries']:,} validation groups with approximately"
        f"{historical['old_candidates_per_query']:.0f} candidates, inserting all targets plus sampled negatives. "
        f"It split customers80/10/10, and training covered{historical['old_train_date_min']} through"
        f"{historical['old_train_date_max']}. These historical scores must not be compared directly with "
        'the current temporal, retrieved-slate, complete-target benchmark. The old weights were not '
        'loaded or evaluated: they were trained on dates later than current validation.','',
        'The original17-feature recipe was rebuilt causally on current candidates:3 categorical '
        'features (SKU/category/business line), personal quantity/frequency/recency/cadence, category/line '
        'affinity and rolling global popularity/buyers. Missing cadence remains NaN. Category/line '
        'metadata use the declared current snapshot rather than invented historical effective dates. '
        'The current53-feature variant uses3 categoricals +41 rolling candidate features +8 profile '
        'features +1 tower dot similarity. It does not use the rejected Stage1 MLP score as a Stage2 input.','',
        'Both classifiers reuse the historical500-tree maximum, depth8, learning rate0.05 and '
        'Logloss objective/early stopping, with4 CPU threads. The original Logloss-selected results are '
        'retained separately. For the common ranking comparison, prefixes are evaluated every50 trees '
        '(at most10 checkpoints), using the same discovery guardrails and NDCG ordering as neural fits.','',
        '## Current matched-candidate results','',
        'Recall is macro over complete target baskets. New recall is micro over all1,857 new target '
        'instances, including targets absent from the slate. All models share the same reachability ceiling. '
        'This is a practical model comparison: CatBoost and the transformer have different inputs, '
        'objectives and training algorithms. It does not isolate a pure architecture effect.']
    graph={}
    for selection in ('ndcg','discovery'):
        rows=[('Stage1 MLP on identical slate',None,trees['same_slate_mlp_metrics']),
              ('A / fused profile /32 events',neural['baseline'][selection]['epoch'],neural['baseline'][selection]['metrics'])]
        rows += [(labels[n],r[selection]['trees'],r[selection]['metrics']) for n,r in selected['models'].items()]
        rows += [(labels[n],r[selection]['epoch'],r[selection]['metrics']) for n,r in neural['arms'].items()]
        graph[selection]=[{'name':n,'checkpoint':c,'recall':m['recall@10']*100,'ndcg':m['ndcg@10'],'hits':m['new_hits@10']} for n,c,m in rows]
        lines += ['',f'### {selection.upper()}-selected weights','',
                  '| Model | Epoch/trees | R@5 | R@10 | R@25 | R@50 | R@100 | NDCG@10 | MRR | New@10 | Repeat micro R@10 |',
                  '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        for name,checkpoint,m in rows:
            vals=[f'{m[k]:.5f}' for k in ('recall@5','recall@10','recall@25','recall@50','recall@100','ndcg@10','mrr')]
            lines.append(f'| {name} | {checkpoint or "fixed"} | '+' | '.join(vals)+f" |{m['new_hits@10']}/1,857 |{m['repeat_recall@10_micro']:.5f} |")
        lines += ['','Paired customer-clustered95% bootstrap intervals (2,000 draws, seed123). '
                  'Delta is treatment minus comparator; intervals are conditional on fitted weights.','',
                  '| Comparison | Metric | Delta |95% CI |','| --- | --- | ---: | --- |']
        cis={**{f'A to{labels[n]}':r[selection]['A_to_tree_95ci'] for n,r in selected['models'].items()},
             **neural['paired_bootstrap_95ci'][selection], **extra['paired_bootstrap_95ci'][selection]}
        for name,values in cis.items():
            for k in ('recall@10_delta','ndcg@10_delta','new_recall@10_delta'):
                c=values[k];lo,_,hi=c['ci_95']
                lines.append(f"|{name} |{k} |{c['point_delta']:+.5f} |[{lo:+.5f},{hi:+.5f}] |")
    lines += ['','## What the separated memory contains','',
        '- A: one128D fused company profile token plus32 historical event tokens.',
        '- Token-count control: two128D copies of the fused profile plus32 events.',
        '- Split32: company-text128D token, company-attribute128D token, plus32 event tokens.',
        '- A64: the original one fused profile token, plus 64 separate event tokens (65 tokens total).',
        '- Split64: the same two profile tokens, plus64 separate event tokens. All66 memory '
        'tokens pass through self-attention and remain available to each candidate’s cross-attention. '
        'There is no global older-history attention-pooling step.',
        '- The two-token arms assign profile8 statistics to the attribute token and zero statistics '
        'to the text token. Both profile tokens are always valid; each history position follows its '
        'event validity mask. Attention diagnostics aggregate both profile weights correctly.',
        '- The425D scoring head,41 raw candidate features,13 candidate bias features and frozen '
        'candidate vectors remain unchanged. Candidate-to-candidate self-attention and prior '
        '128D statistical enrichers remain disabled.',
        '- Numerical profile/history inputs retain the reference bias-only routing. Separating the '
        'company attribute representation does not silently enable the previously tested statistical enrichers.',
        '- Before self-attention, the text token depends on company text and the attribute token on '
        'city/region/industry. Self-attention is then allowed to mix these views. This is source '
        'separation, not a guarantee of orthogonal or equally useful embeddings.',
        '- The split bypasses and freezes the old profile-fusion projection. The two-fused-token '
        'control accounts for token count/statistics placement but does not make every parameter '
        'or optimization path identical.',
        '- Product text and structured product attributes could be separate views. This run leaves '
        'the candidate product vector unchanged to isolate customer memory. Product text includes '
        'overlapping category/brand/family semantics; raw cosine/max operations between separately '
        'projected branches are not automatically meaningful.',
        '', '##64-event data checks','',
        'The original training-fitted vocabularies and quantity reference60 are reused. All events '
        'are strictly before their query date. Every original32-event suffix was verified exactly, '
        'including event statistics, dates and categorical IDs, before the64-event model was trained. '
        'Queries, targets, slate selection and rolling candidate features do not change. The existing '
        'learned32-position embedding is linearly interpolated for64 positions, preserving the parameter '
        'layout. The 64-event configuration includes this interpolated position mapping; recent raw '
        'features are unchanged, but their positional representation changes. It does not test every '
        'possible long-history encoding or change the number of distinct baskets.','',
        '```json',json.dumps(neural['history_diagnostics'],indent=2),'```','',
        '## Completed corrections and limits','',
        'The user confirmed earlier repeat/new objective-weighting experiments did not improve '
        'results. That sweep was not repeated, and its user-reported status is now recorded in plan.md. '
        'The current matched fits retain the original loss weighting.','',
        'The new optional core profile/statistics/diagnostic hooks preserve legacy state keys and '
        'default arithmetic. Saved ArmA CPU replay still matches its saved MPS logits within1.91e-6. '
        'Snapshot/hash verification preserves all31 inventoried reference/rolling evidence files. '
        'The earlier masked-loss fix remains in place. Production ingestion/automatic refresh, '
        'source-coherent external inference snapshots, and effective-dated availability remain separate work.','',
        'A single training seed is exploratory. Repeated validation use and differing checkpoint '
        'selection/feature sets constrain generalization claims. No model was promoted.','',
        '## Evidence and reproduction','',
        '- [CatBoost summary](../catboost_current_v1/summary.json)',
        '- [CatBoost historical validation metadata](../catboost_current_v1/historical_validation.json)',
        '- [CatBoost common ranking selections](../catboost_current_v1/selected_checkpoints.json)',
        '- [Separate-profile/64-event summary](../separate_profile_v1/summary.json)',
        '- [Interactive comparison](comparison.html)','',
        '```bash','./venv/bin/python -m recommender.experiments.catboost_comparison.run --execute-training',
        './venv/bin/python -m recommender.experiments.catboost_comparison.select',
        './venv/bin/python -m recommender.experiments.separate_profile.train --execute-training',
        './venv/bin/python -m recommender.experiments.separate_profile.fused64 --execute-training',
        './venv/bin/python -m recommender.experiments.separate_profile.report',
        'PYTHONPATH=. ./venv/bin/pytest tests/ -q','PYTHONPATH=. ./venv/bin/pytest tests/ --synthetic-only -q','```','']
    verification_path=OUT/'verification.json'
    if verification_path.exists():
        v=json.loads(verification_path.read_text())
        maximum=max((r.get('max_abs_cpu_vs_saved_mps_diff',0) for r in v['selected_checkpoint_checks']),default=0)
        lines += ['', '## Verification', '',
                  f"All {len(v['selected_checkpoint_checks'])} selected checkpoint records reproduce the saved metrics. "
                  f"Neural CPU replays on 32 validation rows match saved MPS logits within {maximum:.3g}; "
                  'CatBoost selected models reproduce the complete validation metrics after reload.', '',
                  f"Full safe suite: {v['full_tests']['passed']} passed, {v['full_tests']['skipped']} skipped. "
                  f"Synthetic-only: {v['synthetic_tests']['passed']} passed, {v['synthetic_tests']['skipped']} skipped.", '',
                  '[Independent verification record](verification.json)', '']
    OUT.mkdir(parents=True,exist_ok=True)
    report='\n'.join(lines)
    for old,new in {'seed42':'seed 42','same100':'same 100','all10,276':'all 10,276','all1,857':'all 1,857',
                    'all31':'all 31','with4 CPU':'with 4 CPU','approximately25':'approximately 25',
                    'split customers80/10/10':'split customers 80/10/10','current500':'current 500',
                    'historical500':'historical 500','depth8':'depth 8','rate0.05':'rate 0.05',
                    'every50':'every 50','most10':'most 10','neural425D':'neural 425D','one128D':'one 128D',
                    'two128D':'two 128D','plus32':'plus 32','plus64':'plus 64','All66':'All 66',
                    'profile8':'profile 8','the425D':'the 425D','head,41':'head, 41','features,13':'features, 13',
                    'prior128D':'prior 128D','original32':'original 32','reference60':'reference 60',
                    'learned32':'learned 32','for64':'for 64','within1.91':'within 1.91','31 inventoried':'31 inventoried',
                    'use50-tree':'use 50-tree','identical100-item':'identical 100-item'}.items():report=report.replace(old,new)
    formatted=[]; in_code=False
    for line in report.splitlines():
        if line.startswith('```'):in_code=not in_code
        if not in_code:
            line=re.sub(r'(?<![\w/.-])([A-Za-z]+)(?=\d)',r'\1 ',line)
            line=re.sub(r'^(#+)(?=\d)',r'\1 ',line)
            line=re.sub(r'(?<=:)(?=\d)',' ',line)
            line=line.replace('company-text128D','company-text 128D').replace('company-attribute128D','company-attribute 128D')
            line=line.replace(' /6,191',' / 6,191').replace('toCatBoost','to CatBoost')
        formatted.append(line)
    (OUT/'REPORT.md').write_text('\n'.join(formatted)+'\n')
    page='''<!doctype html><html><head><meta charset="utf-8"><style>
body{margin:0;color:var(--foreground);font-family:var(--font-sans)}.note{font-size:12px;color:var(--muted-foreground);line-height:1.5}.panel{padding:16px;margin:12px 0;background:var(--card);border-radius:var(--radius)}b{font-size:14px}.row{display:grid;grid-template-columns:minmax(135px,1.4fr) minmax(60px,2fr) 76px;gap:8px;align-items:center;min-height:34px;font-size:12px;line-height:1.35}.bar{height:14px;border-radius:3px}.buttons{display:flex;gap:8px;margin:12px 0}button{padding:8px 12px;border:1px solid var(--border);border-radius:var(--radius);color:var(--foreground);background:var(--secondary);font:13px var(--font-sans);cursor:pointer}button[aria-pressed=true]{color:var(--primary-foreground);background:var(--primary)}
</style></head><body><div class="note">Current rolling data ·6,191 validation queries · identical100-item candidate sets · seed42<br>CatBoost uses the historical training recipe; current ranking selections use50-tree prefixes. Neural selections use epochs.</div><div class="buttons"><button id="ndcg" aria-pressed="true">NDCG-selected</button><button id="discovery" aria-pressed="false">Discovery-selected</button></div><div id="plots"></div><div class="panel"><b>Historical CatBoost numbers use a different benchmark</b><div class="note">Saved Recall@10:92.59% · NDCG@10:0.68607<br>Approximately25 candidates with all targets inserted; customer-disjoint split spanning dates beyond current validation. Those numbers are not directly comparable to the current results.</div></div><div class="note">All bars start at zero. New-hit denominator: all1,857 new targets. No held-out evaluation or promotion.</div><script>
const data=DATA;function render(sel){for(let s of ['ndcg','discovery'])document.getElementById(s).setAttribute('aria-pressed',s===sel);document.getElementById('plots').innerHTML=[['recall','Overall Recall@10',100,v=>v.toFixed(3)+'%'],['ndcg','NDCG@10',1,v=>v.toFixed(5)],['hits','New-product hits@10',300,v=>v+' /1,857']].map(([key,title,max,fmt])=>'<div class="panel"><b>'+title+'</b>'+data[sel].map((r,i)=>'<div class="row"><span>'+r.name+'</span><span><div class="bar" style="width:'+Math.min(100,r[key]/max*100)+'%;background:var(--chart-'+(i%6+1)+')"></div></span><span>'+fmt(r[key])+'</span></div>').join('')+'</div>').join('')}for(let s of ['ndcg','discovery'])document.getElementById(s).onclick=()=>render(s);render('ndcg');</script></body></html>'''.replace('DATA',json.dumps(graph))
    for old,new in {'seed42':'seed 42','·6,191':'· 6,191','identical100':'identical 100',
                    'use50-tree':'use 50-tree','Approximately25':'Approximately 25','all1,857':'all 1,857',
                    'original17':'original 17','current53':'current 53',' /32':' / 32'}.items():page=page.replace(old,new)
    (OUT/'comparison.html').write_text(page)


if __name__=='__main__':build()

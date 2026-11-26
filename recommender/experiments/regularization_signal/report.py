"""Report fixed-input MLP/regularization/augmentation evidence with explicit model identities."""

import base64
import json
from pathlib import Path

from recommender.artifacts import atomic_json
from recommender.experiments.dynamic_selector.feature_schema import NAMES, RELATION_NAMES
from .model import VARIANTS
from .train_ranker import EVIDENCE

RANKER_LABELS={'control':'Fresh control','head_half':'Half scoring-head dropout',
               'profile_half':'Half profile dropout','attention_half':'Half attention dropout',
               'weight_decay_zero':'Zero weight decay','all_zero':'All active dropout/weight decay zero',
               'direct52_all_zero':'Direct52 + all regularization zero'}
MLP_LABELS={'mlp_fixed_regularized':'MLP / fixed pairs / dropout .2,.1',
            'mlp_fixed_unregularized':'MLP / fixed pairs / zero dropout and weight decay',
            'mlp_rotating_negatives':'MLP / rotating negatives / zero dropout and weight decay'}


def fmt_ci(metrics,name):
    m=metrics[name+'_delta'];lo,_,hi=m['ci_95']
    return f"{m['point_delta']:+.5f} [{lo:+.5f}, {hi:+.5f}]"


def build():
    ranker=json.loads((EVIDENCE/'ranker_summary.json').read_text())
    tab=json.loads((EVIDENCE/'tabular_summary.json').read_text())
    tv=json.loads((EVIDENCE/'tabular_verification.json').read_text())
    rv=json.loads((EVIDENCE/'ranker_verification.json').read_text())
    if ranker['frozen_inputs']!=tab['frozen_inputs']:raise AssertionError('Frozen input snapshots differ.')
    active=json.loads(Path('checkpoints/stage1-mlp-reference/repeat-no-month-v1/manifest.json').read_text())
    contract=active['mlp_contract']['input_contract'];retained=contract['retained_context_names']
    transitions=[]
    for name in RELATION_NAMES:
        source=NAMES.index(name)
        if name not in retained:raise AssertionError('A transition input is absent from the active selector.')
        transitions.append({'name':name,'source41_column':source,'active_stage1_mlp_column':256+retained.index(name),
                            'pointwise_stage2_mlp_column':397+source-13,
                            'statistical_mlp_numeric_column':source,'catboost_field':name})
    atomic_json({'relations':transitions,'ordered_raw_feature_names':tab['feature_names'],
                 'raw_field_count':52,'numeric_value_parity_rows':4096,
                 'frozen_inputs':ranker['frozen_inputs'],'catboost_retrained_or_changed':False,
                 'statistics_policy':'rolling_observed_v2','heldout_access':False},EVIDENCE/'feature_lock.json')
    lines=['# Fixed-input regularization and negative-resampling experiments','',
           'Training is before 2025-10-01, validation is 2025-10-01 through 2026-02-18. '
           '26,253 training / 6,191 validation queries; 10,276 full validation targets, including 1,857 new targets. '
           'Seed42; ten epochs per fit. No held-out access, promotion, commit or push.','',
           'Observed result: the 32,953-parameter full-pool MLP improved from NDCG@10=0.58915 with regularization '
           'to0.59143 with zero dropout/weight decay, then0.59372 with rotating negatives. The frozen CatBoost '
           'has0.59373. This is a useful small-model result; uncertainty does not establish superiority over '
           'CatBoost. None of the Stage2 NDCG-selected treatments beats the saved rolling reference0.60223. '
           'The zero-regularization discovery checkpoint reaches258 new hits versus255 for the reference, '
           'with lower overall Recall@10 and NDCG. References remain unchanged.','',
           '## Transition features already reach both neural scorers','',
           'The active Stage1 MLP keeps 39/41 context features, omitting only month sine/cosine, '
           'plus six customer statistics. The pointwise Stage2 scorer receives all 41 context features directly. '
           'The following are physical input columns, not merely indirect information inside embeddings:','',
           '| Feature | Active Stage1 MLP column | Pointwise Stage2 MLP column | CatBoost input |',
           '| --- | ---: | ---: | --- |']
    for row in transitions:
        lines.append(f"| {row['name']} | {row['active_stage1_mlp_column']} | {row['pointwise_stage2_mlp_column']} | direct scalar |")
    lines+=['','All columns are zero-indexed. Counts use strictly prior observed purchases, never prediction-day '
            'or held-out purchases. Same-basket statistics describe earlier observed baskets. The direct52 '
            'treatment adds all profile8 scalars and explicit candidate SKU/category/business-line embeddings '
            'to the Stage2 scoring head; the existing transition inputs remain present.','',
            '## CatBoost identities: avoid conflating three experiments','',
            '1. **Historical 17-field classifier:** three categorical fields plus fourteen numerical features. '
            'The recipe is preserved in `experiments/catboost_current_v1/historical_validation.json`. Its '
            'historical customer-disjoint, approximately25-candidate, target-inserted result is not comparable '
            'to this temporal full-pool benchmark. The later causal reconstruction of that recipe on the current '
            '100 retrieved candidates reached R@10=0.75793, NDCG@10=0.59457, 143 new hits. It depended on that candidate policy.',
            '2. **Current 53-field retrieved-slate classifier:** three categoricals +41 rolling scalars +8 profile '
            'scalars +one tower-dot score. This is a distinct earlier experiment, not a standalone raw-data model.',
            '3. **Standalone 52-field classifier:** the same three categoricals, 41 rolling scalars and profile8, '
            'without the tower-dot score or a neural shortlist. It scores every eligible SKU. This was a new '
            'baseline added to answer the standalone question, not a replacement or reproduction of the original17-field recipe.',
            '', 'This campaign freezes the third model, its 52-field schema, numeric values, source identities, '
            'saved model files and results. CatBoost is neither retrained nor changed. See [feature_lock.json](feature_lock.json).','',
            '## Full-pool MLP: use exactly those information inputs','',
            'The standalone MLP has 49 numeric inputs plus the same three categorical fields, encoded with learned '
            'SKU16/category8/line8 embeddings; head81→128→64→1, LayerNorm and GELU, 32,953 parameters. '
            'No company/product text vectors, pretrained towers, Stage1 scores, history Transformer or retrieved '
            'shortlist are inputs. Categorical encoding differs naturally between model families; raw fields and '
            'numeric values do not. Direct parity against CatBoost frame construction was checked on4,096 sampled rows.',
            '', 'Fixed-pair arms use exactly the seed42 CatBoost training sample:2,670,525 pairs, including45,225 '
            'eligible positives, representing15,999,865 eligible pairs through importance weights. All positives '
            'have weight1; uniformly sampled negatives have weight eligible_negative_count/sample_count. The '
            'objective is the same weighted binary Logloss; no pos_weight10, listwise term or repeat/new weighting. '
            'A fixed global importance normalizer makes minibatch gradients estimate the full sampled objective. '
            'All models rank all4,700,328 eligible validation pairs without inserting targets.',
            '', 'MLP arms share initial weights and independent deterministic batch orders, AdamW lr0.001, '
            'cosine decay to1e-5, batch4096, CPU2 threads, ten epochs. Removing dropout and weight decay together '
            'is a combined regularization change. The third arm changes only the sampled negatives relative to '
            'the second. Per-epoch draw digests, positives and importance mass are recorded.','',
            '| NDCG-selected model | Epoch/trees | Recall@10 | NDCG@10 | New hits@10 | Qualified |',
            '| --- | ---: | ---: | ---: | ---: | --- |']
    tree=tab['catboost_baseline']['ndcg'];m=tree['metrics']
    lines.append(f"| Frozen standalone CatBoost52 | {tree['trees']} | {m['recall@10']:.5f} | {m['ndcg@10']:.5f} | {m['new_hits@10']} | {tree['qualified']} |")
    for name,arm in tab['arms'].items():
        r=arm['selected']['ndcg'];m=r['metrics']
        lines.append(f"| {MLP_LABELS[name]} | {r['epoch']} | {m['recall@10']:.5f} | {m['ndcg@10']:.5f} | {m['new_hits@10']} | {r['qualified']} |")
    lines+=['','Qualification uses the preserved discovery guardrails:Recall@10≥0.75588 and NDCG@10≥0.59189. '
            'The first two MLP arms have diagnostic discovery selections but do not pass the NDCG guardrail. '
            'Full discovery/logloss selections are retained in [tabular_summary.json](tabular_summary.json); '
            'do not mix their hits with NDCG-selected metrics.','',
            'Paired customer-clustered bootstrap95% intervals, conditional on selected weights:', '',
            '| Comparison | NDCG@10 delta [95% CI] | Recall@10 delta [95% CI] |',
            '| --- | --- | --- |']
    comparisons=[('Regularized fixed → unregularized fixed',tab['paired_bootstrap_95ci']['ndcg']['mlp_fixed_unregularized']),
                 ('Unregularized fixed → rotating negatives',tv['fixed_unregularized_to_rotating_95ci']),
                 ('Frozen CatBoost → rotating-negative MLP',tab['arms']['mlp_rotating_negatives']['selected']['ndcg']['tree_to_mlp_95ci'])]
    for label,ci in comparisons:
        lines.append(f"| {label} | {fmt_ci(ci,'ndcg@10')} | {fmt_ci(ci,'recall@10')} |")
    lines+=['','The rotating-negative MLP numerically matches CatBoost NDCG and has more top10 new hits '
            '(224 versus202), but the paired CatBoost comparison does not establish superiority. It is a promising '
            'small standalone alternative, not an end-to-end neural pipeline promotion. The existing rolling '
            'Stage1 frozen selector scores R@10=0.75845/NDCG@10=0.58822/206 new hits under the saved replay; '
            'this new MLP changes inputs, objective and training as well as representation, so that comparison '
            'is practical rather than an architecture-only ablation. New MLP scores were not used to rebuild '
            'Stage2 candidate slates in this campaign.','',
            '## Stage2 regularization: keep architecture and statistics fixed','',
            'Seven fresh fits share the same initial weights, original32-event memory, head256/128/64, '
            'rolling quota100_50 candidates, RankerLoss, lr0.001 cosine schedule, batch64, ten epochs and MPS. '
            'Candidate self-attention, token enrichment and Stage1 MLP score inputs remain disabled. '
            'Head half-dropout, profile half-dropout, attention half-dropout and zero weight decay are separate '
            'single-axis changes. The final two runs deliberately combine changes. Both module Dropout and '
            'MultiheadAttention float dropout are versioned/reconstructed in saved checkpoints.','',
            '| Variant | Head dropout | Profile dropout | Memory/cross attention dropout | Weight decay | Direct52 |',
            '| --- | --- | --- | --- | ---: | --- |']
    for name,cfg in VARIANTS.items():
        h=cfg['head_scale'];p=cfg['profile_scale'];a=cfg['attention_scale']
        lines.append(f"| {RANKER_LABELS[name]} | {','.join(f'{v*h:.2f}' for v in (.2,.2,.1))} | {.1*p:.2f} | {.1*a:.2f} | {cfg['weight_decay']} | {cfg['direct52']} |")
    lines+=['','Profile means company-text and tabular projections; attention includes memory attention-weight, '
            'memory feedforward/residual and candidate cross-attention dropout. Inactive attached ProductTower, '
            'unused old head and disabled candidate-attention branch are not changed. Frozen Stage1 towers are '
            'not retrained: changing dropout at request time cannot change cached vectors.','',
            '| NDCG-selected model | Epoch | Recall@10 | NDCG@10 | New hits@10 |',
            '| --- | ---: | ---: | ---: | ---: |']
    r=ranker['baseline']['ndcg'];m=r['metrics']
    lines.append(f"| Saved rolling pointwise A | {r['epoch']} | {m['recall@10']:.5f} | {m['ndcg@10']:.5f} | {m['new_hits@10']} |")
    for name,arm in ranker['arms'].items():
        r=arm['ndcg'];m=r['metrics']
        lines.append(f"| {RANKER_LABELS[name]} | {r['epoch']} | {m['recall@10']:.5f} | {m['ndcg@10']:.5f} | {m['new_hits@10']} |")
    lines+=['','Paired NDCG-selected comparisons:', '',
            '| Variant | NDCG delta vs fresh control [95% CI] | NDCG delta vs saved A [95% CI] |',
            '| --- | --- | --- |']
    for name,comparisons in ranker['paired_bootstrap_95ci']['ndcg'].items():
        if name=='control':continue
        lines.append(f"| {RANKER_LABELS[name]} | {fmt_ci(comparisons['fresh_control_to_treatment'],'ndcg@10')} | "
                     f"{fmt_ci(comparisons['saved_A_to_treatment'],'ndcg@10')} |")
    lines+=['','Discovery-selected records are separately shown to keep checkpoint selection explicit:','',
            '| Variant | Epoch | Recall@10 | NDCG@10 | New hits@10 |',
            '| --- | ---: | ---: | ---: | ---: |']
    for name,arm in ranker['arms'].items():
        r=arm['discovery'];m=r['metrics']
        lines.append(f"| {RANKER_LABELS[name]} | {r['epoch']} | {m['recall@10']:.5f} | {m['ndcg@10']:.5f} | {m['new_hits@10']} |")
    coverage=tv['rotating_negative_coverage']
    lines+=['','## What augmentation actually added','',
            f"Ten rotating draws exposed {coverage[-1]['distinct_training_pairs_so_far']:,} distinct training "
            f"query/product pairs versus {coverage[0]['distinct_training_pairs_so_far']:,} in the fixed draw. "
            'Every epoch preserves all45,225 eligible positive instances and the full-pool importance weighting. '
            'All products are eligible at their original query date. Validation candidates, inputs and targets '
            'remain unchanged. This broadens negative comparisons, not independent positive purchase evidence.',
            '', 'Arbitrary fake purchases, SMOTE/mixup over SKU IDs, or date jitter would alter history, eligibility, '
            'quantities and derived statistics inconsistently. Existing query transitions already supply historical '
            'prefix examples. History masking/cropping could be tested as an explicitly noisy training view with '
            'a coherent statistics policy, but it adds regularization and was not combined with this screen. '
            'No future validation/held-out target is used as an augmented training positive.','',
            '## Interpretation and limits','',
            'Regularization is model-dependent: reducing it helped the small statistics-only MLP; it does not '
            'follow that every ranker stage should use zero dropout. Weight decay was already modest at1e-4. '
            'Compare each Stage2 treatment with both its fresh control and saved A, including all ranking metrics '
            'and paired uncertainty, rather than choosing a single discovery maximum. A 128-query memorization '
            'probe from the preceding audit verified small-cohort capacity, not optimal full-data generalization.',
            '', 'This remains a single-seed screen with repeated validation use and fixed ten-epoch schedules. '
            'No model is promoted. Catalogue metadata are snapshot_unverified and upstream inference snapshot '
            'coherence remains a separate production prerequisite. Standalone models use common statistical '
            'containers but discard neural fields; they have no neural inference dependency.', '',
            '## Verification and reproduction','',
            f"Independent replay covers {len(rv['selected_records'])} selected ranker and {len(tv['selected_records'])} "
            'selected standalone MLP records. Full-denominator saved metrics, independent CPU logits, '
            'augmentation draw digests/positive counts and frozen artifact hashes are checked. Ranker training '
            'also verifies fitted permutation/padding stability. Full safe/synthetic suite results are in '
            '[checks.json](checks.json). No held-out data was accessed.','',
            '```bash',
            './venv/bin/python -m recommender.experiments.regularization_signal.train_ranker --execute-training',
            './venv/bin/python -m recommender.experiments.regularization_signal.train_tabular --execute-training --device cpu',
            './venv/bin/python -m recommender.experiments.regularization_signal.verify --stage tabular',
            './venv/bin/python -m recommender.experiments.regularization_signal.verify --stage ranker',
            './venv/bin/python -m recommender.experiments.regularization_signal.report',
            'PYTHONPATH=. ./venv/bin/pytest tests/ --synthetic-only -q',
            'PYTHONPATH=. ./venv/bin/pytest tests/ -q', '```','',
            'Training and verification refuse overwriting completed evidence. Reproduction requires an explicitly '
            'separate run namespace. Source code is isolated under recommender/experiments/regularization_signal/.']
    content='\n'.join(lines)+'\n'
    spacing={'Stage1':'Stage 1','Stage2':'Stage 2','Seed42':'Seed 42','seed42':'seed 42',
             'CatBoost52':'CatBoost 52','Direct52':'Direct 52','to0.':'to 0.','then0.':'then 0.',
             'has0.':'has 0.','reference0.':'reference 0.','reaches258':'reaches 258','versus255':'versus 255',
             'versus202':'versus 202','approximately25':'approximately 25','original17':'original 17',
             'original32':'original 32','head81':'head 81','head256':'head 256','on4,096':'on 4,096',
             'sample:2,':'sample: 2,','including45,':'including 45,','representing15,':'representing 15,',
             'weight1;':'weight 1;','all4,700':'all 4,700','all45,225':'all 45,225','lr0.001':'lr 0.001',
             'to1e-5':'to 1e-5','batch4096':'batch 4096','batch64':'batch 64','CPU2':'CPU 2',
             'bootstrap95%':'bootstrap 95%','guardrails:Recall':'guardrails: Recall','at1e-4':'at 1e-4',
             'top10':'top-10'}
    for before,after in spacing.items():content=content.replace(before,after)
    (EVIDENCE/'REPORT.md').write_text(content)
    figure(ranker,tab)


def figure(ranker,tab):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    colors=['#176b98','#cf7632','#218b6e','#a34f97','#687582']
    for name,color in zip(('control','head_half','attention_half','all_zero','direct52_all_zero'),colors):
        history=ranker['arms'][name]['history']
        axes[0].plot([r['epoch'] for r in history],[r['metrics']['ndcg@10'] for r in history],
                     color=color,label={'control':'Control','head_half':'Half head dropout','attention_half':'Half attention dropout',
                                        'all_zero':'Zero active regularization','direct52_all_zero':'Direct52 + zero regularization'}[name])
    for (name,arm),color in zip(tab['arms'].items(),colors):
        axes[1].plot([r['epoch'] for r in arm['history']],[r['metrics']['ndcg@10'] for r in arm['history']],
                     color=color,label={'mlp_fixed_regularized':'Fixed pairs, regularized',
                                        'mlp_fixed_unregularized':'Fixed pairs, zero regularization',
                                        'mlp_rotating_negatives':'Rotating negatives, zero regularization'}[name])
    axes[1].axhline(tab['catboost_baseline']['ndcg']['metrics']['ndcg@10'],ls='--',color='#687582',lw=1,label='Frozen CatBoost52')
    axes[0].axhline(ranker['baseline']['ndcg']['metrics']['ndcg@10'],ls=':',color='#687582',lw=1,label='Saved rolling reference')
    axes[0].set_title('Stage 2: regularization by component',fontsize=10)
    axes[1].set_title('Small MLP: fixed inputs, varied training',fontsize=10)
    for ax in axes:
        ax.set(xlabel='Epoch',ylabel='Validation NDCG@10',xticks=[1,3,5,7,10])
        ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15)
        ax.legend(fontsize=6.8,loc='lower right',frameon=False)
    fig.savefig(EVIDENCE/'regularization-comparison.png',dpi=160,facecolor='white');plt.close(fig)
    encoded=base64.b64encode((EVIDENCE/'regularization-comparison.png').read_bytes()).decode()
    (EVIDENCE/'regularization-comparison.html').write_text(
        '<!DOCTYPE html><html><head><meta charset="utf-8"></head><body><div style="color:var(--foreground);font-size:13px">'
        f'<img style="display:block;width:100%;border-radius:var(--radius)" alt="Validation ranking across regularization and negative-resampling experiments" src="data:image/png;base64,{encoded}">'
        '<p>Seed 42 · validation only · ten epochs. Stage 2 shares identical 100-item slates; standalone MLPs score the full eligible pool. '
        'CatBoost inputs and saved model stay fixed.</p></div></body></html>')


if __name__=='__main__':build()

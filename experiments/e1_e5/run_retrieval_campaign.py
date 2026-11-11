"""Run after E1/E2's 12 fits finish; screen all retrieval variants then repeat leads.

PYTHONPATH=. ./venv/bin/python experiments/e1_e5/run_retrieval_campaign.py
"""
import json
from pathlib import Path
import time
import torch
from recommender.experiment_retrieval import CONFIGS, OUTPUT, train_retrieval, downstream
from recommender.experiment_retrieval_data import prepare_retrieval_data


def main():
    # Serialize accelerator work with the signal campaign for useful telemetry.
    signal_comparison = Path('checkpoints/e1-e5/signals-v1/comparison.json')
    while True:
        if signal_comparison.exists() and len(json.loads(signal_comparison.read_text())['results']) == 12:
            break
        time.sleep(5)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    train, val, info, ts, vs = prepare_retrieval_data()
    screens = {}
    for name in CONFIGS:
        stage1 = train_retrieval(name, 42, train, val, info, ts, vs, OUTPUT, 30, device)
        stage2 = downstream(name, 42, train, val, info, ts, vs, OUTPUT, device)
        screens[name] = {'stage1': stage1, 'stage2': stage2}
    base = screens['S0']
    promising = []
    for name, result in screens.items():
        if name == 'S0': continue
        stage1_gain = max(result['stage1']['metrics'][key] - base['stage1']['metrics'][key]
                          for key in ('new_hits@25', 'new_hits@50'))
        stage2_gain = result['stage2']['metrics']['new_hits@10'] - base['stage2']['metrics']['new_hits@10']
        if stage1_gain >= 10 or (stage2_gain >= 5 and result['stage2'].get('qualified_recall_guardrail', True)):
            promising.append(name)
    repeated = {'S0'} if promising else set()
    repeated.update(promising)
    if any(name.startswith('E3') for name in promising):
        repeated.update(('E3mean', 'E3attention'))
    if any(name.startswith(('E4', 'E5')) for name in promising):
        repeated.add('E4k1')
        for name in promising:
            if name.startswith(('E4', 'E5')) and name[-1] in '23':
                repeated.update((f'E4k{name[-1]}', f'E5k{name[-1]}'))
    decision = {'screening_seed': 42, 'promising': promising, 'repeated': sorted(repeated),
                'rule': '>=10 additional Stage1 unseen hits@25 or @50, OR >=5 downstream new hits@10 with fixed recall guardrail; repeat paired controls; no promotion'}
    Path('experiments/e1_e5/repetition_selection.json').write_text(json.dumps(decision, indent=2)+'\n')
    for seed in (17, 2026):
        for name in CONFIGS:
            if name in repeated:
                train_retrieval(name, seed, train, val, info, ts, vs, OUTPUT, 30, device)
                downstream(name, seed, train, val, info, ts, vs, OUTPUT, device)
    Path('experiments/e1_e5/campaign_complete.json').write_text(json.dumps(decision, indent=2)+'\n')


if __name__ == '__main__':
    main()

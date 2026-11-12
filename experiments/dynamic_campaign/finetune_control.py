"""Pretrained residual proposal: matched extra budget and clean initial ranker."""
from recommender.experiments.dynamic_selector.train import fit_ranker,evaluated_checkpoint
from experiments.dynamic_campaign.run import promising_ranker


def run(results,train,val,manifest,ts,vs,root,device):
    # Also enforce the predeclared cheap-model noninferiority check when this
    # module is entered by a campaign process that began before the supplement.
    pointwise=results['rankers'].get('production50-pointwise-42')
    reference=results['rankers']['production50-control-42']['metrics']
    low_cost=False
    if pointwise is not None:
        metric=pointwise['metrics']
        low_cost=(pointwise['qualified'] and metric['new_hits@10']>=reference['new_hits@10']-5
                  and metric['recall@10']>=reference['recall@10']-.001 and metric['ndcg@10']>=reference['ndcg@10']-.001)
        if low_cost:
            for seed in (17,2026):
                key=f'production50-pointwise-{seed}'
                if key not in results['rankers']:
                    results['rankers'][key]=fit_ranker('pointwise',seed,ts,vs,train,val,manifest,root,device,'production50',10)
    def fit(mode,seed):
        initial=evaluated_checkpoint(results['rankers'][f'production50-control-{seed}'])
        result=fit_ranker(mode,seed,ts,vs,train,val,manifest,root,device,'finetune50',10,
                          initial_control_checkpoint=initial,learning_rate=.0001)
        results['rankers'][f'finetune50-{mode}-{seed}']=result
        return result
    control=fit('control',42)
    residual=fit('context_residual',42)
    promising=promising_ranker(residual,control)
    budget_promising=promising_ranker(control,results['rankers']['production50-control-42'])
    if promising or budget_promising:
        for seed in (17,2026):
            fit('control',seed)
            if promising:fit('context_residual',seed)
    return {'initialization':'same clean matched selected ranker, never legacy learned weights',
            'budget':'matched additional 10 epochs at learning rate 0.0001',
            'promising_repeated':promising,'budget_effect_repeated':budget_promising,
            'pointwise_low_cost_repeated':low_cost,
            'checkpoint_selection':'unchanged simultaneous quality floors'}

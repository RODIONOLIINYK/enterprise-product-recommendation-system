"""Fixed-weight E4/E5 scorer/head perturbations using purchase caches only."""
from dataclasses import replace
import json
from pathlib import Path
import torch
from recommender.experiment_retrieval import OUTPUT, CONFIGS, make_model, metrics_from_scores
from recommender.experiment_retrieval_data import prepare_retrieval_data


def main():
    torch.set_num_threads(2)
    _,val,info,_,source=prepare_retrieval_data()
    results={}
    with torch.no_grad():
        for path in sorted(OUTPUT.glob('*/result.json')):
            result=json.loads(path.read_text()); settings=result['settings']; name=settings['run']
            if not name.startswith(('E4','E5')) or CONFIGS[name].interests<2: continue
            run_dir=path.parent
            checkpoint=torch.load(run_dir/'best.pt',map_location='cpu')
            cache=torch.load(run_dir/'val_purchase_cache.pt',map_location='cpu')
            model=make_model(CONFIGS[name],info).eval()
            model.load_state_dict(checkpoint['model_state_dict'],strict=True)
            u,v=cache['customer_vectors'],cache['product_vectors']
            indices=torch.arange(len(v)); flags=val['familiarity']
            original_cpu=metrics_from_scores(model.score_cached(u,v,indices,flags),val,source)
            perturbations={}
            for label, vectors, scoring in (
                ('first_interest_only',u[:,:1],CONFIGS[name].scoring),
                ('alternate_aggregation',u,'smooth' if CONFIGS[name].scoring=='hard' else 'hard')):
                model.config=replace(CONFIGS[name],scoring=scoring)
                scores=model.score_cached(vectors,v,indices,flags)
                perturbations[label]=metrics_from_scores(scores,val,source)
            query=checkpoint['model_state_dict']['interest_queries']
            normalized=torch.nn.functional.normalize(query,dim=-1)
            perturbations['query_norms']=query.norm(dim=-1).tolist()
            perturbations['query_parameter_cosines']=(normalized@normalized.T).tolist()
            perturbations['original']=original_cpu
            perturbations['recorded_selected_metrics']=result['metrics']
            perturbations['evaluation_device']='cpu'
            perturbations['kind']='Fixed weights, cache-only perturbation; not a retrained control'
            results[run_dir.name]=perturbations
    Path('experiments/e1_e5/scoring_ablations.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__': main()

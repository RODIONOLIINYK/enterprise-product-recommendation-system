"""Read-only replay of audited reference weights on development caches."""
import json
from pathlib import Path
import torch
from recommender.experiment_signals import load_inputs, make_model
from recommender.experiment_retrieval_data import CACHE
from recommender.experiment_retrieval import metrics_from_scores
from recommender.shared_data import DEFAULT_RETRIEVER
from recommender.train_shared_ranker import make_loader, diagnostics
from recommender.train_ranker import evaluate_ranker, checkpoint_digest
from recommender.ranker import RankerLoss


def main():
    torch.set_num_threads(2)
    _,source,manifest=load_inputs()
    val=torch.load(CACHE/'val.pt',map_location='cpu')
    info=torch.load(CACHE/'manifest.pt',map_location='cpu')
    checkpoint=torch.load(DEFAULT_RETRIEVER,map_location='cpu')
    state=checkpoint['model_state_dict']
    vectors=manifest['product_vectors'][info['expanded_indices']]
    u=source['stage1_customer_vectors']
    bias=torch.nn.functional.pad(state['product_bias'],(0,len(vectors)-len(state['product_bias'])))
    with torch.no_grad():
        scores=u@vectors.T/checkpoint['config']['temperature']+bias
        scores+=val['familiarity'].float()@state['familiarity_weights']
    result={'stage1':{'path':str(DEFAULT_RETRIEVER),'sha256':checkpoint_digest(DEFAULT_RETRIEVER),
                      'metrics':metrics_from_scores(scores,val,source),'customer_vectors':'exact frozen reference purchase snapshots'},'stage2':{}}
    for name in ('best_ranker','best_discovery_ranker','best_ndcg_ranker'):
        path=Path('checkpoints/ranker')/f'{name}.pt'
        ckpt=torch.load(path,map_location='cpu')
        model=make_model('control').base
        model.load_state_dict(ckpt['model_state_dict'],strict=True)
        metrics=evaluate_ranker(model,make_loader(source),manifest['product_vectors'],torch.device('cpu'),RankerLoss())
        result['stage2'][name]={'path':str(path),'sha256':checkpoint_digest(path),'epoch':ckpt['epoch'],'metrics':metrics,
                              'diagnostics':diagnostics(model,source,manifest['product_vectors'],torch.device('cpu'))}
    Path('experiments/e1_e5/reference_replay.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'stage1':result['stage1']['metrics'],'stage2':{k:v['metrics']['new_hits@10'] for k,v in result['stage2'].items()}},indent=2))


if __name__=='__main__': main()

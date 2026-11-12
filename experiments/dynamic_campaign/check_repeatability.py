"""Synthetic MPS update reproducibility; no purchase inputs or model training fits.

Reference metadata and declared clean upstream projections construct the model.
Prints measurements; --output can write a separately chosen new record.
"""
import argparse
import json
from pathlib import Path
import torch
from recommender.artifacts import atomic_json
from recommender.experiments.dynamic_selector.train import make_ranker
from recommender.ranker import RankerLoss


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path)
    args=parser.parse_args();torch.set_num_threads(2)
    if not torch.backends.mps.is_available():print('MPS unavailable; this diagnostic is platform-specific.');return
    torch.manual_seed(9);n=64
    x=dict(text_embedding=torch.randn(n,1024),city_id=torch.randint(0,3,(n,)),region_id=torch.randint(0,3,(n,)),
        industry_id=torch.randint(0,3,(n,)),tabular_features=torch.rand(n,8),sequence=torch.rand(n,32,9),
        attention_mask=torch.ones(n,32,dtype=torch.long),seq_purchase_dates=torch.arange(32).expand(n,-1).float()/365.25,
        prediction_day=torch.full((n,),100.),candidate_vectors=torch.randn(n,100,128),pairwise_features=torch.rand(n,100,41),
        cached_customer_vectors=torch.randn(n,128),seq_product_ids=torch.randint(1,30,(n,32)),
        seq_category_ids=torch.randint(0,3,(n,32)),seq_subcategory_ids=torch.randint(0,3,(n,32)),seq_business_line_ids=torch.randint(0,3,(n,32)))
    labels=torch.zeros(n,100,device='mps');labels[:,0]=1;runs=[]
    for _ in range(2):
        model=make_ranker('control',42).to('mps').train();inputs={k:v.to('mps') for k,v in x.items()}
        initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};rng=torch.mps.get_rng_state().clone();outputs=[]
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001)
        for _ in range(5):
            optimizer.zero_grad(set_to_none=True);out=model(**inputs).logits
            RankerLoss()(out,labels)['loss'].backward();optimizer.step();torch.mps.synchronize();outputs.append(out.detach().cpu())
        runs.append((initial,rng,outputs,{k:v.detach().cpu().clone() for k,v in model.state_dict().items()}))
    a,b=runs
    result={'synthetic_only':True,'device':'mps','deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
        'initial_weights_equal':all(torch.equal(a[0][k],b[0][k]) for k in a[0]),'initial_mps_rng_equal':torch.equal(a[1],b[1]),
        'batch':64,'candidates':100,'history':32,'updates':5,
        'train_logit_max_abs_differences':[float((x-y).abs().max()) for x,y in zip(a[2],b[2])],
        'after_five_updates_parameter_max_abs_difference':max(float((a[3][k]-b[3][k]).abs().max()) for k in a[3]),
        'note':'Saved checkpoint replay is checked separately; seeded MPS training is not claimed bitwise deterministic.'}
    print(json.dumps(result,indent=2))
    if args.output:atomic_json(result,args.output,overwrite=False)


if __name__=='__main__':main()

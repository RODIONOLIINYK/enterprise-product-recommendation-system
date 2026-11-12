"""Matched training loops with explicit manifests, no legacy artifact writes."""
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from recommender.artifacts import atomic_json, create_run_directory, file_digest, json_digest
from recommender.experiment_signals import sync
from recommender.experiments.candidate_context.model import CandidateContextRanker, CachedPointwiseRanker
from recommender.ranker import RankerLoss, UPSCANRRanker
from recommender.shared_data import DEFAULT_RETRIEVER, atomic_torch_save
from recommender.train_shared_ranker import make_loader
from recommender.train_ranker import ranker_batch_inputs
from .data import UPSTREAM
from .evaluate import ranking_metrics
from .feature_schema import contract, schema_digest
from .model import PointwiseSelector


def run_directory(root, name, settings):
    directory = Path(root) / 'runs' / name
    if directory.exists():
        manifest = json.loads((directory / 'run.json').read_text())
        if manifest.get('settings_digest') != json_digest(settings): raise ValueError('Run settings differ')
        if manifest['status'] != 'complete': raise ValueError('Incomplete run: choose an explicitly new run ID')
        checkpoint=manifest.get('selected_checkpoint') or manifest.get('diagnostic_checkpoint')
        digest=manifest.get('selected_sha256') or manifest.get('diagnostic_sha256')
        if not checkpoint or file_digest(directory / checkpoint) != digest: raise ValueError('Checkpoint checksum')
        return directory, json.loads((directory / 'result.json').read_text())
    directory = create_run_directory(root, name)
    atomic_json({'version':1,'status':'running','settings_digest':json_digest(settings), 'selected_checkpoint':None}, directory / 'run.json')
    return directory, None


def finish_run(directory, settings, result):
    qualified=result.get('qualified',True)
    checkpoint='best.pt'
    if not qualified:
        checkpoint='best_unqualified.pt'
        (directory/'best.pt').replace(directory/checkpoint)
        result['selected_checkpoint']=None
        result['diagnostic_checkpoint']=str(directory/checkpoint)
    atomic_json(result, directory / 'result.json', overwrite=False)
    atomic_json({'version':1,'status':'complete','settings_digest':json_digest(settings),
                 'selected_checkpoint':checkpoint if qualified else None,
                 'selected_sha256':file_digest(directory/checkpoint) if qualified else None,
                 'diagnostic_checkpoint':checkpoint if not qualified else None,
                 'diagnostic_sha256':file_digest(directory/checkpoint) if not qualified else None}, directory / 'run.json')


def evaluated_checkpoint(result):
    """Explicitly include failed-guardrail diagnostics without publishing an alias."""
    return Path(result.get('selected_checkpoint') or result['diagnostic_checkpoint'])


def selector_inputs(data, rows, vectors, device):
    return (data['u'][rows].to(device), vectors,
            torch.from_numpy(np.array(data['features'][rows.numpy()], dtype=np.float32)).to(device),
            data['dot_scores'][rows].to(device))


@torch.no_grad()
def predict_selector(model, data, device):
    model.eval(); vectors=data['v'].to(device)
    return torch.cat([model(*selector_inputs(data, rows, vectors, device)).cpu()
                      for rows in torch.arange(len(data['u'])).split(128)])


def fit_selector(mode, seed, train, val, manifest, root, device, epochs=10):
    settings={'stage':'selector','mode':mode,'seed':seed,'epochs':epochs,'batch_size':64,'lr':.001,
              'identity':manifest['identity'],'schema_digest':schema_digest(),'loss':'eligible BCE pos_weight20',
              'selection':'new_hits@25 then @50','initialization':'scratch','upstream_digest':manifest['upstream_digest']}
    directory, existing=run_directory(root,f'selector-{mode}-seed{seed}',settings)
    if existing: return existing
    torch.manual_seed(seed);np.random.seed(seed)
    model=PointwiseSelector(mode).to(device);vectors=train['v'].to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs,eta_min=1e-5)
    generator=torch.Generator().manual_seed(seed)
    best=(-1,-1);history=[];sync(device);started=time.perf_counter()
    for epoch in range(1,epochs+1):
        model.train();losses=[]
        for rows in torch.randperm(len(train['u']),generator=generator).split(64):
            optimizer.zero_grad(set_to_none=True)
            scores=model(*selector_inputs(train,rows,vectors,device))
            allowed=train['allowed'][rows].to(device);targets=train['target'][rows].to(device).float()
            loss=F.binary_cross_entropy_with_logits(scores[allowed],targets[allowed],pos_weight=torch.tensor(20.,device=device))
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        scores=predict_selector(model,val,device);metrics=ranking_metrics(scores,val)
        key=(metrics['new_hits@25'],metrics['new_hits@50'])
        history.append({'epoch':epoch,'train_loss':float(np.mean(losses)),'metrics':metrics})
        if key>best:
            best=key
            atomic_torch_save({'settings':settings,'feature_contract':contract(),'model_state_dict':model.state_dict(),
                               'epoch':epoch,'metrics':metrics},directory/'best.pt')
        print(f"selector {mode} seed={seed} epoch={epoch} new25={key[0]} new50={key[1]}",flush=True)
    sync(device);elapsed=time.perf_counter()-started
    checkpoint=torch.load(directory/'best.pt',map_location='cpu');model.load_state_dict(checkpoint['model_state_dict'])
    metrics=ranking_metrics(predict_selector(model,val,device),val,save=directory/'predictions.npz')
    assert metrics==checkpoint['metrics']
    for split,data in (('train',train),('val',val)):
        atomic_torch_save({'scores':predict_selector(model,data,device),'model_digest':file_digest(directory/'best.pt'),
                           'identity':manifest['identity']},directory/f'{split}_scores.pt')
    result={'settings':settings,'metrics':metrics,'epoch':checkpoint['epoch'],'training_seconds':elapsed,
            'parameters':sum(p.numel() for p in model.parameters()),'selected_checkpoint':str(directory/'best.pt')}
    # Post-training feature-group ablation, not a separately optimized replacement.
    ablations={}
    original=model.feature_mask.clone()
    for name,cols in (('no_dynamic',range(13,33)),('no_relationships',range(33,41)),('no_embeddings',None)):
        if cols is None: continue  # Embedding-free is a different model, not zero-mask calibration.
        model.feature_mask.copy_(original);model.feature_mask[list(cols)]=0
        ablations[name]=ranking_metrics(predict_selector(model,val,device),val)
    model.feature_mask.copy_(original)
    result['ablations']=ablations
    atomic_json(history,directory/'history.json',overwrite=False);finish_run(directory,settings,result)
    return result


def make_ranker(mode, seed):
    torch.manual_seed(seed)
    if mode=='pointwise':return CachedPointwiseRanker()
    base=UPSCANRRanker.from_retriever_checkpoint(DEFAULT_RETRIEVER,statistics_attention=True,excluded_statistics=('history_repeat',),cached_only=True)
    # Use no learned legacy tower/ranker weights. Copy matching projections only
    # from the declared demand-free upstream after resetting all learned branches.
    def reset(module):
        if isinstance(module,torch.nn.MultiheadAttention): module._reset_parameters()
        elif hasattr(module,'reset_parameters'): module.reset_parameters()
    base.apply(reset)
    state=torch.load(UPSTREAM/'best.pt',map_location='cpu')['model_state_dict']
    with torch.no_grad():
        for name in ('text_proj.0.weight','text_proj.0.bias','city_embedding.weight','region_embedding.weight','industry_embedding.weight'):
            dict(base.named_parameters())[name].copy_(state['towers.customer_tower.'+name])
        from recommender.features import LEGACY_PROFILE_FEATURES
        config=torch.load(DEFAULT_RETRIEVER,map_location='cpu')['config']
        target=base.tabular_proj[0].weight;source=state['towers.customer_tower.tabular_proj.0.weight']
        target[:,:48].copy_(source[:,:48])
        for i,name in enumerate(config['profile_features']):
            if name in LEGACY_PROFILE_FEATURES: target[:,48+LEGACY_PROFILE_FEATURES.index(name)].copy_(source[:,48+i])
        base.tabular_proj[0].bias.copy_(state['towers.customer_tower.tabular_proj.0.bias'])
    # Additional modules must not change shared initialization.
    with torch.random.fork_rng(devices=[]):
        return CandidateContextRanker(base,mode)


def context_inputs(batch, vectors, model, device):
    inputs=ranker_batch_inputs(batch,vectors,model,device)
    inputs['cached_customer_vectors']=batch['stage1_customer_vectors'].to(device)
    return inputs


@torch.no_grad()
def predict_ranker(model, slate, vectors, device, **options):
    model.eval()
    return torch.cat([model(**context_inputs(batch,vectors,model,device),**options).logits.cpu() for batch in make_loader(slate)])


def fit_ranker(mode, seed, train_slate, val_slate, train, val, manifest, root, device, name, epochs=10,
               initial_control_checkpoint=None, learning_rate=.001):
    reference=torch.load('checkpoints/ranker/best_discovery_ranker.pt',map_location='cpu')
    recall_guard=reference['discovery_min_overall_recall']
    ndcg_guard=reference['val_metrics']['val_ndcg@10']
    settings={'stage':'ranker','mode':mode,'seed':seed,'epochs':epochs,'batch_size':64,'lr':learning_rate,
              'identity':manifest['identity'],'feature_contract':contract(),'upstream_digest':manifest['upstream_digest'],
              'train_slate_identity':train_slate['cache_identity'],'val_slate_identity':val_slate['cache_identity'],
              'recall_guardrail':recall_guard,'ndcg_guardrail':ndcg_guard,
              'guardrail_artifact_sha256':file_digest('checkpoints/ranker/best_discovery_ranker.pt'),
              'selection':'new_hits@10 then NDCG subject to both reference quality floors; unqualified diagnostics have no selected alias'}
    if initial_control_checkpoint is not None:
        settings['initial_control_digest']=file_digest(initial_control_checkpoint)
        settings['effective_budget']='10 fresh epochs + matched 10 fine-tuning epochs'
    directory,existing=run_directory(root,f'{name}-{mode}-seed{seed}',settings)
    if existing: return existing
    torch.manual_seed(seed);np.random.seed(seed)
    model=make_ranker(mode,seed).to(device);vectors=train['v'].to(device)
    initial_parity=None
    if initial_control_checkpoint is not None:
        initial=torch.load(initial_control_checkpoint,map_location='cpu')
        if initial['settings']['mode']!='control' or initial['settings']['identity']!=manifest['identity']:
            raise ValueError('Fine-tuning requires matching clean control provenance')
        state=initial['model_state_dict']
        if any(not key.startswith('base.') for key in state):raise ValueError('Unexpected control state keys')
        model.base.load_state_dict({key[len('base.'):]:value for key,value in state.items()},strict=True)
        if mode=='context_residual':
            model.eval()
            batch=next(iter(make_loader(val_slate)))
            supplied=context_inputs(batch,vectors,model,device)
            plain={key:value for key,value in supplied.items() if key!='cached_customer_vectors'}
            with torch.no_grad():
                expected=model.base(**plain).logits
                actual=model(**supplied).logits
            torch.testing.assert_close(actual,expected,atol=0,rtol=0)
            initial_parity=float((actual-expected).abs().max())
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=learning_rate,weight_decay=.0001)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs,eta_min=1e-5);loss_fn=RankerLoss().to(device)
    loader=make_loader(train_slate,shuffle=True,seed=seed)
    best=(-1,-1.,-1);history=[];sync(device);started=time.perf_counter()
    for epoch in range(1,epochs+1):
        model.train();losses=[]
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            logits=model(**context_inputs(batch,vectors,model,device)).logits
            loss=loss_fn(logits,batch['labels'].to(device))['loss']
            if mode=='group_gate': loss=loss+.001*model.repeat_gate.sigmoid()
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();losses.append(float(loss.detach()))
        scheduler.step()
        metrics=ranking_metrics(predict_ranker(model,val_slate,vectors,device),val,val_slate['candidate_indices'])
        key=(int(metrics['recall@10']>=recall_guard and metrics['ndcg@10']>=ndcg_guard),metrics['new_hits@10'],metrics['ndcg@10'])
        history.append({'epoch':epoch,'train_loss':float(np.mean(losses)),'metrics':metrics})
        if key>best:
            best=key
            atomic_torch_save({'settings':settings,'epoch':epoch,'metrics':metrics,'model_state_dict':model.state_dict()},directory/'best.pt')
        print(f"ranker {name}/{mode} seed={seed} epoch={epoch} new={metrics['new_hits@10']} rec={metrics['recall@10']:.5f} ndcg={metrics['ndcg@10']:.5f}",flush=True)
    sync(device);elapsed=time.perf_counter()-started
    checkpoint=torch.load(directory/'best.pt',map_location='cpu');model.load_state_dict(checkpoint['model_state_dict'],strict=True)
    scores=predict_ranker(model,val_slate,vectors,device)
    metrics=ranking_metrics(scores,val,val_slate['candidate_indices'],directory/'predictions.npz');assert metrics==checkpoint['metrics']
    result={'settings':settings,'epoch':checkpoint['epoch'],'metrics':metrics,'qualified':bool(best[0]),
            'parameters':sum(p.numel() for p in model.parameters()),'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'training_seconds':elapsed,'selected_checkpoint':str(directory/'best.pt')}
    if initial_parity is not None:result['initial_pretrained_logit_max_abs_difference']=initial_parity
    if mode=='group_gate': result['repeat_gate']=float(model.repeat_gate.sigmoid().detach())
    if mode in ('recent_sku','hierarchy','complements','dynamic_features'):
        ablated=dict(val_slate);ablated['pairwise_features']=val_slate['pairwise_features'].clone();ablated['pairwise_features'][...,13:]=0
        result['ablation_no_added_signal']=ranking_metrics(predict_ranker(model,ablated,vectors,device),val,val_slate['candidate_indices'])
    atomic_json(history,directory/'history.json',overwrite=False);finish_run(directory,settings,result)
    return result

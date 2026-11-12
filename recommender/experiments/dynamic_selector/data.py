"""Versioned development-only, full-eligible-pool data and cached tower inputs."""
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from recommender.artifacts import atomic_json, file_digest, json_digest
from recommender.catalogue import load_product_catalogue
from recommender.company import load_companies
from recommender.development import load_development_purchases
from recommender.experiment_retrieval_data import prepare_retrieval_data
from recommender.scope import get_datasets_locs
from recommender.shared_data import atomic_torch_save
from recommender.split import TRAIN_CUTOFF, TEST_START
from .feature_builder import DynamicFeatureBuilder
from .feature_schema import NAMES, contract, schema_digest

CACHE = Path('data/interim/dynamic_campaign_v1')
UPSTREAM = Path('checkpoints/e1-e5/retrieval-v1/S0-seed42')


def sliced(data, rows, n):
    return {key: value[rows] if isinstance(value, torch.Tensor) and value.ndim and len(value) == n else
            [value[i] for i in rows.tolist()] if isinstance(value, list) and len(value) == n else value
            for key, value in data.items()}


def load_campaign_data(cache=CACHE, build=True):
    cache = Path(cache)
    train, val, info, ts, vs = prepare_retrieval_data()
    sources, retrieval = [], []
    for source, data in ((ts, train), (vs, val)):
        rows = torch.where(data['allowed'].sum(1) >= 100)[0]
        sources.append(sliced(source, rows, len(source['customer_ids'])))
        retrieval.append(sliced(data, rows, len(source['customer_ids'])))
    upstream_digest = file_digest(UPSTREAM / 'best.pt')
    identity = json_digest({'feature_schema': schema_digest(), 'upstream': upstream_digest,
                            'source': info['identity'], 'train_rows': sources[0]['transition_keys'],
                            'val_rows': sources[1]['transition_keys'], 'minimum_eligible': 100})
    cache.mkdir(parents=True, exist_ok=True)
    manifest_path = cache / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest['identity'] != identity or manifest['feature_contract'] != contract():
            raise ValueError('Campaign cache schema/origin mismatch')
        for split in ('train', 'val'):
            if file_digest(cache / f'{split}_features.npy') != manifest['feature_sha256'][split]:
                raise ValueError('Campaign feature checksum mismatch')
    else:
        if not build: raise FileNotFoundError('Run feature preparation first')
        if any(cache.glob('*features.npy')): raise ValueError('Incomplete campaign cache; use a new namespace')
        purchases = load_development_purchases('data/interim/cleaned_purchases.csv')
        catalogue = load_product_catalogue(get_datasets_locs('items'))
        companies = load_companies(purchases=purchases, minimum_purchases=1)
        builder = DynamicFeatureBuilder(purchases, catalogue, companies, info['product_ids'])
        arrays = [np.lib.format.open_memmap(cache / f'{split}_features.npy', mode='w+', dtype=np.float16,
                   shape=(len(source['customer_ids']), len(info['product_ids']), len(NAMES)))
                  for split, source in zip(('train', 'val'), sources)]
        queries = sorted((int(day), split, row) for split, source in enumerate(sources)
                         for row, day in enumerate(source['prediction_days'].tolist()))
        started = time.perf_counter()
        for i, (day, split, row) in enumerate(queries):
            date = pd.Timestamp('2025-01-01') + pd.Timedelta(days=day)
            arrays[split][row] = builder.build(sources[split]['customer_ids'][row], date)
            if i % 1000 == 0:
                print(f'features {i}/{len(queries)} seconds={time.perf_counter()-started:.1f}', flush=True)
        for array in arrays: array.flush()
        manifest = {'version': 1, 'identity': identity, 'feature_contract': contract(),
                    'upstream_digest': upstream_digest, 'source_identity': info['identity'],
                    'feature_sha256': {split: file_digest(cache / f'{split}_features.npy') for split in ('train', 'val')},
                    'train_rows': len(sources[0]['customer_ids']), 'val_rows': len(sources[1]['customer_ids']),
                    'preparation_seconds': time.perf_counter() - started, 'test_access': False,
                    'metadata_policy': 'snapshot_unverified', 'minimum_eligible': 100,
                    'product_ids': info['product_ids']}
        atomic_json(manifest, manifest_path, overwrite=False)
    outputs = []
    # Diagnostic-only availability; never treat missing family metadata as a
    # demonstrated first-time family purchase in cohort reporting.
    diagnostic_catalogue=load_product_catalogue(get_datasets_locs('items')).set_index('product_id')
    family_values=diagnostic_catalogue['product_family'].reindex(info['product_ids']).fillna('').astype(str)
    family_known=torch.tensor(family_values.str.strip().ne('').to_numpy(dtype=bool,copy=True))
    for split, source, data, original in zip(('train', 'val'), sources, retrieval, (ts, vs)):
        payload = torch.load(UPSTREAM / f'{split}_purchase_cache.pt', map_location='cpu')
        if payload['model_digest'] != upstream_digest or payload['data_identity'] != info['identity']:
            raise ValueError('Upstream purchase cache provenance mismatch')
        rows = torch.where((train if split == 'train' else val)['allowed'].sum(1) >= 100)[0]
        data.update(source=source, u=payload['customer_vectors'][rows].squeeze(1),
                    v=payload['product_vectors'], dot_scores=payload['scores'][rows],
                    features=np.load(cache / f'{split}_features.npy', mmap_mode='r'))
        data['family_known']=family_known
        days = source['prediction_days']
        if bool((days >= (TEST_START - pd.Timestamp('2025-01-01')).days).any()): raise ValueError('Held-out rows')
        if split == 'train' and bool((days >= (TRAIN_CUTOFF - pd.Timestamp('2025-01-01')).days).any()): raise ValueError('Train dates')
        if data['u'].shape != (len(days), 128) or data['features'].shape != (len(days), len(info['product_ids']), len(NAMES)):
            raise ValueError('Campaign cache row/feature dimensions')
        outputs.append(data)
    return *outputs, info, manifest


def make_slates(data, scores, product_ids, policy):
    from .model import select_candidates
    indices = select_candidates(scores.cpu(), data['allowed'], data['unseen'], policy)
    source = dict(data['source'])
    source['candidate_indices'] = indices
    source['candidate_vectors'] = None
    source['stage1_customer_vectors'] = data['u']
    source['candidate_pids'] = [[product_ids[i] for i in row] for row in indices.tolist()]
    source['labels'] = data['target'].gather(1, indices).float()
    rows = np.arange(len(indices))[:, None]
    source['pairwise_features'] = torch.from_numpy(np.array(data['features'][rows, indices.numpy()], dtype=np.float32))
    source['cache_identity'] = json_digest({'policy': policy, 'indices': indices.tolist(), 'features': schema_digest()})
    return source


def save_slates(data, path):
    path = Path(path)
    if path.exists(): raise FileExistsError(path)
    atomic_torch_save(data, path)

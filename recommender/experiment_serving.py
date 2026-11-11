"""Explicit global refresh for experimental purchase caches (never a request hook)."""
import copy
import torch
from recommender.experimental_retrieval import PurchaseEmbeddingCache


def refresh_catalogue(model, old_product_ids, new_product_ids, model_version, catalogue_version):
    """Return a separate model/cache namespace with canonical bias remapping.

    Encode the refreshed product metadata with returned_model.product_vectors
    once, rebuild customer snapshots via on_purchase using the new one-based
    history IDs, then switch the serving reference atomically. The old model and
    cache remain usable until the rebuild completes. No demand values are used.
    """
    if len(set(new_product_ids)) != len(new_product_ids):
        raise ValueError('Catalogue IDs must be unique')
    if len(old_product_ids) != model.towers.product_bias.numel():
        raise ValueError('Old catalogue does not match learned bias')
    refreshed=copy.deepcopy(model).eval()
    old_index={sku:i for i,sku in enumerate(old_product_ids)}
    bias=torch.zeros(len(new_product_ids),device=model.towers.product_bias.device)
    with torch.no_grad():
        for i,sku in enumerate(new_product_ids):
            if sku in old_index:
                bias[i]=model.towers.product_bias[old_index[sku]]
    refreshed.towers.product_bias=torch.nn.Parameter(bias,requires_grad=False)
    return refreshed,PurchaseEmbeddingCache(refreshed,model_version,catalogue_version)

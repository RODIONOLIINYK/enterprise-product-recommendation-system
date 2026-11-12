"""Named input contracts, sealed slate payloads and isolated training runs.

Model schema 3 remains unchanged. Cache format 4 adds validation metadata and
uses a new filename; historical caches/checkpoints are never migrated in place.
"""
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4
import pandas as pd
import torch
from recommender.split import TRAIN_CUTOFF, TEST_START

CACHE_FORMAT_VERSION = 4
CHECKPOINT_FORMAT_VERSION = 4


def feature_contract():
    from recommender.candidates import PAIRWISE_FEATURE_NAMES
    from recommender.features import LEGACY_PROFILE_FEATURES
    from recommender.sequences import ENRICHED_FEATURE_NAMES
    return {"version": 1, "floating_dtype": "float32", "company_text_dimension": 1024,
            "candidate_vector_space": "retriever_digest", "profile": list(LEGACY_PROFILE_FEATURES),
            "history": list(ENRICHED_FEATURE_NAMES), "pairwise": list(PAIRWISE_FEATURE_NAMES),
            "history_ids": "one_based_training_vocabulary; zero_unknown_or_padding",
            "candidate_ids": "zero_based_ordered_catalogue", "history_length": 32,
            "date_origin": "2025-01-01", "purchase_date_units": "years/365.25",
            "prediction_date_units": "calendar_days", "history_policy": "strictly_before_prediction"}


def file_digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def transition_keys(transitions):
    return [(str(row["customer_id"]), int((pd.Timestamp(row["as_of_date"]).normalize() - pd.Timestamp("2025-01-01")).days),
             tuple(sorted(str(sku) for sku in row["positive_product_ids"]))) for row in transitions]


TENSOR_WIDTHS = {"text_embedding": (1024,), "city_id": (), "region_id": (), "industry_id": (),
                 "tabular_features": (8,), "sequence": (32, 9), "attention_mask": (32,),
                 "seq_product_ids": (32,), "seq_purchase_dates": (32,), "seq_category_ids": (32,),
                 "seq_subcategory_ids": (32,), "seq_business_line_ids": (32,), "prediction_days": ()}
SEALED_KEYS = (*TENSOR_WIDTHS, "retriever_tabular_features", "candidate_indices", "pairwise_features", "labels",
               "candidate_vectors", "customer_ids", "positive_pids", "candidate_pids", "row_keys", "cache_identity",
               "cache_format_version", "pairwise_metadata_policy", "input_contract", "input_contract_digest", "catalogue_product_ids")


def payload_digest(data):
    digest = hashlib.sha256()
    for key in SEALED_KEYS:
        value = data.get(key)
        digest.update(key.encode())
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().contiguous()
            digest.update(repr((tuple(value.shape), str(value.dtype))).encode())
            digest.update(value.numpy().tobytes())
        else:
            digest.update(json.dumps(value, sort_keys=True).encode())
    return digest.hexdigest()


def seal_slates(data, transitions, product_ids):
    data.setdefault("pairwise_metadata_policy", "snapshot_unverified")
    data.update(cache_format_version=CACHE_FORMAT_VERSION, input_contract=feature_contract(),
                input_contract_digest=json_digest(feature_contract()), row_keys=transition_keys(transitions),
                catalogue_product_ids=list(product_ids))
    data["cache_payload_digest"] = payload_digest(data)
    return data


def validate_slates(data, *, transitions=None, product_ids=None, split=None, candidates=None):
    """Reject malformed, stale, permuted and future-bearing payloads before use."""
    def require(condition, message):
        if not condition:
            raise ValueError(f"Invalid slate cache: {message}")
    require(isinstance(data, dict), "expected mapping")
    require(data.get("cache_format_version") == CACHE_FORMAT_VERSION, "cache format; rebuild in v4 namespace")
    require(data.get("input_contract") == feature_contract(), "ordered feature contract")
    require(data.get("input_contract_digest") == json_digest(feature_contract()), "feature contract digest")
    catalogue = data.get("catalogue_product_ids")
    require(isinstance(catalogue, list) and bool(catalogue) and len(catalogue) == len(set(catalogue)) and all(isinstance(sku, str) for sku in catalogue), "catalogue IDs")
    if product_ids is not None:
        require(catalogue == list(product_ids), "catalogue order")
    customers = data.get("customer_ids")
    require(isinstance(customers, list) and len(customers) > 0, "customer rows")
    n = len(customers)
    for key, width in TENSOR_WIDTHS.items():
        value = data.get(key)
        require(isinstance(value, torch.Tensor) and value.shape == (n, *width), f"{key} shape")
        require(bool(torch.isfinite(value).all()), f"{key} nonfinite")
    indices = data.get("candidate_indices")
    require(isinstance(indices, torch.Tensor) and indices.ndim == 2 and indices.shape[0] == n, "candidate shape")
    c = indices.shape[1]
    require(c > 0 and (candidates is None or c == candidates), "candidate count")
    for key in ("city_id", "region_id", "industry_id", "seq_product_ids", "seq_category_ids", "seq_subcategory_ids", "seq_business_line_ids", "candidate_indices"):
        value = data[key]
        require(value.dtype in (torch.int32, torch.int64) and bool((value >= 0).all()), f"{key} index dtype/range")
    require(bool((indices < len(catalogue)).all()), "candidate index bounds")
    for key, shape in (("pairwise_features", (n, c, 13)), ("labels", (n, c))):
        value = data.get(key)
        require(isinstance(value, torch.Tensor) and value.shape == shape and bool(torch.isfinite(value).all()), f"{key} shape/finite")
        require(bool(((value >= 0) & (value <= 1)).all()), f"{key} range")
    for key in ("text_embedding", "tabular_features", "sequence", "seq_purchase_dates", "pairwise_features", "labels"):
        require(data[key].dtype == torch.float32, f"{key} float32 dtype")
    for index in (0, 10, 11, 12):
        values = data["pairwise_features"][..., index]
        require(bool(((values == 0) | (values == 1)).all()), "binary familiarity flags")
    mask = data["attention_mask"]
    require(bool(((mask == 0) | (mask == 1)).all()), "binary history mask")
    require(torch.equal(data["sequence"][..., 8], mask.to(data["sequence"].dtype)), "history validity indicator")
    require(not bool((mask[:, 1:] > mask[:, :-1]).any()), "right-padded history")
    for key in ("seq_product_ids", "seq_category_ids", "seq_subcategory_ids", "seq_business_line_ids"):
        require(not bool(data[key].masked_select(~mask.bool()).any()), f"{key} padding")
    days = data["prediction_days"]
    require(torch.equal(days, days.round()), "integral prediction days")
    cutoff = (TRAIN_CUTOFF - pd.Timestamp("2025-01-01")).days
    test_start = (TEST_START - pd.Timestamp("2025-01-01")).days
    if split == "train": require(bool((days < cutoff).all()), "training split dates")
    elif split in ("val", "validation"): require(bool(((days >= cutoff) & (days < test_start)).all()), "validation split dates")
    elif split == "test": require(bool((days >= test_start).all()), "test split dates")
    else: require(bool((days < test_start).all()), "development split dates")
    require(not bool(((data["seq_purchase_dates"] * 365.25).round() >= days[:, None]).masked_select(mask.bool()).any()), "future history")
    for key in ("candidate_pids", "positive_pids"):
        require(isinstance(data.get(key), list) and len(data[key]) == n, f"{key} rows")
    row_keys = []
    for row, cid in enumerate(customers):
        positives, pids = data["positive_pids"][row], data["candidate_pids"][row]
        require(isinstance(cid, str) and bool(cid), "customer ID")
        require(len(pids) == c and len(set(pids)) == c and all(isinstance(s, str) for s in pids), "candidate IDs/duplicates")
        require([catalogue[i] for i in indices[row].tolist()] == pids, "candidate ID/index alignment")
        require(bool(positives) and len(set(positives)) == len(positives) and all(isinstance(sku, str) for sku in positives), "positive IDs")
        expected = torch.tensor([float(sku in positives) for sku in pids], dtype=data["labels"].dtype)
        require(torch.equal(data["labels"][row].cpu(), expected), "labels disagree with target IDs")
        row_keys.append((cid, int(days[row]), tuple(sorted(positives))))
    require(data.get("row_keys") == row_keys, "row identities/dates")
    require(len({key[:2] for key in row_keys}) == n, "duplicate row identities")
    if transitions is not None:
        require(row_keys == transition_keys(transitions), "source transition order/dates/targets")
    tab = data.get("retriever_tabular_features")
    if tab is not None:
        require(tab.ndim == 2 and tab.shape[0] == n and bool(torch.isfinite(tab).all()), "retriever profile")
    vectors = data.get("candidate_vectors")
    if vectors is not None:
        require(vectors.shape == (n, c, 128) and bool(torch.isfinite(vectors).all()), "candidate vectors")
        require(vectors.dtype == torch.float32, "candidate vectors float32 dtype")
    if tab is not None:
        require(tab.dtype == torch.float32, "retriever profile float32 dtype")
    require(data.get("cache_payload_digest") == payload_digest(data), "payload checksum")
    return data


def atomic_json(value, path, *, overwrite=True):
    path = Path(path)
    with NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
        try:
            json.dump(value, temporary, indent=2)
            temporary.write("\n")
            temporary.flush()
            if overwrite:
                temporary_path.replace(path)
            else:
                os.link(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)


def create_run_directory(root, run_id=None):
    root = Path(root)
    run_id = run_id or f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{uuid4().hex[:10]}"
    if Path(run_id).name != run_id or run_id in (".", ".."):
        raise ValueError("run_id must be a single directory name")
    directory = root / "runs" / run_id
    directory.mkdir(parents=True, exist_ok=False)
    atomic_json({"version": 1, "run_id": run_id, "status": "running", "selected_checkpoint": None}, directory / "run.json")
    return directory


def ranker_provenance(*, retriever_digest, train_data, val_data, model_config, selection_metric, metadata_policy="snapshot_unverified"):
    contract = feature_contract()
    return {"version": 1, "input_contract": contract, "input_contract_digest": json_digest(contract),
            "retriever_digest": retriever_digest, "slates": {
                name: {"identity": data["cache_identity"], "payload_digest": data["cache_payload_digest"]}
                for name, data in (("train", train_data), ("val", val_data))},
            "model_config": json.loads(json.dumps(model_config)), "model_config_digest": json_digest(model_config),
            "selection_metric": selection_metric, "metadata_policy": metadata_policy}


def validate_ranker_provenance(checkpoint, *, retriever_digest, slate_identity=None, payload_digests=None, allow_legacy=True):
    if checkpoint.get("retriever_digest") != retriever_digest or checkpoint.get("slate_schema_version") != 3:
        raise ValueError("Ranker checkpoint retriever/model schema mismatch")
    provenance = checkpoint.get("provenance")
    version = checkpoint.get("checkpoint_format_version")
    if version is not None and version != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("Unsupported ranker checkpoint container format")
    if provenance is None:
        if version is not None:
            raise ValueError("Versioned checkpoint lacks required provenance")
        if not allow_legacy:
            raise ValueError("Legacy checkpoint lacks ordered input/slate provenance")
        return "legacy_unverified_input_contract"
    if (provenance.get("version") != 1 or provenance.get("input_contract") != feature_contract()
            or provenance.get("input_contract_digest") != json_digest(feature_contract())
            or provenance.get("retriever_digest") != retriever_digest
            or provenance.get("model_config") != checkpoint.get("config")
            or provenance.get("model_config_digest") != json_digest(checkpoint.get("config"))
            or provenance.get("model_config_digest") != json_digest(provenance.get("model_config"))
            or provenance.get("selection_metric") != checkpoint.get("selection_metric")
            or provenance.get("metadata_policy") != checkpoint.get("config", {}).get("pairwise_metadata_policy", "snapshot_unverified")):
        raise ValueError("Ranker checkpoint feature/model provenance mismatch")
    for name in ("train", "val"):
        source = provenance.get("slates", {}).get(name, {})
        digest = source.get("payload_digest", "")
        if not source.get("identity") or not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Ranker checkpoint lacks originating slate provenance")
    if slate_identity is not None and provenance["slates"]["val"]["identity"] != slate_identity:
        raise ValueError("Ranker checkpoint originates from different validation slates")
    for name, digest in (payload_digests or {}).items():
        if provenance["slates"][name]["payload_digest"] != digest:
            raise ValueError("Ranker checkpoint slate payload differs from originating cache")
    return "verified_named_contract"


def resolve_run_checkpoint(path):
    """Resolve an explicitly selected completed run; never search parent aliases."""
    path = Path(path)
    if path.is_dir():
        path = path / "run.json"
    if path.name != "run.json":
        return path
    manifest = json.loads(path.read_text())
    name = manifest.get("selected_checkpoint")
    if manifest.get("status") != "complete" or not name:
        raise ValueError("Run has no qualifying selected checkpoint")
    if Path(name).name != name:
        raise ValueError("Invalid checkpoint path in run manifest")
    checkpoint = path.parent / name
    if file_digest(checkpoint) != manifest.get("selected_sha256"):
        raise ValueError("Selected checkpoint differs from run manifest")
    return checkpoint

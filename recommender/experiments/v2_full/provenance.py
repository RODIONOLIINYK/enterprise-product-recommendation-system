"""Content identities for resumable full-pool experiments."""
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def input_digest(value):
    """Hash ordered input structures without copying full memory-mapped arrays into RAM."""
    digest = hashlib.sha256()

    def visit(obj):
        if isinstance(obj, torch.Tensor):
            obj = obj.detach().cpu().numpy()
        if isinstance(obj, np.ndarray):
            if obj.dtype.hasobject:
                visit(obj.tolist())
                return
            visit(("ndarray", obj.dtype.str, obj.shape))
            if obj.flags.c_contiguous and obj.size:
                digest.update(memoryview(obj).cast("B"))
            elif obj.size:
                for block in np.nditer(obj, flags=["external_loop", "buffered"], order="C"):
                    digest.update(block.tobytes())
        elif isinstance(obj, pd.DataFrame):
            visit(("dataframe", list(obj.columns), list(map(str, obj.dtypes))))
            visit(pd.util.hash_pandas_object(obj, index=True).to_numpy())
        elif isinstance(obj, Mapping):
            # Feature concatenation can depend on the iteration order of attribute mappings.
            digest.update(b"{")
            for key, item in obj.items():
                visit(key)
                visit(item)
            digest.update(b"}")
        elif isinstance(obj, (list, tuple)):
            digest.update(b"[")
            for item in obj:
                visit(item)
            digest.update(b"]")
        else:
            if isinstance(obj, np.generic):
                obj = obj.item()
            if isinstance(obj, Path):
                obj = str(obj)
            digest.update(json.dumps(obj, ensure_ascii=False).encode())
            digest.update(b"\0")

    visit(value)
    return digest.hexdigest()


def prepared_dataset_digest(shared, train, val, train_candidates, val_candidates, purchases, catalogue):
    train_fields = {k: v for k, v in train.items() if k != "cand_38_path"}
    val_fields = {k: v for k, v in val.items() if k != "cand_38_path"}
    return input_digest((shared, train_fields, val_fields, train_candidates, val_candidates, purchases, catalogue))

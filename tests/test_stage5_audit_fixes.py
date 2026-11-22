"""Regression cases for Stage 5 reference protection, temporal features, and serving."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from recommender.artifacts import file_digest
from recommender.experiments.roadmap_v2 import (
    demo_dashboard as demo,
    evaluate_stage5_test as cases,
    native_base_name as native,
    run_stage5 as training,
    run_stage5_clean_v4,
    run_stage5_clean_v4b,
    run_stage5_pure_bn,
    run_stage5_v5_shared_uv,
    stage5_artifacts as artifacts,
)
from recommender.experiments.roadmap_v2.parsed_spec_features import (
    parse_viscosity_grades,
    parse_company_equipment_domains,
)
from recommender.experiments.roadmap_v2.stage5_base_entity import (
    build_base_name_sequences,
    build_base_name_universe,
    evaluate_base_name_predictions,
)
from test_stage5_native_bn import _synthetic_fixtures


@pytest.fixture
def pointers(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "EVIDENCE_ROOT", tmp_path)
    paths = []
    for i, name in enumerate(("stage5_reference", "stage5_3stage_reference")):
        checkpoint = tmp_path / f"frozen{i}" / "best.pt"
        summary = tmp_path / f"results{i}.json"
        checkpoint.parent.mkdir()
        for p in (checkpoint, summary):
            p.write_bytes(b"pinned")
        pointer = tmp_path / (name + ".json")
        pointer.write_text(
            json.dumps(
                {
                    "checkpoints": {
                        k: {"path": str(p), "sha256": file_digest(p)}
                        for k, p in (("ranker", checkpoint), ("summary", summary))
                    }
                }
            )
        )
        paths.append((checkpoint, summary, pointer))
    return paths


@pytest.mark.parametrize(
    "runner",
    [
        run_stage5_clean_v4.run_clean_v4,
        run_stage5_clean_v4b.run_clean_v4b,
        run_stage5_pure_bn.run_pure_bn_experiment,
        run_stage5_v5_shared_uv.run_v5_shared_uv,
    ],
)
def test_runners_refuse_pinned_outputs_before_loading_data(tmp_path, pointers, runner):
    new_run = tmp_path / "never-created"
    for checkpoint, summary, pointer in pointers:
        alias = tmp_path / (summary.stem + "-alias")
        alias.symlink_to(summary)
        for pinned in (checkpoint, pointer, alias):
            with pytest.raises(FileExistsError, match="pinned"):
                runner(output_dir=new_run, summary_path=pinned)
        with pytest.raises(FileExistsError, match="pinned"):
            runner(output_dir=checkpoint.parent)
        assert checkpoint.read_bytes() == summary.read_bytes() == b"pinned"
    assert not new_run.exists()


def test_output_paths_are_exclusive(tmp_path, pointers):
    a, summary = artifacts.stage5_run_paths(tmp_path)
    b, _ = artifacts.stage5_run_paths(tmp_path)
    assert a.is_dir() and b.is_dir() and a != b and summary.parent == a
    with pytest.raises(FileExistsError):
        artifacts.stage5_run_paths(tmp_path, a)


def test_dashboard_validates_every_artifact_before_opening_data(monkeypatch, pointers):
    checkpoint, summary, pointer = pointers[0]
    monkeypatch.setattr(demo, "ACTIVE_REF_PATH", pointer)
    monkeypatch.setattr(demo, "load_development_purchases", lambda *_: pytest.fail("Opened purchase data"))
    for pinned in (summary, checkpoint):
        pinned.write_bytes(b"changed")
        with pytest.raises(ValueError, match="checksum mismatch"):
            demo.Stage5LiveDemoEngine(allow_heldout_test=True)
        pinned.write_bytes(b"pinned")


@pytest.fixture
def vectors(tmp_path):
    directory = tmp_path / "tower"
    directory.mkdir()
    (directory / "best.pt").write_bytes(b"tower")
    names = ["A", "B"]
    splits = {
        s: {"customer_ids": ["c1", "c2"], "prediction_days": torch.tensor(days)}
        for s, days in (("train", [1, 2]), ("val", [10, 20]))
    }
    inputs = {
        "text_embedding": torch.ones(2, 4),
        "demand_features": torch.zeros(2, 65),
        "attr_ids": {"unit": torch.arange(2)},
    }
    for name, split in splits.items():
        identity = artifacts.vector_identity(
            split, names, file_digest(directory / "best.pt"), artifacts.product_inputs_digest(inputs)
        )
        torch.save(
            {**identity, "u_bn": torch.zeros(2, 128), "v_bn": torch.ones(2, 128)},
            directory / f"{name}_vectors.pt",
        )
    return directory, splits, names, inputs


def _load_vectors(directory, splits, names, inputs, *, allow_unsealed_legacy=False):
    exports, legacy = artifacts.read_stage5_vectors(directory, allow_unsealed_legacy=allow_unsealed_legacy)
    return artifacts.validate_stage5_vectors(
        exports, splits, names, file_digest(directory / "best.pt"), inputs, legacy=legacy
    )


@pytest.mark.parametrize("field", ["tower", "catalogue", "row_keys", "text", "specs", "attributes"])
def test_sealed_vector_load_rejects_same_shape_mismatches(vectors, field):
    directory, splits, names, inputs = vectors
    assert len(_load_vectors(*vectors)) == 3
    if field == "tower":
        (directory / "best.pt").write_bytes(b"other")
    elif field == "catalogue":
        names.reverse()
    elif field == "row_keys":
        splits["val"]["prediction_days"] = splits["val"]["prediction_days"].flip([0])
    elif field == "attributes":
        inputs["attr_ids"]["unit"] += 1
    else:
        inputs["text_embedding" if field == "text" else "demand_features"][1, 0] += 1
    with pytest.raises(ValueError, match="vector identity"):
        _load_vectors(*vectors, allow_unsealed_legacy=True)


def test_vector_shapes_partial_seals_and_legacy_escape_hatch(vectors):
    directory, _, _, _ = vectors
    for split in ("train", "val"):
        torch.save(
            {"u_bn": torch.ones(2, 128), "v_bn": torch.zeros(2, 128)}, directory / f"{split}_vectors.pt"
        )
    with pytest.raises(ValueError, match="identity"):
        _load_vectors(*vectors)
    _load_vectors(*vectors, allow_unsealed_legacy=True)
    path = directory / "val_vectors.pt"
    original = torch.load(path, weights_only=True)
    for field, value in [
        ("ordered_keys", []),
        ("u_bn", torch.ones(1, 128)),
        ("v_bn", torch.ones(2, 128)),
        ("u_bn", torch.full((2, 128), torch.nan)),
    ]:
        torch.save({**original, field: value}, path)
        with pytest.raises(ValueError):
            _load_vectors(*vectors, allow_unsealed_legacy=True)


def test_cutoff_freezes_static_and_history_specs_with_unseen_packs(tmp_path, monkeypatch):
    cat, co, p, comm, ids, universe, reps = _synthetic_fixtures()
    # A late pack adds a viscosity shared by the second formulation and a new volume tier.
    cat.loc[1, ["product_name", "short_name"]] = ["grease ISO VG 220 209L", "209L"]
    paid = p.iloc[[0]].copy()
    first_sales = pd.Series(pd.to_datetime(["2025-05-01", "2025-07-01", "2025-05-02"]), index=ids)
    allowed = native.observed_pack_ids(first_sales, pd.Timestamp("2025-07-01"))
    assert allowed == {ids[0], ids[2]}
    never = cat.iloc[[1]].assign(product_id="never", product_name="ISO VG 68 900L")
    catalogue = pd.concat([cat, never])
    monkeypatch.setattr(
        native,
        "load_text_encoder",
        lambda: SimpleNamespace(encode=lambda texts, **kw: np.zeros((len(texts), 4), np.float32)),
    )

    def build(catalog, members):
        return native.build_base_name_static_representations(
            universe, ids, catalog, reps, allowed_member_pids=members, cache_path=tmp_path / "text.pt"
        )

    strict = build(catalogue, allowed)
    control = build(cat.loc[cat.product_id.isin(allowed)], None)
    assert torch.equal(strict["full_specs_multihot"], control["full_specs_multihot"])
    assert strict["pack_free_texts"] == control["pack_free_texts"]
    legacy = build(catalogue, None)
    assert legacy["volume_multihot"][0].sum() == 2 and strict["volume_multihot"][0].sum() == 1

    def share(members):
        engine = native.NativeBaseNameFeatureEngine(
            paid, comm.iloc[:0], catalogue, co, ids, universe, reps, allowed_member_pids=members
        )
        features, _ = engine.build_split("C1", "2025-05-03")
        return features[1, native.CANDIDATE_38_FEATURE_NAMES.index("user_viscosity_order_share")]

    assert share(None) == 1 and share(allowed) == 0
    salts = native.representative_attr_ids(
        {"product_attr_ids": {"x": torch.arange(1, 4)}}, ids, reps, {ids[0]}
    )
    assert salts["x"].tolist() == [1, 0]


def test_all_public_feature_lookups_exclude_prediction_day():
    cat, co, p, ce, ids, u, reps = _synthetic_fixtures()
    eng = native.NativeBaseNameFeatureEngine(p, ce, cat, co, ids, u, reps)
    for method in (eng.build, eng.build_merged_profile, eng._comm.build):
        np.testing.assert_array_equal(method("C1", "2025-08-01"), method("C1", "2025-08-01 23:59"))
    for a, b in zip(
        eng.build_split("C1", "2025-08-01"), eng.build_split("C1", "2025-08-01 12:00"), strict=True
    ):
        np.testing.assert_array_equal(a, b)
    assert set(eng._global_cache) == {pd.Timestamp("2025-08-01")}


def test_oov_stats_use_distinct_names_but_embedding_ids_stay_zero():
    cat, _, _, _, ids, _, _ = _synthetic_fixtures()
    cat = pd.concat(
        [
            cat,
            cat.iloc[[0]].assign(base_name="unknown A", product_id="A"),
            cat.iloc[[0]].assign(base_name="unknown B", product_id="B"),
        ]
    )
    p = pd.DataFrame(
        {
            "customer_id": ["X"] * 3,
            "product_id": ["A", "B", "A"],
            "quantity": [1.0, 3.0, 2.0],
            "purchase_date": pd.date_range("2025-01-01", periods=3),
        }
    )
    u = build_base_name_universe(ids, cat, p)
    a = demo.Stage5LiveDemoEngine.__new__(demo.Stage5LiveDemoEngine)
    a.universe, a.product_ids, a.cat_indexed, a.all_purchases = u, ids, cat.set_index("product_id"), p
    a.representatives = cat.iloc[[0, 2]]
    a.vocabs = dict.fromkeys(("category_vocab", "subcategory_vocab", "business_line_vocab"), {})
    a._init_fast_sequence_and_resolver_indices()
    batch = build_base_name_sequences(p, ["X"], [3], u, a.representatives, {}, {}, {})
    fast = a._fast_build_sequence("X", 3)
    for key in batch:
        assert torch.equal(batch[key], fast[key])
    assert batch["attention_mask"].sum() == 3 and not batch["seq_product_ids"].any()
    assert batch["sequence"][0, :3, 7].tolist() == [0, 0, 1]
    assert batch["sequence"][0, 1, 5] == 0
    np.testing.assert_allclose(batch["sequence"][0, :3, 6], [1, 0.75, 0.5])


def test_multigrade_share_is_union_of_distinct_collapsed_events():
    cat, co, p, ce, ids, _, _ = _synthetic_fixtures()
    cat = cat.assign(
        base_name=["ISO VG 220", "NLGI 2", "ISO VG 220 NLGI 2"],
        product_name=["ISO VG 220", "NLGI 2", "ISO VG 220 NLGI 2"],
    )
    p = p.iloc[:3].assign(
        customer_id="C1", product_id=ids, purchase_date=pd.date_range("2025-05-01", periods=3)
    )
    u = build_base_name_universe(ids, cat, p)
    i = u["bn_to_idx"]["ISO VG 220 NLGI 2"]
    eng = native.NativeBaseNameFeatureEngine(p, ce.iloc[:0], cat, co, ids, u, u["bn_rep_pid_idx"])
    for day in ["2025-05-03", "2025-05-04"]:
        cand, _ = eng.build_split("C1", day)
        assert cand[i, native.CANDIDATE_38_FEATURE_NAMES.index("user_viscosity_order_share")] == 1


@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_live_features_match_offline_storage(dtype):
    values = np.full((3, 38), 0.1234567, dtype=np.float32)
    offline = torch.from_numpy(values.astype(native.candidate_storage_dtype(dtype))).float()
    live = demo._live_candidate_features(values, native.candidate_storage_dtype(dtype))
    assert torch.equal(offline, torch.from_numpy(live))
    assert bool(np.array_equal(values, live)) == (dtype == "float32")


def test_grade_punctuation_case_and_company_word_boundaries():
    assert parse_viscosity_grades("iso vg 46, 20L") == {"ISO46"}
    assert parse_viscosity_grades("SAE 5W-30, 4L") == {"5W-30"}
    for decimal in ("0.4кг", "0,4кг"):
        assert not parse_viscosity_grades(f"Gadus S2 V220 {decimal}") & {"NLGI0"}
    assert "marine_rail_aviation" not in parse_company_equipment_domains("транспортные услуги")
    assert "marine_rail_aviation" in parse_company_equipment_domains("морской порт")


def test_dashboard_novelty_includes_unmodeled_new_and_repeat(monkeypatch):
    # Use a synthetic evaluation boundary wholly within the training period.
    monkeypatch.setattr(cases, "TEST_START", pd.Timestamp("2025-01-02"))
    monkeypatch.setattr(cases, "SLATE_SIZE", 1)
    events = pd.DataFrame(
        {
            "customer_id": ["X"] * 4,
            "product_id": ["A", "A", "B", "B"],
            "purchase_date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-02", "2025-01-03"]),
        }
    )
    u = {
        "all_pid_to_bn": {"p": "modeled", "A": "older", "B": "newer"},
        "bn_to_idx": {"modeled": 0},
        "pid_idx_to_bn_idx": np.array([0]),
    }
    rows, _ = cases.build_test_basket_cases(events, ["p"], u, SimpleNamespace(eligible_on=lambda dt: {"p"}))
    assert dict(rows[0].unmodeled_novelty) == {"older": False, "newer": True}
    assert dict(rows[1].unmodeled_novelty) == {"newer": False}
    assert demo._case_filter_indices(rows)["has_new"] == [0]


def test_padding_cannot_change_cutoff_or_discount():
    truth = torch.tensor([[True]])
    for rank in (2, 6, 11):
        ranking = np.array([[-1] * (rank - 1) + [0]])
        m = evaluate_base_name_predictions(ranking, truth, truth, ["x"])
        assert m["true_new_bn_hits@5"] == (rank <= 5)
        assert m["bn_ndcg@10"] == pytest.approx(1 / np.log2(rank + 1) if rank <= 10 else 0)


def test_customer_activity_and_manager_change_window_boundaries():
    cat, co, _, ce, ids, u, reps = _synthetic_fixtures()
    date = pd.Timestamp("2025-09-01")
    dates = [date - pd.Timedelta(days=i) for i in (91, 90, 30, 0)]
    p = pd.DataFrame({"customer_id": "C1", "product_id": ids[0], "purchase_date": dates, "quantity": 1.0})
    ce = ce.assign(purchase_date=dates[:2])
    eng = native.NativeBaseNameFeatureEngine(p, ce, cat, co, ids, u, reps)
    _, customer = eng.build_split("C1", date)
    idx = native.CUSTOMER_DYNAMIC_11_FEATURE_NAMES.index
    assert customer[idx("customer_30d_activity")] == pytest.approx(np.log1p(1) / np.log1p(1000))
    assert customer[idx("customer_90d_activity")] == pytest.approx(np.log1p(2) / np.log1p(1000))
    assert customer[idx("mgr_changed_recently")] == 1
    _, tomorrow = eng.build_split("C1", date + pd.Timedelta(days=1))
    assert tomorrow[idx("mgr_changed_recently")] == 0


def test_replay_export_roundtrip_and_unsealed_value_verification(tmp_path, monkeypatch):
    """Exercise the actual exporter using inference-only replay, with no fitting or private inputs."""

    class TinyTower(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.0))

        def product_vectors(self, inputs):
            return self.scale * inputs["text_embedding"]

        def encode_purchase(self, inputs, vectors):
            return self.scale * inputs["text_embedding"].unsqueeze(1)

        def score_cached(self, customer, vectors, indices, familiarity):
            return customer.squeeze(1) @ vectors.T

    monkeypatch.setattr(training, "_stage5_tower", lambda *args, **kwargs: TinyTower())
    monkeypatch.setattr(torch.optim.AdamW, "step", lambda *a, **kw: pytest.fail("Unexpected training step"))
    tower = TinyTower().eval()
    best = tmp_path / "source" / "best.pt"
    best.parent.mkdir()
    torch.save({"epoch": 1, "h25": 2, "h50": 2, "model_state_dict": tower.state_dict()}, best)
    inputs = {
        "text_embedding": torch.eye(2, 128),
        "demand_features": torch.zeros(2, 65),
        "attr_ids": {"cat": torch.arange(2)},
    }
    names = ["oil", "grease"]
    rows = {
        "customer_ids": ["a", "b"],
        "prediction_days": torch.arange(2),
        "target_bn": torch.eye(2).bool(),
        "new_target_bn": torch.eye(2).bool(),
        "unseen_bn": torch.ones(2, 2, dtype=torch.bool),
        "familiarity_bn": torch.zeros(2, 2, 4),
        "text_embedding": inputs["text_embedding"].clone(),
    }
    rows.update(
        {
            key: torch.zeros(2, dtype=torch.long)
            for key in ("city_id", "region_id", "industry_id", "sales_branch")
        }
    )
    rows.update(
        {
            key: torch.zeros(2, 1)
            for key in (
                "tabular_features",
                "sequence",
                "attention_mask",
                "seq_product_ids",
                "seq_purchase_dates",
            )
        }
    )
    splits = {"train": rows, "val": {**rows, "prediction_days": torch.tensor([10, 11])}}
    out_dir = tmp_path / "export"
    training.train_stage5_two_tower(
        bn_data=splits,
        prod_inputs=inputs,
        universe={"num_base_names": 2, "trained_base_names": 2, "base_names": names},
        enrichment={},
        out_dir=out_dir,
        replay_checkpoint=best,
        device="cpu",
    )
    exports, legacy = artifacts.read_stage5_vectors(out_dir)
    assert not legacy
    artifacts.validate_stage5_vectors(exports, splits, names, file_digest(best), inputs)
    verify = run_stage5_v5_shared_uv._verify_unsealed_vectors
    verify(tower, inputs, splits, exports, torch.device("cpu"))
    exports["train"]["u_bn"] = exports["train"]["u_bn"].flip([0])
    with pytest.raises(ValueError, match="ordered basket"):
        verify(tower, inputs, splits, exports, torch.device("cpu"))


def test_frozen_product_specs_retain_original_grade_parser(tmp_path, monkeypatch):
    cat, _, _, _, ids, u, reps = _synthetic_fixtures()
    cat = cat.assign(product_name=["iso vg 46, 20L", "unused", "unused"])
    monkeypatch.setattr(
        native,
        "load_text_encoder",
        lambda: SimpleNamespace(encode=lambda texts, **kw: np.zeros((len(texts), 4), np.float32)),
    )
    opts = dict(cache_path=tmp_path / "text.pt")
    legacy = native.build_base_name_static_representations(u, ids, cat, reps, frozen_v5_specs=True, **opts)
    new = native.build_base_name_static_representations(u, ids, cat, reps, **opts)
    index = native.VISCOSITY_TO_IDX["ISO46"]
    assert legacy["viscosity_multihot"][0, index] == 0
    assert new["viscosity_multihot"][0, index] == 1


@pytest.mark.parametrize("replay", [False, True])
def test_dashboard_request_labels_and_precise_ranker_inputs(monkeypatch, replay):
    """Exercise the same live entry point as HTTP with synthetic history, ranker and labels."""
    import threading

    cat, co, purchases, commercial, ids, universe, reps = _synthetic_fixtures()
    purchases["purchase_date"] = pd.to_datetime(purchases.purchase_date)
    engine = demo.Stage5LiveDemoEngine.__new__(demo.Stage5LiveDemoEngine)
    engine.lock = threading.Lock()
    engine.dev = torch.device("cpu")
    engine.candidate_dtype = native.candidate_storage_dtype("float16")
    engine.frozen_v5_replay = replay
    engine.universe, engine.product_ids, engine.n_bn = universe, ids, len(reps)
    engine.cat_indexed, engine.representatives = cat.set_index("product_id"), cat.iloc[reps]
    engine.bn_meta = [
        {**row.to_dict(), "bn_idx": i, "category": row["product_category"]}
        for i, (_, row) in enumerate(engine.representatives.iterrows())
    ]
    engine.all_purchases = purchases
    engine.histories = dict(tuple(purchases.groupby("customer_id")))
    engine.texts, engine.profiles, engine.company_map = {}, {}, {}
    engine.vocabs = dict.fromkeys(("category_vocab", "subcategory_vocab", "business_line_vocab"), {})
    engine.native_engine = native.NativeBaseNameFeatureEngine(
        purchases, commercial, cat, co, ids, universe, reps, allow_non_monotonic=True, frozen_v5_replay=replay
    )
    monkeypatch.setattr(demo, "TEST_START", pd.Timestamp("2025-05-01"))
    engine._init_fast_sequence_and_resolver_indices()
    engine._init_relationship_checkpoints()
    engine.cases = [
        cases.HeldoutTestBasket(
            customer="C1",
            date=pd.Timestamp("2025-07-01"),
            first=False,
            has_modeled_target=True,
            target=np.array([True, False]),
            new=np.zeros(2, bool),
            allowed=np.ones(2, bool),
            unseen=np.zeros(2, bool),
            extra_bn_targets=2,
            extra_new_bn_targets=1,
            sku_target=np.ones(3, bool),
            sku_allowed=np.ones(3, bool),
            extra_sku_targets=2,
            extra_new_sku_targets=1,
            unmodeled_novelty=(("OOV repeat", False), ("OOV new", True)),
        )
    ]
    engine.v_bn = torch.zeros(2, 128)
    engine.tower = SimpleNamespace(encode_purchase=lambda inputs, vecs: torch.zeros(1, 1, 128))
    engine.ranker = demo.SharedUVStage5Ranker(aux_60d_enabled=False).eval()
    captured = {}

    def capture(module, args, kwargs):
        captured.update(kwargs)

    engine.ranker.register_forward_pre_hook(capture, with_kwargs=True)
    response = engine.evaluate_case(0)
    assert {g["base_name"]: g["is_new"] for g in response["ground_truth_basket"] if not g["is_modeled"]} == {
        "OOV repeat": False,
        "OOV new": True,
    }
    cand, cust = engine.native_engine.build_split("C1", "2025-07-01")
    assert torch.equal(captured["candidate_features"][0], torch.from_numpy(cand).half().float())
    assert torch.equal(captured["customer_features"][0], torch.from_numpy(cust))
    assert response["case_metrics"]["total_new_targets"] == 1

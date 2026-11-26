"""Train only Arm D: 128D token enrichment without candidate-set attention.

A-C artifacts are verified and reused read-only; the follow-up has its own namespace.
"""

import argparse
import json
from pathlib import Path

import torch

from recommender.artifacts import atomic_json, file_digest
from recommender.candidates import STATISTICS_POLICY_ROLLING
from recommender.experiments.rolling_candidate_v2.data import (
    make_rolling_slate_loader,
    prepare_or_load_rolling_feature_bundle,
    prepare_or_load_rolling_selector_scores,
    prepare_or_load_rolling_slate_cache,
    rolling_ranker_batch_inputs,
)
from recommender.experiments.rolling_candidate_v2.evaluate import paired_rolling_intervals
from recommender.ranker import RankerLoss

from .benchmark import run_stage2_latency_benchmark
from .config import ARM_A_NAME, ARM_C_NAME, CACHE_ROOT
from .model import load_enrichment_ranker_checkpoint
from .train import fit_single_enrichment_arm, snapshot_preserved_hashes, verify_three_arm_init_parity

ARM_D_NAME = "arm_d_pointwise_enriched_128d"
ARTIFACT_ROOT = Path("checkpoints/pointwise-token-enrichment-v1")
EVIDENCE_ROOT = Path("experiments/pointwise_token_enrichment_v1")
PRIOR_SUMMARY = Path("experiments/candidate_token_enrichment_v1/summary.json")


def validate_matched_inputs(prior, meta, feature_manifest, score_manifest, slate_manifest):
    """Refuse an architecture comparison if sources or prepared candidate slates differ."""
    expected = {
        "observed_source_digest": meta["observed_source_digest"],
        "feature_bundle_identity": feature_manifest["identity"],
        "rolling_selector_scores_identity": score_manifest["identity"],
        "slate_cache_identities": slate_manifest["split_cache_identity"],
        "statistics_policy": STATISTICS_POLICY_ROLLING,
        "teacher_score_included": False,
        "seed": 42,
        "epochs": 10,
        "batch_size": 64,
    }
    for key, value in expected.items():
        if prior.get(key) != value:
            raise ValueError(f"Prior campaign does not match current {key}; refusing comparison.")
    # Bind all upstream identities retained by the matched controls, not just the slates.
    for arm in prior["arms"].values():
        settings = arm["settings"]
        for key in ("data_identity", "feature_contract_digest", "observed_source_digest"):
            actual = meta["feature_schema_digest"] if key == "feature_contract_digest" else meta[key]
            if settings[key] != actual:
                raise ValueError(f"Prior arm input mismatch: {key}.")


def prior_artifact_hashes(prior):
    """Verify selected predictions, weights, and histories before reusing a control."""
    found = {}
    for arm in prior["arms"].values():
        paths = [(arm["history_json"], arm["history_sha256"])]
        for selection in ("ndcg", "discovery"):
            entry = arm[selection]
            paths.extend((entry[path], entry[digest]) for path, digest in (
                ("checkpoint", "checkpoint_sha256"),
                ("predictions_npz", "predictions_sha256"),
                ("logits_pt", "logits_sha256"),
            ))
        for path, expected in paths:
            actual = file_digest(path)
            if actual != expected:
                raise ValueError(f"Prior campaign artifact changed: {path}.")
            found[path] = actual
    return found


@torch.no_grad()
def verify_pointwise_init(arm_a, arm_c, slates, vectors):
    """D keeps C's enrichment initialization and A's independent candidate scoring."""
    arm_d = arm_c.clone_arm(candidate_attention_enabled=False, token_enrichment_enabled=True)
    a_state, c_state, d_state = arm_a.state_dict(), arm_c.state_dict(), arm_d.state_dict()
    assert all(torch.equal(value, d_state[key]) for key, value in c_state.items())
    assert all(torch.equal(value, d_state[key]) for key, value in a_state.items())
    assert not any(p.requires_grad for m in (
        arm_d.context_project, arm_d.candidate_attention, arm_d.residual_head
    ) for p in m.parameters())
    batch = next(iter(make_rolling_slate_loader(slates["val"], batch_size=32, shuffle=False)))
    inputs = rolling_ranker_batch_inputs(batch, vectors, torch.device("cpu"))
    arm_d.eval()
    arm_a.eval()
    z_d, z_a = arm_d(**inputs).logits, arm_a(**inputs).logits
    torch.testing.assert_close(z_d, z_a, rtol=0, atol=0)
    for split in ("train", "val"):
        if not bool(slates[split]["candidate_mask"].all()):
            raise ValueError("Existing A-C losses used full slates; padded follow-up is not matched.")
    criterion = RankerLoss()
    y = batch["labels"]
    legacy = criterion(z_a, y)
    corrected = criterion(z_a, y, candidate_mask=inputs["candidate_mask"])
    assert all(torch.equal(legacy[key], corrected[key]) for key in legacy)
    return arm_d, {
        "rows": len(y), "max_abs_logit_diff_vs_A": 0.0,
        "common_initial_weights_exact": True,
        "enrichment_initial_weights_match_C": True,
        "candidate_attention_trainable": False,
        "all_valid_training_and_validation_slates": True,
        "masked_loss_all_valid_exact_parity": True,
    }


def run(*, device=None, artifact_root=ARTIFACT_ROOT, evidence_root=EVIDENCE_ROOT):
    prior_hash = file_digest(PRIOR_SUMMARY)
    prior = json.loads(PRIOR_SUMMARY.read_text())
    before = snapshot_preserved_hashes()
    if before != prior["preserved_hashes"]:
        raise ValueError("Reference/rolling snapshot differs from A-C; do not reuse their results.")
    prior_hashes = prior_artifact_hashes(prior)
    print("Verified prior artifacts; loading sealed rolling inputs.", flush=True)
    data, info, feature_manifest, meta = prepare_or_load_rolling_feature_bundle(
        cache_root=CACHE_ROOT, build=False, statistics_policy=STATISTICS_POLICY_ROLLING,
    )
    _, score_manifest = prepare_or_load_rolling_selector_scores(data, info, meta, cache_root=CACHE_ROOT)
    slates, slate_manifest = prepare_or_load_rolling_slate_cache(data, info, meta, cache_root=CACHE_ROOT)
    validate_matched_inputs(prior, meta, feature_manifest, score_manifest, slate_manifest)
    vectors = data["train"]["v"]
    _, initial = verify_three_arm_init_parity(slates["val"], vectors, seed=42)
    arm_d, init_check = verify_pointwise_init(initial[ARM_A_NAME], initial[ARM_C_NAME], slates, vectors)
    print("Initialization, source identities and all-valid loss parity verified; training Arm D only.", flush=True)
    result = fit_single_enrichment_arm(
        arm_name=ARM_D_NAME, init_model=arm_d,
        train_slate=slates["train"], val_slate=slates["val"], val_data=data["val"],
        product_vectors=vectors, meta=meta, slate_manifest=slate_manifest,
        artifact_root=artifact_root, seed=42, epochs=10, batch_size=64, device=device,
    )
    print("Arm D trained; computing paired customer-clustered comparisons.", flush=True)
    # Verify the entire optimization recipe and checkpoint selection matched the controls.
    d_settings = result["settings"]
    varying = {"arm_name", "candidate_attention_enabled", "token_enrichment_enabled"}
    a_settings = prior["arms"][ARM_A_NAME]["settings"]
    if {k: v for k, v in d_settings.items() if k not in varying} != {
        k: v for k, v in a_settings.items() if k not in varying
    }:
        raise AssertionError("Arm D's training recipe changed relative to A.")
    arms = {**prior["arms"], ARM_D_NAME: result}
    intervals = {
        selection: {
            name + "_to_D": paired_rolling_intervals(
                arm[selection]["predictions_npz"], result[selection]["predictions_npz"]
            ) for name, arm in prior["arms"].items()
        } for selection in ("ndcg", "discovery")
    }
    models = {
        name: load_enrichment_ranker_checkpoint(arm["ndcg"]["checkpoint"], map_location="cpu")[0]
        for name, arm in arms.items()
    }
    benchmark = run_stage2_latency_benchmark(
        models, slates["val"], vectors, evidence_root=evidence_root,
    )
    if snapshot_preserved_hashes() != before or prior_artifact_hashes(prior) != prior_hashes:
        raise AssertionError("Existing reference/cache/experimental artifacts changed.")
    if file_digest(PRIOR_SUMMARY) != prior_hash:
        raise AssertionError("Prior campaign summary changed.")
    summary = {
        "training_executed": True, "heldout_access": False, "promoted": False,
        "newly_trained_arms": [ARM_D_NAME], "seed": 42, "epochs": 10, "batch_size": 64,
        "prior_summary": str(PRIOR_SUMMARY), "prior_summary_sha256": prior_hash,
        "statistics_policy": STATISTICS_POLICY_ROLLING, "teacher_score_included": False,
        "observed_source_digest": meta["observed_source_digest"],
        "feature_bundle_identity": feature_manifest["identity"],
        "slate_cache_identities": slate_manifest["split_cache_identity"],
        "initialization_verification": init_check, "arms": arms,
        "paired_bootstrap_95ci": intervals, "latency_benchmark": benchmark,
        "preserved_hashes": before, "preserved_prior_artifact_hashes": prior_hashes,
        "source_code_sha256": {str(p): file_digest(p) for p in (
            Path(__file__).relative_to(Path.cwd()), Path("recommender/ranker.py"),
            Path("recommender/experiments/candidate_token_enrichment/model.py"),
            Path("recommender/experiments/candidate_token_enrichment/train.py"),
            Path("recommender/experiments/candidate_token_enrichment/evaluate.py"),
        )},
        "limits": [
            "Single training seed; paired bootstrap intervals are conditional on fitted weights.",
            "A-C reused read-only; source identities, artifacts and all-valid objective verified.",
            "Frozen upstream vectors remain a sealed development snapshot, not a live refresh service.",
        ],
    }
    evidence_root.mkdir(parents=True, exist_ok=True)
    atomic_json(summary, evidence_root / "summary.json", overwrite=False)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-training", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    if not args.execute_training:
        parser.error("Pass --execute-training to run the seed-42 enrichment-only arm.")
    summary = run(device=args.device)
    print(json.dumps({s: summary["arms"][ARM_D_NAME][s]["metrics"] for s in ("ndcg", "discovery")}, indent=2))


if __name__ == "__main__":
    main()

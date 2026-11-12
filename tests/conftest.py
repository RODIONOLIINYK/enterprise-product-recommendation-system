"""Keep the ordinary test command independent of held-out real purchases."""

import pytest
from pathlib import Path
import sys


def pytest_addoption(parser):
    parser.addoption("--synthetic-only", action="store_true", default=False,
                     help="Exclude local artifacts and prohibit repository private-data access.")
    parser.addoption(
        "--allow-heldout-data", action="store_true", default=False,
        help="Run unsplit real-purchase tests; requires explicit final-test authorization.",
    )


LOCAL_ARTIFACT_NODES = (
    "test_adversarial_m8_retriever.py::TestAdversarialM8Retriever::test_saved_checkpoint",
    "test_challenger_m8_2_verification.py::TestChallengerM8RetrieverVerification::test_01_",
    "test_challenger_m8_2_verification.py::TestChallengerM8RetrieverVerification::test_02_",
    "test_challenger_m8_2_verification.py::TestChallengerM8RetrieverVerification::test_03_",
    "test_challenger_m9_2_adversarial.py::TestChallengerM9AdversarialVerification::test_01_",
    "test_ranker.py::TestDownstreamEvaluationIntegration::test_from_retriever_checkpoint",
)


def pytest_configure(config):
    if not config.getoption("--synthetic-only"):
        return
    if config.getoption("--allow-heldout-data"):
        raise pytest.UsageError("--synthetic-only cannot authorize held-out reads")
    root = Path(__file__).resolve().parents[1]
    forbidden = [(root / name).resolve() for name in ("data", "checkpoints", "private", "datasets")]
    # Audit open(mode=None) lacks dir_fd context (notably shutil.rmtree);
    # guard high-level Python file reads while CI itself has no private files.
    def guard(event, args):
        if event != "open" or not isinstance(args[0], (str, bytes)) or args[1] is None:
            return
        path = Path(args[0].decode() if isinstance(args[0], bytes) else args[0]).resolve()
        if path.is_dir():
            return  # Directory descriptors used by temporary-tree cleanup contain no records.
        if any(path == base or base in path.parents for base in forbidden):
            raise RuntimeError("Synthetic-only test attempted repository private-data/artifact access")
    sys.addaudithook(guard)


def pytest_collection_modifyitems(config, items):
    for item in items:
        relative = item.nodeid.removeprefix("tests/")
        if any(relative.startswith(prefix) for prefix in LOCAL_ARTIFACT_NODES):
            item.add_marker("local_artifact")
        real = item.get_closest_marker("real_purchase_data") is not None
        local = item.get_closest_marker("local_artifact") is not None
        if config.getoption("--synthetic-only") and (real or local):
            item.add_marker(pytest.mark.skip(reason="Synthetic-only excludes private data and local checkpoints"))
        elif real and not config.getoption("--allow-heldout-data"):
            item.add_marker(pytest.mark.skip(reason="Reads unsplit real purchases; explicit final-test authorization required"))

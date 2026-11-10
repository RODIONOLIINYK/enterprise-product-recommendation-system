"""Keep the ordinary test command independent of held-out real purchases."""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--allow-heldout-data", action="store_true", default=False,
        help="Run unsplit real-purchase tests; requires explicit final-test authorization.",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--allow-heldout-data"):
        return
    skip = pytest.mark.skip(
        reason="Reads unsplit real purchases; requires explicitly authorized --allow-heldout-data.",
    )
    for item in items:
        if item.get_closest_marker("real_purchase_data") is not None:
            item.add_marker(skip)

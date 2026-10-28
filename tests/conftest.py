import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-screenshots",
        action="store_true",
        default=False,
        help="Run optional screenshot tests (requires built UI and Chromium)",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-screenshots"):
        return
    skip = pytest.mark.skip(reason="Use --run-screenshots to run screenshot tests")
    for item in items:
        if "screenshot" in item.keywords:
            item.add_marker(skip)

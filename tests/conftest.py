# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--serial",
        action="store_true",
        help="only run serial marked tests",
    )


def pytest_collection_modifyitems(config, items):
    run_only_serial = config.getoption("--serial")
    for item in items:
        if "serial" in item.keywords and not run_only_serial:
            item.add_marker(
                pytest.mark.skip(
                    reason="This test requires running serially. Use option --serial to run only serial tests"))
        elif "serial" not in item.keywords and run_only_serial:
            item.add_marker(pytest.mark.skip(reason="Only running serial tests."))

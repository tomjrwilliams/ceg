import os
from pathlib import Path

TEST_DB = Path("./__local__/test.db")
TEST_LOGS = Path("./__local__/test.log")

os.environ["FRD_DATA"] = "./__local__/frd"

if TEST_DB.exists():
    os.remove(TEST_DB)
if TEST_LOGS.exists():
    os.remove(TEST_LOGS)

assert not TEST_DB.exists()
assert not TEST_LOGS.exists()

os.environ["TEST_DB"] = str(TEST_DB)
os.environ["TEST_LOGS"] = str(TEST_LOGS)

import pytest
import logging

logging.getLogger("numba").setLevel(logging.INFO)

logger = logging.Logger("PYTEST")

log_handler = logging.FileHandler(TEST_LOGS)
log_handler.setLevel(logging.DEBUG)
logger.addHandler(log_handler)

def pytest_runtest_logstart(nodeid, location):
    filename, lineno, testname = location
    spec = dict(file=filename, line=lineno, name=testname)
    logger.debug("----")
    for k, v in spec.items():
        logger.debug(f"PYTEST {k}: {v}")
    
def pytest_runtest_logreport(report: pytest.TestReport):
    if report.when == "call":
        result: str = report.longreprtext
        if len(result.strip()):
            logger.debug("PYTEST result:")
            logger.debug(result)
        logger.debug(f"PYTEST outcome: {report.outcome} in {report.duration}")

def pytest_runtest_logfinish(nodeid, location):
    logger.debug(f"----")

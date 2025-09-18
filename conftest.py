
import os
from pathlib import Path

TEST_DB = Path("./__local__/test.db")

if TEST_DB.exists():
    os.remove(TEST_DB)
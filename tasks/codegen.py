"""
Code-Generation task
--------------------
Genome = str (Python function).  Fitness = (#failed unit tests).
"""

from __future__ import annotations

import subprocess
import tempfile
import textwrap


# ---------- Unit test stub ----------
UNITTEST_CODE = """
import math
from solution import square

def test_square():
    for i in range(10):
        assert square(i) == i*i
"""

TARGET_TEMPLATE = """
def square(x):
    # TODO: student implementation
    return 0
"""


def _run_pytest(code_str: str) -> int:
    with tempfile.TemporaryDirectory() as td:
        (Path := __import__("pathlib").Path)
        Path(td, "solution.py").write_text(code_str)
        Path(td, "test_solution.py").write_text(UNITTEST_CODE)
        res = subprocess.run(["pytest", "-q"], cwd=td, capture_output=True, text=True)
        failed = 0 if res.returncode == 0 else 1
        return failed


def fitness(code_str: str) -> float:
    try:
        return float(_run_pytest(code_str))
    except Exception:
        return 1.0  # max penalty


def repair(code_str: str) -> str:
    if "def square" not in code_str:
        code_str = TARGET_TEMPLATE + "\n" + code_str
    return textwrap.dedent(code_str)



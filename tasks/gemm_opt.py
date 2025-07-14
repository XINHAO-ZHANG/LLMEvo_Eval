# gemm_opt.py
"""
GEMM CPU‑kernel speed‑up optimisation task for the evolution framework.

* **Genome** – a *string* containing full C source for a GEMM kernel (must
  compile as a stand‑alone translation unit).  You may return only the kernel
  body – we prepend the mandatory header & `main` wrapper automatically.
* **Fitness / loss** – we use **negative speed‑up**:  
  `loss = - (T_baseline / T_candidate)`  
  so that *lower* loss is *better* (consistent with the existing prompt task
  convention where loss↓ = quality↑).

This file plugs into the same interface as `prompt_opt.py`:  it implements
`seed_pool`, `eval`, `repair`, `configure`, `describe`, plus helper functions.
"""
from __future__ import annotations

import hashlib
import json
import random
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

from evolve_core.db import Genome  # type: ignore
from gem5_eval import Simulator  # PIE API – must be installed in env

# ---------------------------------------------------------------------------
#  Configuration constants (over‑ride from driver if wished)
# ---------------------------------------------------------------------------
DEFAULT_TASK_NAME = "gemm"
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # repo root
KERNELS_DIR = PROJECT_ROOT / "kernels"
VARIANTS_DIR = PROJECT_ROOT / "data" / "gemm_variants"
BASELINE_SRC = KERNELS_DIR / "gemm_baseline.c"  # PolyBench naive 3‑for GEMM

# gem5 simulator configuration – adjust to match your Docker image tags.
GEM5_OPTS: Dict[str, str] = {
    "cpu_model": "skylake",  # same as PIE paper
    "opt_level": "O2",       # compile flag inside gem5 container
}

# cache mapping sha1(source) → (loss, candidate_time)
_CACHE: Dict[str, Tuple[float, float]] = {}

# baseline timing will be populated lazily on first call to `_get_baseline()`
_BASELINE_TIME: float | None = None

# ---------------------------------------------------------------------------
#  Internal helpers
# ---------------------------------------------------------------------------

def _sha1(txt: str) -> str:
    return hashlib.sha1(txt.encode()).hexdigest()


def _write_kernel_to_tmp(source: str) -> Path:
    """Write C code to a temporary file and return the path."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="gemm_cand_"))
    src = tmp_dir / "candidate.c"
    src.write_text(source)
    return src


def _get_baseline() -> float:
    """Run (and memoise) baseline kernel inside gem5; returns time in seconds."""
    global _BASELINE_TIME
    if _BASELINE_TIME is not None:
        return _BASELINE_TIME

    if not BASELINE_SRC.exists():
        raise FileNotFoundError(f"Baseline GEMM source not found: {BASELINE_SRC}")

    sim = Simulator(kernel_src=str(BASELINE_SRC), **GEM5_OPTS)  # type: ignore[arg-type]
    _BASELINE_TIME = sim.run()
    return _BASELINE_TIME


def _gem5_time(src: Path) -> float:
    """Compile & run given C file in gem5; return runtime (seconds)."""
    sim = Simulator(kernel_src=str(src), **GEM5_OPTS)  # type: ignore[arg-type]
    return sim.run()


# ---------------------------------------------------------------------------
#  Public API – required by evolution framework
# ---------------------------------------------------------------------------

def configure(cfg=None):
    """Called once by driver; here we just warm‑up baseline measurement."""
    _ = _get_baseline()
    print(f"[gemm_opt] Baseline time = {_BASELINE_TIME:.6f} s (gem5)")


def load_eval_data(task=DEFAULT_TASK_NAME, split="test"):
    """Compat stub – framework expects it; GEMM uses no external data."""
    return [], []


def seed_pool(
    n: int,
    rng: random.Random,
    init_dir: Path | None = None,
) -> List[Genome]:
    """Return *n* initial genomes drawn from `data/gemm_variants/*.c`.

    Each genome gets an immediate loss evaluation so the driver can rank them
    without extra `eval()` calls during generation 0.
    """
    variants_dir = init_dir or VARIANTS_DIR
    paths = list(variants_dir.glob("*.c"))
    if not paths:
        raise FileNotFoundError(f"No .c files found in {variants_dir}. “
                                "Provide at least one seed kernel.")

    rng.shuffle(paths)
    picked = (paths * ((n // len(paths)) + 1))[:n]  # repeat if not enough files

    genomes: List[Genome] = []
    for idx, p in enumerate(picked):
        code = p.read_text()
        loss = eval(code)  # will use cache if evaluated before
        genomes.append(Genome(genome=code, loss=loss,
                              extra={"source": str(p), "idx": idx}))
    return genomes


def eval(code: str, task=DEFAULT_TASK_NAME, split="test") -> float:
    """Fitness function – lower is better (negative speed‑up)."""
    # Cache hit?
    h = _sha1(code)
    if h in _CACHE:
        return _CACHE[h][0]

    baseline_time = _get_baseline()

    # Write candidate kernel to file
    src = _write_kernel_to_tmp(code)
    try:
        cand_time = _gem5_time(src)
    except Exception as e:
        # gem5 run or compilation failed – penalise heavily
        loss = 1e6
    else:
        speedup = baseline_time / cand_time
        loss = -speedup  # negative so higher speedup ⇒ lower loss

    _CACHE[h] = (loss, cand_time if "cand_time" in locals() else float("inf"))
    return loss


def repair(codes: List[str]) -> List[str]:
    """Remove duplicates & blanks in offspring population."""
    seen = set(); out: List[str] = []
    for c in codes:
        c2 = c.strip()
        if c2 and c2 not in seen:
            out.append(c2); seen.add(c2)
    return out


# ---------- prompt helpers for LLM‑driven mutation ----------------------------------

def get_zero_shot_prompt():
    """System+user messages for the *very first* prompt that asks an LLM to
    generate seed GEMM kernels (only used if you bootstrap via LLM).
    """
    sys = (
        "You are a CPU‑kernel optimisation expert.  Produce a C implementation "
        "of a general matrix‑matrix multiplication C = A×B for floating‑point "
        "matrices (double) that will run as fast as possible on an Intel "
        "Skylake‑class CPU.  Keep the same signature: \n"
        "void gemm(int N, double C[N][N], const double A[N][N], const double B[N][N]);\n"
        "*Do not* change parameter order.  Return **only** the kernel code body "
        "(no markdown backticks)."
    )
    return [{"role": "system", "content": sys}]


def get_evolve_prompt(sampled_parents: List[Genome]):
    """Prompt used in every generation to produce a child genome from parents."""
    sys = (
        "You are a CPU‑kernel optimisation expert participating in an "
        "evolutionary search.  Below are several candidate GEMM kernels with "
        "their *negative speed‑up* scores (higher speed‑up ⇒ lower score).  "
        "Write a *new* kernel that is likely to achieve an even lower score.  "
        "You may recombine and mutate the parents: reorder loops, add tiling, "
        "insert AVX2/AVX‑512 intrinsics, OpenMP pragmas, etc., provided the "
        "output remains identical.  Return ONLY compilable C code without "
        "markdown fencing."
    )

    parent_block = "\n\n".join(
        f"/* Parent {i} – score={g.loss:.3f} */\n{g.genome}"
        for i, g in enumerate(sampled_parents, 1)
    )
    user = (
        "Here are the parents:\n" + parent_block + "\n\n" +
        "Produce the child kernel code below:"
    )
    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": user},
    ]


def parse_response(resp: Dict[str, str]):
    """Extract C code text from LLM response JSON."""
    txt = resp.get("text", "").strip()
    # strip triple backticks or ```c fences if present
    if txt.startswith("```"):
        txt = txt.strip("`\n ")
        # after removing fence the first line might be "c" or "C"
        first_newline = txt.find("\n")
        if first_newline != -1 and txt[:first_newline].lower() in {"c", "cpp"}:
            txt = txt[first_newline + 1 :].lstrip()
    return {"genome": txt, "usage": resp.get("usage", {"total_tokens": 0})}


def describe() -> str:
    return (
        "GEMM CPU‑kernel speed‑up optimisation task.  "
        "Genome = C source;  fitness = –(T_base/T_candidate) measured via gem5."
    )

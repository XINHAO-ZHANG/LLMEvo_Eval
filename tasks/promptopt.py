"""
Prompt Optimisation task
------------------------
Genome = str (prompt).  Fitness = 1 - accuracy on a fixed eval set.
"""

from __future__ import annotations

import random
from typing import List

from llm_ops.api import call_llm  # stub; replace with real API

# ---------- Dummy eval set ----------
_EVAL_QUESTIONS = [
    ("What is 2+2?", "4"),
    ("Capital of France?", "Paris"),
    # ... add more
]


def fitness(prompt: str) -> float:
    correct = 0
    for q, ans in _EVAL_QUESTIONS:
        resp = call_llm(prompt + "\nQ: " + q, model="gpt-3.5-turbo", max_tokens=4196)
        out = resp["text"].strip().split("\n")[0]
        if ans.lower() in out.lower():
            correct += 1
    accuracy = correct / len(_EVAL_QUESTIONS)
    return 1.0 - accuracy  # minimise


def repair(prompt: str) -> str:  # trivial
    return prompt.strip() or "Answer the question briefly."



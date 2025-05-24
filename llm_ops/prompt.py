"""
llm_ops.prompt
==============
Centralised prompt builders.
"""

from __future__ import annotations
from textwrap import dedent
from typing import List, Dict, Any
import json, textwrap


# --------------------------------------------------------------------------- #
# 1) Initial population prompt
# --------------------------------------------------------------------------- #
def build_init(
    task_mod,
    pop_size: int,
    *,
    persona: str | None = None,
    batch: int | None = None,
    seen: list[int] | None = None,) -> list[dict]:
    """Return system+user message to generate *pop_size* genomes."""

    batch = batch or pop_size
    assert batch <= pop_size, "Batch size cannot be greater than population size."


    sys = persona or "You are a combinatorial optimisation researcher."
    ques_block = json.dumps(task_mod.get_ques(), ensure_ascii = False, indent=2)
    user = dedent(
        f"""
        TASK  : {task_mod.describe()}
        {ques_block}

        REQUIREMENTS: 
        - Generate **exactly {batch} distinct genomes**.
        - {task_mod.get_genome_desc()}
        - Do NOT wrap genomes in quotes.
        - Output **single JSON** with key \"genomes\" and NOTHING else.
        - No markdown fences, no explanation.
        - Already have {len(seen) if seen else 0} genomes: {seen}, do NOT replicate them.
        - Use creative exploration, avoid duplicates.

        NOW RETURN JSON ONLY."""
    )
    print(user)
    return [{"role": "system", "content": sys},
            {"role": "user", "content": user}]


# --------------------------------------------------------------------------- #
# 2)  sampled evovle prompt  (crossover)
# --------------------------------------------------------------------------- #

def build_evolve_prompt(
        task_mod, 
        parents: Any, 
        scores: List[int]
) -> List[Dict[str, str]]:
    sys = f"""You are an optimization expert helping to solve a hard combinatorial problem. You will be shown several candidate solutions with their scores.
Your goal is to propose better solutions (lower score is better). """
    parent_block = json.dumps(
        [{"genome":g, "score":s} for g,s in zip(parents,scores)],
        ensure_ascii=False, indent=2
    )
    ques_block = json.dumps(task_mod.get_ques(), ensure_ascii = False, indent=2)
    user = dedent(
        f"""
        TASK DESC    : {task_mod.describe()}
        QUESTION    : {ques_block}
        Here are {len(parents)} previous solutions and their scores:
        ```json
        {parent_block}
        ```
        Please return one BETTER child genome as JSON without any extra text: {{ "genome": "<full-new>" }}
        """
    )
    return [{"role": "system", "content": sys},
            {"role": "user", "content": user}]


def build_mutate(
    parent_genome: Any,
    task_mod,
    reflection: str | None = None,
) -> List[Dict[str, str]]:
    sys = f"""You are a mutation operator in intelligent variation in genetic algorithms.
    Your task is to analyze a solution, identify its weaknesses, and apply targeted mutations to improve it.
    You can use the following mutation operations.
    {task_mod.get_mutation_ops()}
    Your goal is to select the variant operations that are most likely to improve the outcome by understanding the problem and the current solution."""

    ques_block = json.dumps(task_mod.get_ques(), ensure_ascii = False, indent=2)
    user = dedent(
        f"""
        TASK DESC    : {task_mod.describe()}
        CURRENT PARENT    : ```\n{parent_genome}\n```
        QUESTION    : {ques_block}
        REFLECTION: {reflection or 'N/A'}

        STEPS TO FOLLOW:
        1. Analyze the strengths and weaknesses of the current parent or solution for solving the given question.
        2. Select 1-3 targeted variations to improve the solution.
        3. Ensure that the mutated solution is still valid.

        MUTATION CONSTRAINTS:
        - Apply at most 3 mutation operations
        - Minimum of 1 mutation
        - Not necessary to use all types of mutations, just choose the most appropriate combination.

        Please return BETTER child genome as JSON, either
          {{ "genome": "<full-new>" }}
        or
         {{ "ops": {task_mod.get_mutation_ops_list()}}}
        """
    )
    return [{"role": "system", "content": sys},
            {"role": "user", "content": user}]


# --------------------------------------------------------------------------- #
# 2.1) Free Mutation prompt (no operator constraints)
# --------------------------------------------------------------------------- #
def build_free_mutate(
    parent_genome: Any,
    task_mod,
    reflection: str | None = None,
) -> List[Dict[str, str]]:
    sys = """You are an expert optimization engineer."""

    ques_block = json.dumps(task_mod.get_ques(), ensure_ascii = False, indent=2)
    user = dedent(
        f"""
        TASK DESCRIPTION: {task_mod.describe()}
        CURRENT BEST SOLUTION: ```\n{parent_genome}\n```
        PROBLEM DETAILS: {ques_block}
        REFLECTION (insights from previous attempts): {reflection or 'N/A'}
        SOLUTION REQUIREMENTS: {task_mod.get_genome_desc()}

        STEPS TO FOLLOW:
        1. Thoroughly analyze the current solution to understand what works and what doesn't.
        2. Design a completely new and improved solution that addresses the weaknesses of the current one.
        3. Ensure your solution is valid according to the requirements.
        4. Make sure your solution is creative and pushes beyond local optima.

        Return your solution as valid JSON with the format:
        {{ "genome": <your complete solution> }}

        Your solution should be significantly different from the current one.
        """
    )
    return [{"role": "system", "content": sys},
            {"role": "user", "content": user}]


# --------------------------------------------------------------------------- #
# 3) Fusion (crossover) prompt
# --------------------------------------------------------------------------- #
def build_fusion(
    parent_a: Any,
    parent_b: Any,
    *,
    task_mod,
    reflection: str | None = None,
) -> List[Dict[str, str]]:
    sys = """You are a crossover operator; combine strengths of two parents.
Your task is to create a high-quality offspring that combines the strengths of both parents by deeply analyzing their solutions.
Unlike a simple exchange of gene fragments, you should understand the semantics of the solutions, recognize patterns, and apply domain knowledge to reorganize them."""
    ques_block = json.dumps(task_mod.get_ques(), ensure_ascii = False, indent=2)
    user = dedent(
        f"""
        TASK   : {task_mod.describe()}
        QUESTION : {ques_block}
        PARENT-A (better): ```\n{parent_a}\n```
        PARENT-B        : ```\n{parent_b}\n```
        REFLECTION HINT : {reflection or 'N/A'}

        
        STEPS TO FOLLOW:
        1. Analyze: Identify the key features, strengths and weaknesses of each parent.
        2. Structural Understanding: Analyze the structure and patterns of the parent's solution, especially the successes.
        3. Strengths Extraction: Identify the reasons for the success of Parent A and the complementary elements in Parent B.
        4. Innovative reorganization: Combine the strengths of both while fixing possible weaknesses.
        5. Validation: Ensure that the generated offspring is valid and meets the mission constraints.


        Output JSON {{ "genome": "<fused-child>" }}.
        Child MUST compile / be valid for the task.
        """
    )
    return [{"role": "system", "content": sys},
            {"role": "user", "content": user}]


# --------------------------------------------------------------------------- #
# 4) Reflection prompt
# --------------------------------------------------------------------------- #
def build_reflection(
    population,
    task_name: str,
    top_n: int = 3,
    worst_n: int = 2,
) -> List[Dict[str, str]]:
    top = "\n---\n".join(str(ind.genome) for ind in population.best(top_n))
    worst = "\n---\n".join(str(ind.genome) for ind in population[-worst_n:])

    sys = "You are a self-reflection engine."
    user = dedent(
        f"""
        TASK  : {task_name}
        BEST  : \n{top}
        WORST : \n{worst}

        Analyse differences and output concise JSON:
        {{ "reflection": "3 key weaknesses & how to improve..." }}
        (< 60 tokens)
        """
    )
    return [{"role": "system", "content": sys},
            {"role": "user", "content": user}]


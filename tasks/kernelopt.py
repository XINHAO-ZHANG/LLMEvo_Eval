import os
import random

from kernelbench.dataset import construct_kernelbench_dataset
from kernelbench.eval import eval_kernel_against_ref, load_original_model_and_inputs
from kernelbench.utils import read_file, extract_first_code, create_inference_server_from_presets
from kernelbench.prompt_constructor import prompt_generate_custom_cuda_from_prompt_template
from llm_ops.api import call_llm

# ------------------------------ CONFIG ---------------------------------- #
LEVEL = 1
PROBLEM_ID = 1
GPU_ARCH = ["Ampere"]
GPU_NAME = "A100"
INIT_MODEL_NAME = "openai/gpt-4o-mini"
TEMPERATURE = 0.7
MAX_TOKENS = 8192

PROBLEM_STATEMENT = """\nYou write custom CUDA kernels to replace the pytorch operators in the given architecture to get speedups. \n
    You have complete freedom to choose the set of operators you want to replace. You may make the decision to replace some operators with custom CUDA kernels and leave others unchanged. You may replace multiple operators with custom implementations, consider operator fusion opportunities (combining multiple operators into a single kernel, for example, combining matmul+relu), or algorithmic changes (such as online softmax). You are only limited by your imagination.\n
"""
PROBLEM_INSTRUCTION = """
Optimize the architecture named Model with custom CUDA operators! Name your optimized output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Just output the new model code, no other text, and NO testing code! \n
"""

dataset = construct_kernelbench_dataset(LEVEL)
ref_arch_path = dataset[PROBLEM_ID-1]
problem_name = os.path.basename(ref_arch_path)
ref_arch_src = read_file(ref_arch_path)

# ---------- genome helpers ----------
def seed_pool(n: int, rng: random.Random):
    inference_server = create_inference_server_from_presets(server_type="openai",
                                                        model_name=INIT_MODEL_NAME,
                                                        temperature=TEMPERATURE,
                                                        max_tokens=MAX_TOKENS,
                                                        verbose=False,
                                                        time_generation=True)
    pool = []
    for _ in range(n):
        custom_cuda_prompt = prompt_generate_custom_cuda_from_prompt_template(ref_arch_src)
        custom_cuda = inference_server(custom_cuda_prompt)
        custom_cuda = extract_first_code(custom_cuda, ["python", "cpp"])
        pool.append(custom_cuda)
    return pool

def fitness(genome, verbose=False):
    # genome: 生成的代码字符串
    # ref_code: 参考实现
    result = eval_kernel_against_ref(
        ref_arch_src, genome, verbose=verbose, measure_performance=True, num_correct_trials=5, num_perf_trials=100
    )
    # 你可以根据 result 结构返回分数，比如用运行时间/正确性等
    # return result.get("score", 0.0)
    if not result.compiled:
        return "Compile Error"
    elif not result.correctness:
        return "Calculation Result Incorrect"
    else:
        return result.runtime

def repair(genomes):
    # 可选：修复不合法的代码
    # 这里只是示例，实际可根据需要实现
    return [g if isinstance(g, str) and len(g) > 0 else "// empty" for g in genomes]

def diversity_key(genome):
    # 可选：用前几行代码hash
    return hash(genome[:100])

# ------------------------- CONTEXT FOR LLM ------------------------------ #
def get_ques():
    return {"gpu_name": GPU_NAME, "gpu_arch": GPU_ARCH}

def describe():
    return PROBLEM_STATEMENT + PROBLEM_INSTRUCTION

def get_genome_desc():
    return "一个合法的基因为一段可运行的CUDA核函数代码（字符串）。"

if __name__ == "__main__":
    prompt = seed_pool(1, random.Random())
    result = fitness(prompt[0], ref_arch_src)
    print(result)

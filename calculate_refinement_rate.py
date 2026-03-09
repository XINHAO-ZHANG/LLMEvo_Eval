#!/usr/bin/env python3
"""
计算遗传算法中的 local refinement rate。
支持对 outputs 下不同 temp 的实验文件夹批量计算。
"""

import json
import re
from collections import defaultdict
from pathlib import Path

OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"


def load_generation_data(file_path):
    """加载每一代的数据"""
    generations = []
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                generations.append(data)
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                print(f"Line content (first 100 chars): {line[:100]}...")
                # Try to find where the JSON ends
                try:
                    # Find the position where the first valid JSON ends
                    decoder = json.JSONDecoder()
                    obj, idx = decoder.raw_decode(line)
                    generations.append(obj)
                    print(f"Successfully parsed partial JSON from line {line_num}")
                except Exception as e2:
                    print(f"Failed to parse even partial JSON: {e2}")
                    continue
    return generations

def find_parent_scores(parent_genome, population):
    """在population中找到对应parent genome的score"""
    for individual in population:
        if individual['genome'] == parent_genome:
            return individual['score']
    return None

def calculate_local_refinement_rate(generations, verbose=False):
    """计算 local refinement rate。verbose 为 True 时打印每一代与每个 offspring 的详情。"""
    total_refinements = 0
    total_offspring = 0
    generation_stats = []

    for gen_idx, current_gen in enumerate(generations):
        if gen_idx == 0:
            continue
        if "parent_lineage" not in current_gen or not current_gen["parent_lineage"]:
            continue

        prev_gen = generations[gen_idx - 1]
        prev_population = prev_gen["population"]
        children = current_gen["children"]
        child_scores = current_gen["child_scores"]
        parent_lineage = current_gen["parent_lineage"]

        gen_refinements = 0
        gen_offspring = len(children)

        if verbose:
            print(f"\n=== Generation {current_gen['gen']} ===")
            print(f"Offspring count: {gen_offspring}")

        for i, (child, child_score, parents) in enumerate(zip(children, child_scores, parent_lineage)):
            if not parents:
                continue
            parent_scores = []
            for parent_genome in parents:
                parent_score = find_parent_scores(parent_genome, prev_population)
                if parent_score is not None:
                    parent_scores.append(parent_score)
            if not parent_scores:
                if verbose:
                    print(f"  Offspring {i}: No valid parent scores found")
                continue
            min_parent_score = min(parent_scores)
            is_refinement = child_score < min_parent_score
            if verbose:
                print(f"  Offspring {i}: child_score={child_score}, parent_scores={parent_scores}, min_parent={min_parent_score}, refinement={is_refinement}")
            if is_refinement:
                gen_refinements += 1

        total_refinements += gen_refinements
        total_offspring += gen_offspring
        gen_refinement_rate = gen_refinements / gen_offspring if gen_offspring > 0 else 0
        if verbose:
            print(f"Generation {current_gen['gen']} refinement rate: {gen_refinements}/{gen_offspring} = {gen_refinement_rate:.4f}")
        generation_stats.append({
            "generation": current_gen["gen"],
            "refinements": gen_refinements,
            "offspring": gen_offspring,
            "refinement_rate": gen_refinement_rate,
        })

    overall_refinement_rate = total_refinements / total_offspring if total_offspring > 0 else 0
    return overall_refinement_rate, generation_stats, total_refinements, total_offspring


def find_experiment_dirs(outputs_dir: Path):
    """在 outputs 下查找所有包含 gen_log.jsonl 的实验目录。"""
    if not outputs_dir.is_dir():
        return []
    dirs = []
    for d in outputs_dir.iterdir():
        if d.is_dir() and (d / "gen_log.jsonl").is_file():
            dirs.append(d)
    return sorted(dirs)


def get_best_curve_min(exp_dir: Path):
    """读取 stats.json 中 best_curve 的最小值，若无则返回 None。"""
    stats_file = exp_dir / "stats.json"
    if not stats_file.is_file():
        return None
    try:
        with open(stats_file) as f:
            stats = json.load(f)
        curve = stats.get("best_curve")
        if curve is None or not isinstance(curve, list):
            return None
        return min(curve)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def parse_experiment_dir_name(dir_name: str):
    """
    解析目录名，提取 task_label, temp, seed, ts。
    格式: {task_label}_temp{temp_str}_{seed}_{ts}，例如 tsp30_temp0p5_21_1771350319
    """
    m = re.match(r"^(.+)_temp([^_]+)_(\d+)_(\d+)$", dir_name)
    if not m:
        return None
    task_label, temp_str, seed, ts = m.groups()
    # temp_str 如 0p5 -> 0.5, 1p1 -> 1.1
    try:
        temp = float(temp_str.replace("p", ".").replace("m", "-"))
    except ValueError:
        temp = temp_str
    return {"task_label": task_label, "temp": temp, "temp_str": temp_str, "seed": seed, "ts": ts}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="计算 outputs 下各 temp 实验的 local refinement rate")
    parser.add_argument(
        "outputs_dir",
        nargs="?",
        default=str(OUTPUTS_DIR),
        help="实验输出根目录，默认: outputs",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="打印每一代与每个 offspring 的详情")
    parser.add_argument("--single", type=str, metavar="PATH", help="仅对单个 gen_log.jsonl 或实验目录计算")
    args = parser.parse_args()

    if args.single:
        path = Path(args.single)
        if path.is_dir():
            gen_log = path / "gen_log.jsonl"
            if not gen_log.is_file():
                print(f"目录下无 gen_log.jsonl: {path}")
                return
            file_path = gen_log
            label = path.name
        elif path.is_file():
            file_path = path
            label = path.parent.name if path.name == "gen_log.jsonl" else path.stem
        else:
            print(f"不存在: {path}")
            return
        print(f"Loading: {file_path}")
        generations = load_generation_data(file_path)
        print(f"Loaded {len(generations)} generations for {label}\n")
        overall_rate, gen_stats, total_ref, total_off = calculate_local_refinement_rate(generations, verbose=args.verbose)
        print("=" * 50)
        print(f"SUMMARY — {label}")
        print("=" * 50)
        for stat in gen_stats:
            print(f"Generation {stat['generation']}: {stat['refinements']}/{stat['offspring']} = {stat['refinement_rate']:.4f}")
        print(f"\nOverall Local Refinement Rate: {total_ref}/{total_off} = {overall_rate:.4f} ({overall_rate * 100:.2f}%)")
        return

    outputs_dir = Path(args.outputs_dir)
    experiment_dirs = find_experiment_dirs(outputs_dir)
    if not experiment_dirs:
        print(f"在 {outputs_dir} 下未找到包含 gen_log.jsonl 的实验目录。")
        return

    print(f"在 {outputs_dir} 下找到 {len(experiment_dirs)} 个实验目录\n")
    results = []

    REF_BASE = 290  # 归一化 refinement 用的分母（refinement/290）

    for exp_dir in experiment_dirs:
        gen_log = exp_dir / "gen_log.jsonl"
        generations = load_generation_data(gen_log)
        parsed = parse_experiment_dir_name(exp_dir.name)
        temp_label = f"temp={parsed['temp']}" if parsed else exp_dir.name
        overall_rate, gen_stats, total_ref, total_off = calculate_local_refinement_rate(generations, verbose=args.verbose)
        best_min = get_best_curve_min(exp_dir)
        ref_per_base = total_ref / REF_BASE if REF_BASE else 0
        results.append({
            "dir": exp_dir.name,
            "parsed": parsed,
            "temp_label": temp_label,
            "overall_rate": overall_rate,
            "total_refinements": total_ref,
            "total_offspring": total_off,
            "n_generations": len(gen_stats),
            "best_curve_min": best_min,
            "ref_per_290": ref_per_base,
        })
        if args.verbose:
            print(f"  [{exp_dir.name}] refinement rate = {overall_rate:.4f}\n")

    # 按温度排序
    def sort_key(r):
        p = r.get("parsed")
        if p is None:
            return (0, r["dir"])
        return (p["temp"], r["dir"])

    results.sort(key=sort_key)

    print("=" * 90)
    print("LOCAL REFINEMENT RATE — 各 temp 实验汇总")
    print("=" * 90)
    print(f"{'实验目录':<38} {'温度':<6} {'best_min':<10} {'ref_rate':<14} {'ref/290':<10} {'ref/offspring'}")
    print("-" * 90)
    for r in results:
        p = r["parsed"]
        temp_str = f"{p['temp']}" if p else "-"
        best_str = str(r["best_curve_min"]) if r["best_curve_min"] is not None else "-"
        ref_off = f"{r['total_refinements']}/{r['total_offspring']}"
        print(f"{r['dir']:<38} {temp_str:<6} {best_str:<10} {r['overall_rate']:.4f} ({r['overall_rate']*100:.1f}%)  {r['ref_per_290']:.4f}    {ref_off}")
    print("-" * 90)
    print(f"共 {len(results)} 个实验。ref/290 = refinement数 / 290。")

    # 按温度分组求平均
    by_temp = defaultdict(list)
    for r in results:
        p = r.get("parsed")
        if p is not None:
            by_temp[p["temp"]].append(r)
    if by_temp:
        print()
        print("=" * 70)
        print("按温度平均 (每个温度下所有实验的均值)")
        print("=" * 70)
        print(f"{'温度':<10} {'实验数':<8} {'avg(best_min)':<16} {'avg(ref_rate)':<14} {'avg(ref/290)'}")
        print("-" * 70)
        for temp in sorted(by_temp.keys()):
            group = by_temp[temp]
            n = len(group)
            best_vals = [r["best_curve_min"] for r in group if r["best_curve_min"] is not None]
            avg_best = sum(best_vals) / len(best_vals) if best_vals else None
            avg_rate = sum(r["overall_rate"] for r in group) / n
            avg_ref290 = sum(r["ref_per_290"] for r in group) / n
            avg_best_str = f"{avg_best:.1f}" if avg_best is not None else "-"
            print(f"{temp:<10} {n:<8} {avg_best_str:<16} {avg_rate:.4f} ({avg_rate*100:.1f}%)   {avg_ref290:.4f}")
        print("-" * 70)


if __name__ == "__main__":
    main()
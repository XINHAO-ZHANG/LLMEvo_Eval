#!/usr/bin/env python3
"""
测试新的并行化配置是否正常工作
"""

import tempfile
import time
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

from omegaconf import DictConfig
from scripts.run_exp import load_task
from evolve_core.loop import run_evolve

def test_parallel_config():
    """测试并行配置在实际任务中的工作"""
    
    print("Testing parallel configuration with real tasks...")
    
    # 测试配置
    test_configs = [
        {
            "task": "tsp",
            "max_workers": 2,
            "child_slots": 12,
            "budget": 24  # 2代，每代4个子代
        },
        {
            "task": "promptopt",
            "max_workers": 1,
            "child_slots": 2,
            "budget": 4
        }
    ]
    
    for i, test_cfg in enumerate(test_configs):
        print(f"\n--- Test {i+1}: {test_cfg['task']} with max_workers={test_cfg['max_workers']} ---")
        
        try:
            # 加载任务模块
            task_mod = load_task(test_cfg["task"])
            
            # 配置任务专属参数
            if hasattr(task_mod, 'configure'):
                if test_cfg["task"] == "tsp":
                    task_mod.configure({"CITY_NUM": 10})  # 小规模测试
                elif test_cfg["task"] == "promptopt":
                    task_mod.configure({
                        "eval_task": "sum",
                        "eval_set": "test"
                    })
            
            # 配置对象
            cfg = DictConfig({
                "enable_zero_shot_eval": False,  # 跳过zero-shot加速测试
                "capacity": 20
            })
            
            start_time = time.time()
            
            # 运行进化循环
            with tempfile.TemporaryDirectory() as tmp_dir:
                stats = run_evolve(
                    cfg=cfg,
                    task_mod=task_mod,
                    model_name="openai/gpt-4o-mini",  # 会失败但可以测试并行机制
                    seed=42,
                    n_init=test_cfg.get("n_init", 5),
                    n_parent=test_cfg.get("parent_slots", 2),
                    n_child=test_cfg["child_slots"],
                    budget_calls=test_cfg["budget"],
                    db_mode="simple",
                    db_kwargs={"capacity": 20},
                    max_workers=test_cfg["max_workers"],
                    out_dir=Path(tmp_dir) / "test_output"
                )
            
            duration = time.time() - start_time
            print(f"✅ Task completed in {duration:.2f}s")
            print(f"   Calls made: {stats.calls}")
            print(f"   Best score: {stats.best_curve[-1] if stats.best_curve else 'N/A'}")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            # 不打印完整traceback，因为我们预期会有一些LLM调用失败

if __name__ == "__main__":
    test_parallel_config()

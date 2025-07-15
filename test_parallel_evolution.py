#!/usr/bin/env python3
"""
测试并行化进化循环的性能改进
"""

import time
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

# 模拟任务模块
class MockTask:
    def __init__(self):
        self.eval_count = 0
    
    def __name__(self):
        return "mock_task"
    
    def seed_pool(self, n_init, rng):
        """生成初始种群"""
        for i in range(n_init):
            yield MockGenome(f"initial_{i}", rng.random())
    
    def get_evolve_prompt(self, parents):
        return "Generate a new solution based on these parents: " + str([p.genome for p in parents])
    
    def parse_response(self, response):
        # 模拟解析LLM响应
        return {"genome": f"solution_{time.time()}_{self.eval_count}"}
    
    def eval(self, genome):
        # 模拟评估过程（稍微耗时）
        self.eval_count += 1
        time.sleep(0.1)  # 模拟计算时间
        return abs(hash(genome) % 1000) / 1000.0  # 返回0-1之间的分数
    
    def get_zero_shot_prompt(self, **kwargs):
        return "Solve this problem directly without examples."

# 模拟LLM API调用
def mock_call_llm(prompt, model, max_tokens, seed):
    # 模拟API延迟
    time.sleep(0.05)  
    return {
        "choices": [{"message": {"content": f"Generated solution for {model}"}}],
        "usage": {"total_tokens": 100}
    }

# 模拟数据库
class MockGenome:
    def __init__(self, genome, loss, extra=None):
        self.genome = genome
        self.loss = loss
        self.extra = extra or {}

class MockDB:
    def __init__(self):
        self.pool = []
        self.best_score = float('inf')
    
    def sample(self, n):
        if len(self.pool) < n:
            # 初始化一些虚拟父代
            for i in range(n):
                self.pool.append(MockGenome(f"parent_{i}", i * 0.1))
        return self.pool[:n]
    
    def add(self, genomes):
        for g in genomes:
            self.pool.append(g)
            if g.loss < self.best_score:
                self.best_score = g.loss
    
    def get_best(self):
        if not self.pool:
            return float('inf')
        return min(g.loss for g in self.pool)

def test_parallel_performance():
    """测试并行化的性能提升"""
    print("Testing parallel evolution performance...")
    
    # 替换实际模块以进行测试
    import llm_ops.api
    llm_ops.api.call_llm = mock_call_llm
    
    from evolve_core.loop import run_evolve
    from evolve_core.db import get_db, Genome
    
    # 创建模拟任务
    task = MockTask()
    
    # 测试配置
    test_config = {
        "enable_zero_shot_eval": False  # 跳过zero-shot评估以专注于进化循环
    }
    
    # 测试参数
    n_child = 8
    n_generations = 3
    budget_calls = n_child * n_generations
    
    print(f"Testing with {n_child} children per generation, {n_generations} generations")
    print(f"Total budget: {budget_calls} calls")
    
    # 测试不同的并行度
    for max_workers in [1, 2, 4]:
        print(f"\n--- Testing with max_workers={max_workers} ---")
        
        # 重置任务计数器
        task.eval_count = 0
        
        start_time = time.time()
        
        try:
            # 使用模拟数据库
            class MockDBProvider:
                def __init__(self):
                    pass
                def __call__(self, mode, task_mod, **kwargs):
                    db = MockDB()
                    db.init = lambda n, rng: None  # 模拟初始化
                    db.from_json = lambda path: None
                    db.to_json = lambda path: None
                    return db
            
            # 临时替换get_db
            original_get_db = get_db
            
            def mock_get_db(mode, task_mod, **kwargs):
                return MockDB()
            
            import evolve_core.db
            evolve_core.db.get_db = mock_get_db
            
            # 运行进化
            stats = run_evolve(
                cfg=test_config,
                task_mod=task,
                model_name="mock_model",
                seed=42,
                n_child=n_child,
                budget_calls=budget_calls,
                max_workers=max_workers,
                out_dir=Path("/tmp/test_parallel_evo")
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            print(f"Duration: {duration:.2f}s")
            print(f"Evaluations performed: {task.eval_count}")
            print(f"Calls recorded in stats: {stats.calls}")
            print(f"Throughput: {stats.calls/duration:.2f} calls/second")
            
            # 恢复原始函数
            evolve_core.db.get_db = original_get_db
            
        except Exception as e:
            print(f"Error during test: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_parallel_performance()

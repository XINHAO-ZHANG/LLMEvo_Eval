#!/usr/bin/env python3
"""
简化的并行化性能测试
"""

import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
sys.path.append(str(Path(__file__).parent))

def simulate_child_generation():
    """模拟单个子代生成过程"""
    # 模拟LLM调用延迟
    time.sleep(0.05)  # 50ms
    # 模拟评估延迟  
    time.sleep(0.1)   # 100ms
    return f"child_{time.time()}"

def test_serial_vs_parallel():
    """比较串行和并行的性能差异"""
    n_children = 8
    
    print(f"Testing generation of {n_children} children")
    print("=" * 50)
    
    # 测试串行
    print("Serial execution:")
    start_time = time.time()
    serial_results = []
    for i in range(n_children):
        result = simulate_child_generation()
        serial_results.append(result)
    serial_time = time.time() - start_time
    print(f"Time taken: {serial_time:.2f}s")
    print(f"Results: {len(serial_results)} children")
    
    print("\n" + "-" * 30 + "\n")
    
    # 测试并行 (不同的worker数量)
    for max_workers in [2, 4, 8]:
        print(f"Parallel execution (max_workers={max_workers}):")
        start_time = time.time()
        
        parallel_results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(simulate_child_generation) for _ in range(n_children)]
            for future in as_completed(futures):
                result = future.result()
                parallel_results.append(result)
        
        parallel_time = time.time() - start_time
        speedup = serial_time / parallel_time
        
        print(f"Time taken: {parallel_time:.2f}s")
        print(f"Results: {len(parallel_results)} children")
        print(f"Speedup: {speedup:.1f}x")
        
        if max_workers < 8:
            print()

if __name__ == "__main__":
    test_serial_vs_parallel()

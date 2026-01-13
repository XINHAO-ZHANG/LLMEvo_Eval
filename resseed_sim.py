"""
重新seed simplification任务的初始种群
"""

import sys
import random
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from tasks.promptopt import seed_pool, configure, clear_eval_cache

def main():
    """重新评估simplification任务的初始种群"""
    
    # 配置任务为simplification
    class Config:
        eval_task = "sim"
    
    configure(Config())
    
    # # 清空评估缓存，确保重新评估
    # clear_eval_cache()
    
    # 创建随机数生成器
    rng = random.Random(42)  # 固定种子确保可重复
    
    print("开始重新评估simplification任务的初始种群...")
    print("=" * 60)
    
    try:
        # 重新seed种群，这会触发重新评估
        genomes = seed_pool(
            n=10,  # 评估10个初始prompt
            rng=rng,
            prompt_task="sim",
            eval_set="test"
        )
        
        print("\n" + "=" * 60)
        print("评估完成！结果如下：")
        print("=" * 60)
        
        # 按性能排序（loss越小越好）
        genomes.sort(key=lambda g: g.loss)
        
        for i, genome in enumerate(genomes):
            print(f"\n第 {i+1} 名 (Loss: {genome.loss:.6f}):")
            print(f"Prompt: {genome.genome}")
            print("-" * 40)
        
        print(f"\n最佳性能: {genomes[0].loss:.6f}")
        print(f"最差性能: {genomes[-1].loss:.6f}")
        print(f"平均性能: {sum(g.loss for g in genomes) / len(genomes):.6f}")
        
    except Exception as e:
        print(f"评估过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
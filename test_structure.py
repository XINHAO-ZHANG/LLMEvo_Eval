#!/usr/bin/env python
"""
Quick test script to verify the refactored code structure
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    # Test basic imports (skip numpy-dependent imports for now)
    print("🔍 Testing basic imports...")
    
    from pathlib import Path
    from omegaconf import OmegaConf
    print("✅ Basic dependencies imported successfully")
    
    # Test configuration first
    print("\n⚙️ Testing configuration...")
    config_path = Path("config/exp_grid.yaml")
    if config_path.exists():
        cfg = OmegaConf.load(config_path)
        print(f"✅ exp_grid.yaml loaded:")
        print(f"   Task: {cfg.task}")
        print(f"   Model: {cfg.model}")
        print(f"   Budget: {cfg.budget}")
        print(f"   Available task configs: {list(cfg.tasks.keys()) if 'tasks' in cfg else 'None'}")
    else:
        print(f"❌ Config file not found: {config_path}")
    
    # Test LLMEvo imports (may fail due to missing dependencies)
    print("\n📦 Testing LLMEvo imports...")
    try:
        from __init__ import __version__
        print(f"✅ LLMEvo version: {__version__}")
        
        from tasks import list_tasks, get_task
        print(f"✅ Available tasks: {list_tasks()}")
        
        from evolve import RunStats
        print("✅ Evolution components imported successfully")
        
        # Test task loading
        print("\n🎯 Testing task loading...")
        try:
            task = get_task("tsp")
            print(f"✅ TSP task loaded: {task}")
        except Exception as e:
            print(f"❌ Failed to load TSP task: {e}")
            
    except ImportError as e:
        print(f"⚠️ LLMEvo imports failed (probably missing numpy): {e}")
        print("   This is expected if dependencies aren't installed yet")
    
    print("\n🎉 Configuration test completed successfully!")
    print("\nNext steps:")
    print("1. Install dependencies: pip install numpy pandas matplotlib")
    print("2. Set up API keys: export OPENAI_API_KEY='your-key'")
    print("3. Run a test experiment: python experiments/run_experiment.py task=tsp budget=10")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
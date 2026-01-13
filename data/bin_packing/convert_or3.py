import pickle
import sys
from pathlib import Path

# 添加项目根目录到路径


# 导入OR3数据集
from or_dataset import datasets

def convert_or3_to_pickle():
    """将OR3数据集转换为pickle格式"""
    
    # 检查数据集是否存在
    if 'OR3' not in datasets:
        print("Error: OR3 dataset not found in or_dataset.py")
        return
    
    or3_data = datasets['OR3']
    print(f"Found OR3 dataset with {len(or3_data)} instances")
    
    # 输出文件路径
    output_file = "OR3_dataset.pkl"
    
    
    # 保存为pickle文件
    with open(output_file, 'wb') as f:
        pickle.dump(or3_data, f)
    
    print(f"Successfully converted OR3 dataset to: {output_file}")
    
    # 验证转换结果
    print("\nDataset verification:")
    for instance_name, instance_data in list(or3_data.items())[:3]:  # 只显示前3个实例
        print(f"  {instance_name}:")
        print(f"    Capacity: {instance_data['capacity']}")
        print(f"    Items: {instance_data['num_items']}")
        print(f"    First 10 items: {instance_data['items'][:10]}")
        print(f"    Total weight: {sum(instance_data['items'])}")
        print(f"    L1 lower bound: {sum(instance_data['items']) / instance_data['capacity']:.2f}")
        print()

def test_loading():
    """测试加载转换后的数据集"""
    print("Testing dataset loading...")
    
    dataset_path =  "OR3_dataset.pkl"
    
    try:
        with open(dataset_path, 'rb') as f:
            loaded_data = pickle.load(f)
        
        print(f"Successfully loaded dataset with {len(loaded_data)} instances")
        
        # 验证数据结构
        first_instance = list(loaded_data.values())[0]
        required_keys = ['capacity', 'num_items', 'items']
        
        for key in required_keys:
            if key in first_instance:
                print(f"✓ Found required key: {key}")
            else:
                print(f"✗ Missing required key: {key}")
        
        return True
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return False

if __name__ == "__main__":
    print("Converting OR3 dataset to pickle format...")
    convert_or3_to_pickle()
    
    print("\n" + "="*50)
    test_loading()
    
    print("\n" + "="*50)
    print("Conversion completed! You can now use OR3 dataset in bin_packing tasks.")
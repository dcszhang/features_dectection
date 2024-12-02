import pandas as pd

# 重新定义数据和操作路径
original_pkl_path = "dataset/bytecode_subset.pkl"  # 原始 pkl 文件路径
cleaned_pkl_path = "dataset/cleaned_bytecode.pkl"  # 保存清理后的文件路径

# 加载原始 pkl 文件并清理 "0x"
try:
    # 读取原始 .pkl 文件
    bytecode_list = pd.read_pickle(original_pkl_path)
    
    # 去掉每个 bytecode 前的 "0x"
    cleaned_bytecode_list = [bc[2:] if bc.startswith("0x") else bc for bc in bytecode_list]
    
    # 保存清理后的数据
    pd.to_pickle(cleaned_bytecode_list, cleaned_pkl_path)
    
    # 打印清理后前几行数据
    print("清理后的数据预览（前5行）：")
    for i, bytecode in enumerate(cleaned_bytecode_list[:5]):
        print(f"{i + 1}: {bytecode}")

    print(f"清理后的数据已保存到 {cleaned_pkl_path}")
except Exception as e:
    print(f"清理文件时出错: {e}")

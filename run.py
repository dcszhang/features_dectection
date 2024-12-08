import pickle
import subprocess
import os

def load_bytecode(filepath):
    """加载cleaned_bytecode.pkl文件，返回字节码列表"""
    with open(filepath, 'rb') as file:
        bytecode_list = pickle.load(file)
    if not isinstance(bytecode_list, list):
        raise ValueError("数据格式错误，cleaned_bytecode.pkl应该包含一个列表！")
    return bytecode_list

def write_bytecode_to_file(bytecode, filepath):
    """将字节码写入bytecode文件"""
    with open(filepath, 'w') as file:
        file.write(bytecode)

def run_command():
    """运行命令并打印输出"""
    try:
        result = subprocess.run(['python3', 'rattle-cli.py', '--input', 'bytecode'], 
                                capture_output=True, text=True, check=True)
        print("运行结果:")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("命令运行失败:")
        print(e.stderr)

def main():
    # 文件路径
    pkl_path = 'dataset/cleaned_bytecode.pkl'  # 修改为实际的路径
    bytecode_file_path = 'bytecode'  # 输出字节码文件路径

    if not os.path.exists(pkl_path):
        print(f"文件 {pkl_path} 不存在！请检查路径。")
        return

    # 加载字节码
    try:
        bytecode_list = load_bytecode(pkl_path)
    except Exception as e:
        print(f"加载字节码失败: {e}")
        return

    # 遍历运行字节码
    for idx, bytecode in enumerate(bytecode_list):
        print(f"运行第 {idx + 1} 个字节码...")
        try:
            write_bytecode_to_file(bytecode, bytecode_file_path)
            run_command()
        except Exception as e:
            print(f"处理第 {idx + 1} 个字节码时出错: {e}")

if __name__ == "__main__":
    main()
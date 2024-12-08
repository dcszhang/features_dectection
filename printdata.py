import pickle

# 加载 .pkl 文件
file_path = 'dataset/cleaned_bytecode.pkl'
with open(file_path, 'rb') as file:
    data = pickle.load(file)

# 打印内容类型和字段信息
print("数据类型:", type(data))
# 打印前几个元素
for i in range(5):  # 打印前 5 个元素
    print(f"元素 {i+1}: {data[i]}")

# 如果是字典，打印键值
if isinstance(data, dict):
    print("字段:", data.keys())
elif isinstance(data, list):
    print(f"列表包含 {len(data)} 个元素。第一个元素类型为: {type(data[0])}")
    if len(data) > 0 and isinstance(data[0], dict):
        print("字段:", data[0].keys())
else:
    print("内容:", data)

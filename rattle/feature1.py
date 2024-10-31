


def backward_analysis(variable, all_instructions_by_variable, visited=None):
    """
    通过变量名直接查找定义，依据操作数进行递归回溯。
    param variable: SSA中的变量（如 %1504）
    param all_instructions_by_variable: 包含所有指令的字典，键为变量名，值为指令
    param visited: 已经访问过的变量集合，避免重复回溯
    return: 返回包含所有回溯指令的列表
    """
    if visited is None:
        visited = set()

    trace = []  # 用于存储回溯路径中的所有指令
    variables_to_trace = [variable]  # 初始化要追踪的变量列表

    while variables_to_trace:
        current_variable = variables_to_trace.pop(0)

        # 如果已经访问过该变量，则跳过，防止死循环
        if current_variable in visited:
            continue

        visited.add(current_variable)  # 将当前变量标记为已访问
                # 如果当前变量是常量，直接跳过回溯
        if str(current_variable).startswith('#'):
            # print(f"Constant value encountered: {current_variable}")
            continue
        # 查找该变量的定义
        if current_variable in all_instructions_by_variable:
            insn = all_instructions_by_variable[current_variable]
            trace.append(insn)  # 记录当前指令
            # print(f"Found definition of {current_variable}: {insn}")

            # 对该指令的操作数进行回溯
            for argument in insn.arguments:
                if argument not in visited:  # 防止重复追踪
                    variables_to_trace.append(argument)
        else:
            trace.append(f"No definition found for variable {current_variable}")
    # 保存路径到全局变量
    save_trace_to_global(variable, trace)
    return trace


def build_tree_structure(variable, all_instructions_by_variable, visited=None):
    """
    从地址变量开始构建树结构，并将其保存到全局变量中。
    param variable: 当前地址变量
    param all_instructions_by_variable: 包含所有指令的字典，键为变量名，值为指令
    param visited: 已经访问过的变量集合，避免重复回溯
    return: 树的根节点
    """
    if visited is None:
        visited = set()

    tree_root = TreeNode(variable)  # 创建树的根节点
    variables_to_trace = [(variable, tree_root)]  # 初始化要追踪的变量列表，包含树节点

    while variables_to_trace:
        current_variable, current_node = variables_to_trace.pop(0)

        # 如果已经访问过该变量，则跳过，防止死循环
        if current_variable in visited:
            continue

        visited.add(current_variable)  # 将当前变量标记为已访问

        # 如果当前变量是常量，直接跳过回溯
        if str(current_variable).startswith('#'):
            continue

        # 查找该变量的定义
        if current_variable in all_instructions_by_variable:
            insn = all_instructions_by_variable[current_variable]

            # 对该指令的操作数进行回溯，并构建子节点
            for argument in insn.arguments:
                if argument not in visited:
                    child_node = TreeNode(argument)  # 为操作数创建子节点
                    current_node.add_child(child_node)  # 将子节点加入当前节点
                    variables_to_trace.append((argument, child_node))  # 将子节点加入追踪队列

    # 保存树到全局变量
    save_tree_to_global(variable, tree_root)
    return tree_root  # 返回树的根节点


def analyze_saved_traces(address_variable, all_instructions_by_variable):
    """
    遍历全局变量中保存的所有路径，查找与字符串操作相关的指令，并判断是否匹配第一个特征。
    param address_variable: 当前的地址变量
    """
    # 遍历 all_trace_paths 中的所有保存路径
    for variable, trace in all_trace_paths.items():
        print(f"\t\t\tAnalyzing trace for variable: {variable}")
        
        has_complex_calculation = False
        has_dynamic_input = False
        has_external_call = False
        depth_threshold=5
        for insn in trace:
            # 检查是否是字符串操作指令
            if insn.insn.name in ['MLOAD', 'CALLDATALOAD']:
                has_dynamic_input = True
                # print(f"\t\t\t\tFound dynamic input operation at {insn}")
            elif insn.insn.name == 'ADD':
                has_complex_calculation = True
                # print(f"\t\t\t\tFound complex calculation operation at {insn}")
            elif insn.insn.name in ['CALL', 'DELEGATECALL']:
                has_external_call = True
                # print(f"\t\t\t\tFound external call operation at {insn}")
        # 计算树的层数
        tree_root = all_trees.get(variable)
        if tree_root:
            tree_height = tree_root.get_height()
            print(f"\t\t\t\tTree height for variable {variable}: {tree_height}")
        else:
            tree_height = 0  # 如果没有树，就设置高度为0
        # 匹配条件：(1) + (2) 或者 (1) + (3) 或者 树的高度超过阈值
        if (has_complex_calculation and has_dynamic_input) or (has_complex_calculation and has_external_call)or (tree_height > depth_threshold):
            print(f"\t\t\t[+] Trace for variable {variable} matches the first feature (complex address generation)")
        else:
            print(f"\t\t\t[-] Trace for variable {variable} does not match the first feature")


def save_tree_to_global(variable, tree):
    """
    保存树结构到全局变量
    param variable: 地址变量
    param tree: 树结构
    """
    global all_trees
    all_trees[variable] = tree

class TreeNode:
    def __init__(self, variable):
        self.variable = variable  # 当前节点对应的变量
        self.children = []  # 子节点列表

    def add_child(self, child_node):
        """将子节点添加到当前节点"""
        self.children.append(child_node)

    def __repr__(self):
        """便于调试，打印树节点的变量名"""
        return f"TreeNode({self.variable})"
    
    def print_tree(self, level=0, prefix=""):
            """递归打印整个树结构，使用更加直观的树形格式"""
            # 打印当前节点
            print(f"{prefix}{self.variable}")
            # 遍历子节点，调整前缀以显示树的结构
            for i, child in enumerate(self.children):
                # 如果是最后一个子节点，使用 "└──"，否则使用 "├──"
                if i == len(self.children) - 1:
                    child_prefix = prefix + "    "  # 对齐格式
                    child.print_tree(level + 1, prefix + "└── ")
                else:
                    child_prefix = prefix + "│   "  # 对齐格式
                    child.print_tree(level + 1, prefix + "├── ")
    def get_height(self):
            """递归计算树的高度"""
            if not self.children:
                return 1  # 如果没有子节点，树的高度为1
            return 1 + max(child.get_height() for child in self.children)
# Third_feature.py
def analyze_contract_external(ssa):
    """
    分析合约是否存在外部调用控制的特征
    """
    # 存储所有指令
    all_instructions_by_variable = {}
    # 定义address、value变量来保存所有路径
    all_trace_paths = {}
    # 遍历所有函数并保存指令
    for function in ssa.functions:
        for block in function:
            for insn in block:
                if hasattr(insn, 'return_value') and insn.return_value is not None:
                    variable = insn.return_value  # 获取指令的返回值变量
                    all_instructions_by_variable[variable] = insn  # 将变量和对应的指令保存到字典中

    can_send, functions_that_can_send = ssa.can_send_ether()
    if can_send:
        for function in functions_that_can_send:
            # 初始化每个函数的追踪列表
            # print(f"\t- {function.desc()}")
            _, insns = function.can_send_ether()
            for insn in insns:
                # print(f"\t\t{insn}")
                if insn.insn.name == 'SELFDESTRUCT':
                    address = insn.arguments[0]
                    # print(f'\t\t\t{address.writer}')
                elif insn.insn.name == 'CALL':
                    address = insn.arguments[1]
                    value = insn.arguments[2]
                    # 回溯分析，记录沿途的指令路径
                    trace_address= backward_analysis(address, all_instructions_by_variable)
                    trace_value = backward_analysis(value, all_instructions_by_variable)

                    # 保存结果
                    all_trace_paths[address] = trace_address
                    all_trace_paths[value] = trace_value
        # 分析路径
        results = analyze_saved_traces(all_trace_paths)
        true_or_false = False
        for result in results:
            # print(f"Variable {result['variable']} has external call: {result['has_external_call']}")
            if result['has_external_call']:
                true_or_false = True
        if true_or_false:
            print("Consistent with Scam feature 3.")
        else:
            print("Not consistent with Scam feature 3.")
            

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
    return trace


def analyze_saved_traces(all_trace_paths):
    """
    分析所有保存的路径，检查是否存在外部调用
    :param address_variable: 当前变量
    :param all_trace_paths: 所有回溯路径
    :return: 路径分析结果
    """
    results = []
    for variable, trace in all_trace_paths.items():
        # print(f"Analyzing trace for variable: {variable}")
        has_external_call = False
        for insn in trace:
            if insn.insn.name in ['CALL', 'DELEGATECALL']:
                # print(f"Found external call at {insn}")
                has_external_call = True
        results.append({
            "variable": variable,
            "has_external_call": has_external_call
        })
    return results

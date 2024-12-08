
import rattle
import os
import tempfile
def process_fourth_feature(ssa, threshold=0.1):
    """
    处理第四个特征：隐藏在复杂代码之后的转账操作。
    通过统计 CALL 指令的相关指令数与所在函数的总指令数的占比来判断。
    并对剩余指令检查是否存在外部调用或状态修改。
    :param ssa: 静态单赋值形式的分析对象
    :param threshold: 占比阈值，低于该值标记为 Scam 特征
    :return: 分析结果
    """
    instruction_sequences = {}  # 存储所有函数的指令序列

    # 遍历所有函数，构建 CFG 和 PDG
    for function in sorted(ssa.functions, key=lambda f: f.offset):
        cfg = rattle.ControlFlowGraph(function)
        pdg = rattle.ProgramDependenceGraph(function)
        instruction_sequences[function] = sum(len(block) for block in function)  # 统计函数指令总数
        # print(f"Function: {function.desc()}")
        # print(f"\tTotal instructions: {instruction_sequences[function]}")
    # 创建 SystemDependenceGraph 实例
    sdg = rattle.SystemDependenceGraph(ssa.functions)

    # 使用 CallBacktracking 对 SDG 中的函数进行回溯
    call_backtracking = rattle.CallBacktracking(sdg)

    # 分析 CALL 指令相关指令与函数指令总数的占比
    for func, paths in call_backtracking.get_backtrack_results().items():
        function_total_instructions = instruction_sequences[func]  # 获取BS后总指令数
        call_related_instructions = set()  # 存储与 CALL 相关的指令
        external_call_instructions = set()  # 外部调用相关指令
        for path in paths:
            # # 收集与 CALL 相关的指令
            call_related_instructions.update({insn for block in path for insn in block})

        # if(len(call_related_instructions) == 0):
        #     continue
        # 获取函数所有指令数目
        all_instructions = {insn for block in func for insn in block}

        # 获取去除 Transfer和外部调用相关指令后的剩余指令
        remaining_instructions = all_instructions - call_related_instructions
        
        # 获取去除CALL和 SSTORE 相关指令后的剩余指令
        result_original = process_sstore_recursive_analysis(ssa, remaining_instructions)
        result = result_original[0]
        remaining_instructions = result["remaining_instructions"]
        print(f"\nFunction: {func.desc()}")
        # print(f"Remaining instructions after SSTORE_&Transder_&CALL analysis: {len(remaining_instructions)}")

        # 计算比率
        total_related_instructions = len(all_instructions) - len(remaining_instructions)
        ratio = total_related_instructions / len(all_instructions)

        # 判断是否符合 Scam 特征
        is_scam_feature4 = ratio < threshold

        # 打印分析结果
        print(f"\tTotal instructions: {len(all_instructions)}")
        print(f"\tcall_related_instructions: ", len(call_related_instructions))
        print(f"\tSSTORE_related_instructions: ", len(all_instructions) - len(call_related_instructions) - len(remaining_instructions))
        print(f"\tTotal Useless instructions: {len(remaining_instructions)}")
        print(f"\tRelated Ratio: {ratio:.2%}")

    if(is_scam_feature4):
        print("Consistent with Scam feature 4.")
    else:
        print("Not consistent with Scam feature 4.")

def process_sstore_recursive_analysis(ssa, remaining_instructions):
    """
    主函数，用于分析 SSA 图中所有函数的 SSTORE 相关路径剥离。
    :param ssa: SSA 图对象
    """
    results = []

    # 构建所有指令的字典
    all_instructions_by_variable = {}    
    for func in ssa.functions:
        # 遍历所有函数并保存指令
        for block in func:
            for insn in block:
                # 统一处理 return_value 和 arguments
                for related_var in [insn.return_value] + (insn.arguments if hasattr(insn, 'arguments') else []):
                    if related_var is not None and related_var not in all_instructions_by_variable:
                        all_instructions_by_variable[related_var] = insn
    # print(f"Total variables with instructions: {len(all_instructions_by_variable)}")
    # 对剩余指令执行 SSTORE 分析
    updated_remaining_instructions = iterative_sstore_analysis(
        remaining_instructions, all_instructions_by_variable
    )
    # print(f"Remaining instructions after SSTORE analysis ({len(updated_remaining_instructions)}):")
    # for insn in updated_remaining_instructions:
    #     print(f"\t{insn}")

    results.append({
        "function": func.desc(),
        "remaining_instructions": updated_remaining_instructions,
    })

    return results

def iterative_sstore_analysis(remaining_instructions, all_instructions_by_variable):
    """
    对剩余代码中的 SSTORE 指令进行递归回溯分析，并逐步剔除相关路径的代码。
    :remaining_instructions: 当前分析的指令
    :param all_instructions_by_variable: 包含所有指令的字典，键为变量名，值为指令
    :return: 剩余未关联 SSTORE 的指令
    """
    print(f"Initial remaining instructions: {len(remaining_instructions)}")
    while True:
        # 提取当前剩余指令中的所有 SSTORE 指令
        sstore_instructions = [
            insn for insn in remaining_instructions
            if hasattr(insn, 'insn') and insn.insn.name == 'SSTORE'
        ]
        # print(f"Detected SSTORE instructions ({len(sstore_instructions)}):")
        if not sstore_instructions:
            # 如果没有 SSTORE 指令，退出循环
            break

        for insn in sstore_instructions:
            variable = insn.arguments[1]  # SSTORE 的目标变量(Value)
            # print(f"Analyzing SSTORE at {insn} for variable {variable}")

            # 对目标变量进行回溯分析
            trace = backward_analysis(variable, all_instructions_by_variable)

            # 从剩余指令中剔除回溯路径相关指令
            remaining_instructions -= set(trace)

        # 剔除已经分析过的 SSTORE 指令
        remaining_instructions -= set(sstore_instructions)

    return remaining_instructions





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
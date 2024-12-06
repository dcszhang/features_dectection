
import rattle
import os
import tempfile
def process_fourth_feature(ssa, threshold=0.1):
    """
    处理第四个特征：隐藏在复杂代码之后的转账操作。
    通过统计 CALL 指令的相关指令数与所在函数的总指令数的占比来判断。
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
    results = []
    true_or_false = False
    for func, paths in call_backtracking.get_backtrack_results().items():
        function_total_instructions = instruction_sequences[func]  # 获取函数总指令数

        # print(f"Function: {func.desc()}")
        # print(f"\tTotal instructions in function: {function_total_instructions}")

        for path in paths:
            # 统计路径中所有相关指令数
            related_instructions = sum(len(block) for block in path)

            # 计算相关指令数与函数总指令数的比值
            related_ratio = related_instructions / function_total_instructions

            # 判断是否符合 Scam 特征
            is_scam_feature = related_ratio < threshold

            # 打印分析结果
            # print(f"\tPath starting at block {path[0].offset:#x}:")
            # print(f"\t\tRelated instructions: {related_instructions}")
            # print(f"\t\tRelated ratio: {related_ratio:.2%}")
            if is_scam_feature:
                true_or_false = True

            # # 保存分析结果
            # results.append({
            #     "function": func.desc(),
            #     "path_start_block": path[0].offset,
            #     "related_instructions": related_instructions,
            #     "function_total_instructions": function_total_instructions,
            #     "related_ratio": related_ratio,
            #     "is_scam_feature": is_scam_feature,
            # })
    if(true_or_false):
        print("Consistent with Scam feature 4.")
    else:
        print("Not consistent with Scam feature 4.")

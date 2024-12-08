import rattle

from web3 import Web3
def process_fifth_feature(ssa):
    """
    检测 Feature 5：转账操作前后无关的日志输出。
    :param ssa: 静态单赋值形式的分析对象
    """
    results = []
    can_send, functions_that_can_send = ssa.can_send_ether()
    if can_send:
        for function in sorted(functions_that_can_send, key=lambda f: f.offset):
            _, insns = function.can_send_ether()
            for insn in insns:
                if insn.insn.name == 'SELFDESTRUCT':
                    address = insn.arguments[0]
                elif insn.insn.name == 'CALL':
                    # 找到 CALL 指令所在的基本块
                    current_block = insn.parent_block
                    # 获取当前基本块和前两个基本块
                    cfg = rattle.ControlFlowGraph(function)
                    # 获取前驱基本块
                    predecessors = get_predecessor_blocks(cfg, current_block, 3)

                    if not predecessors:
                        predecessors = find_predecessors_by_offset(function, current_block.offset)

                  
                    # 检查这些基本块是否包含 LOG 指令
                    log_found = False
                    for block in predecessors:
                        for inst in block:
                            if inst.insn.name.startswith("LOG"):  # 检查是否是 LOG 指令
                                print(f"LOG instruction found in block {block.offset:#x}: {inst}")
                                print(f"Start dynamic analysis")
                                log_found = True

                    # 将结果保存到列表
                    results.append({
                        "function": function.desc(),
                        "call_instruction": f"Block {current_block.offset:#x}",
                        "log_found": log_found,
                    })


    # 初始化 Web3
    INFURA_URL = "https://mainnet.infura.io/v3/352b504766c349669daf3f058e02da5f"  # 替换为你的 Infura 项目 ID
    web3 = Web3(Web3.HTTPProvider(INFURA_URL))

    # 检查 Web3 连接
    if not web3.is_connected():
        print("Failed to connect to Ethereum network.")
        exit()
    else:
        print("Connected to Ethereum network.")
    # 目标合约地址
    contract_address = "0x5C6258EE96fD463D1fa6F206CDd79876af2735d9"
    contract_abi = [
    ]

    # 获取合约实例
    contract = web3.eth.contract(address=contract_address, abi=contract_abi)
    print("Contract loaded successfully.")
    # 定义事件过滤器
    from_block = 495  # 合约部署区块
    to_block = 502 # 查询到最新区块
    event_filter = contract.events.DummyLog.create_filter(
        from_block=from_block,
        to_block=to_block
    )
    # 获取事件日志
    logs = event_filter.get_all_entries()
    for log in logs:
        print(f"Event found: Message={log['args']['message']}")

    return results
def find_predecessors_by_offset(function, current_offset, max_blocks=3):
    """
    根据偏移量查找最近的上 `max_blocks` 个基本块。
    :param function: 当前函数对象
    :param current_offset: 当前块的偏移量
    :param max_blocks: 要返回的最大基本块数
    :return: 最近的上 `max_blocks` 个基本块列表
    """
    blocks = sorted(function, key=lambda b: b.offset)
    predecessor_blocks = []

    for block in blocks:
        if block.offset >= current_offset:
            break
        predecessor_blocks.append(block)

    # 返回最后 `max_blocks` 个基本块
    return predecessor_blocks[-max_blocks:]
def get_predecessor_blocks(cfg, block, depth):
    """
    获取当前基本块的前 depth 个基本块。
    :param cfg: 控制流图
    :param block: 当前基本块
    :param depth: 前溯深度
    :return: 前驱基本块列表
    """
    predecessors = []
    queue = [(block, 0)]  # 存储块和当前深度
    visited = set()

    while queue:
        current_block, current_depth = queue.pop(0)

        if current_block in visited:
            continue
        visited.add(current_block)

        # 如果达到所需深度，停止
        if current_depth == depth:
            break

        # 获取直接前驱
        preds = [pred for pred, succ in cfg.edges if succ == current_block]
        predecessors.extend(preds)

        # 将前驱加入队列，增加深度
        for pred in preds:
            queue.append((pred, current_depth + 1))

    return predecessors[:depth]  # 返回最多 depth 个前驱块

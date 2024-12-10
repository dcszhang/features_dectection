import rattle
import requests
from web3 import Web3
import json
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
    # # 配置
    # API_KEY = "BWI4793MTZI4HFAWR767E6VD7QV51XTYWN"
    # BASE_URL = "https://api.etherscan.io/api"

    # def get_contract_logs(contract_address, from_block, to_block, topic=None):
    #     params = {
    #         "module": "logs",
    #         "action": "getLogs",
    #         "fromBlock": from_block,
    #         "toBlock": to_block,
    #         "address": contract_address,
    #         "apikey": API_KEY
    #     }
    #     if topic:
    #         params["topic0"] = topic  # 过滤指定事件
    #     response = requests.get(BASE_URL, params=params)
    #     return response.json()
    

    # contract_address = "0x610178da211fef7d417bc0e6fed39f05609ad788"  # 替换为你的合约地址
    # from_block = "0"  # 起始区块号
    # to_block = "latest"  # 查询到最新区块
    # result = get_contract_logs(contract_address, from_block, to_block)
    # if result["status"] == "1":
    #     print("Logs found:")
    #     for log in result["result"]:
    #         print(json.dumps(log, indent=4))
    # else:
    #     print(f"Error: {result['message']}")


    # 初始化 Web3 连接到本地节点
    LOCAL_RPC_URL = "http://127.0.0.1:8545"  # 本地运行的以太坊节点
    web3 = Web3(Web3.HTTPProvider(LOCAL_RPC_URL))

    # 检查 Web3 连接
    if not web3.is_connected():
        print("Failed to connect to local Ethereum network.")
        exit()
    else:
        print("Connected to local Ethereum network.")

    # 定义合约地址和 ABI
    contract_address = "0x5fbdb2315678afecb367f032d93f642f64180aa3"  # 替换为本地部署的合约地址
    contract_address = Web3.to_checksum_address(contract_address)

    contract_abi = [
        {
            "anonymous": False,
            "inputs": [],
            "name": "PlaceholderEvent",
            "type": "event",
        }
    ]


    # 获取合约实例
    contract = web3.eth.contract(address=contract_address, abi=contract_abi)

    # 查询事件日志
    from_block = 0  # 起始区块
    to_block = "latest"  # 查询到最新区块

    try:
        logs = web3.eth.get_logs({
            "fromBlock": from_block,
            "toBlock": to_block,
            "address": contract_address,
        })

        print(f"Found {len(logs)} logs.")

        for log in logs:
            # 打印原始日志作为参考
            log_dict = dict(log)
            formatted_log = json.dumps(log_dict, indent=4, default=str)
            print(f"Raw log:\n{formatted_log}")

            # 提取并手动解码字符串
            if log['data']:
                data_bytes = Web3.to_bytes(hexstr=log['data'])

                # 解码字符串偏移量（通常是 0x20）
                offset = int.from_bytes(data_bytes[0:32], byteorder='big')
                print(f"String offset: {offset}")

                # 解码字符串长度
                string_length = int.from_bytes(data_bytes[32:64], byteorder='big')
                print(f"String length: {string_length}")

                # 解码实际字符串内容
                string_content = data_bytes[64:64 + string_length].decode('utf-8')
                print(f"Extracted string: {string_content}")
    except Exception as e:
        print(f"Error while fetching logs: {e}")






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

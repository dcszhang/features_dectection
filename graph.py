import networkx as nx
import matplotlib.pyplot as plt

def parse_trace(trace):
    """
    解析输出的依赖关系，构建节点和边
    """
    edges = []
    lines = trace.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        
        # 检查是否为 PHI、MLOAD 等指令
        if '=' in line:
            source, operation = line.split('=', 1)
            source = source.strip()
            if 'PHI' in operation:
                dependencies = operation[operation.index('(')+1:operation.index(')')].split(',')
                for dep in dependencies:
                    dep = dep.strip()
                    edges.append((dep, source))  # 添加依赖关系
            elif 'MLOAD' in operation:
                dependency = operation[operation.index('(')+1:operation.index(')')].strip()
                edges.append((dependency, source))  # MLOAD 依赖
        elif 'Unknown source for' in line:
            # 处理 unknown source
            source = line.split('for ')[1]
            edges.append(("Unknown", source))
    
    return edges


def build_tree_from_edges(edges):
    """
    根据边创建有向图，并确保所有节点都是有效字符串
    """
    graph = nx.DiGraph()
    
    # 添加边到图，并确保所有节点和边都是字符串，并替换特殊字符
    for edge in edges:
        node_from = str(edge[0]).replace("#", "_") if edge[0] else "Unknown Node"
        node_to = str(edge[1]).replace("#", "_") if edge[1] else "Unknown Node"
        graph.add_edge(node_from, node_to)
    
    # 输出节点用于调试
    print("Nodes in the graph:")
    for node in graph.nodes:
        print(f"Node: {node} (Type: {type(node)})")
    
    return graph



def visualize_tree(graph, file_path="tree_structure_graph.png"):
    """
    可视化并保存树状结构图，并检查图中的节点
    """
    plt.figure(figsize=(12, 12))

    # 输出所有节点以进行检查
    print("Visualizing tree with the following nodes:")
    for node in graph.nodes():
        print(f"Node: {node}, Type: {type(node)}")

    # 使用 graphviz 的 dot 布局，适合从上到下的树状结构
    try:
        pos = nx.nx_agraph.graphviz_layout(graph, prog="dot")
    except Exception as e:
        print(f"Error during layout generation: {e}")
        return

    # 绘制图像
    nx.draw(graph, pos, with_labels=True, node_size=2000, node_color="lightblue", font_size=10, font_weight="bold", arrows=True)

    # 保存图像到指定路径
    plt.savefig(file_path)
    plt.close()
    print(f"Tree structure saved to {file_path}")



# 示例 trace
trace = """
To: %271 = AND(#ffffffffffffffffffffffffffffffffffffffff, %1504)
            Trace:
                %1504 = PHI(%715, %1503)
                %1503 = PHI(%1447, %1500)
                %715 = MLOAD(#40)
                %1500 = PHI(%1483, %1493)
                %1447 = PHI(%1233, %1415, %1424, %1434)
                %1493 = PHI(%1310, %1486)
                %1483 = PHI(%1171, %1294, %1476, %1477)
                %1233 = PHI(%1183, %1192, %1211)
                %1415 = PHI(%1183, %1357, %1364, %1394)
                %1424 = PHI(%9, %1235, %1394, %1418, #47fba)
                %1434 = PHI(%1211, %1427)
                %1310 = PHI(%1265, %1302)
                %1486 = PHI(%1357, %1483)
                %1171 = PHI(#60, %1062, %1163, %1165, #9891e)
                %1476 = PHI(%1116, %1475)
                %1477 = PHI(%1171, %1456)
                %1294 = PHI(%1177, %1280, %1281)
                %1183 = PHI(%1136, %1180)
                %1192 = PHI(%959, %1185, #e21df)
                %1211 = PHI(%1021, %1177, %1207)
                %1357 = PHI(%1124, %1171, %1180, %1332)
                %1364 = PHI(%910, %1185, %1359, #b120a)
                %1394 = PHI(%1175, %1207, %1384, %1387)
                %1418 = PHI(%1394, %1417)
                %1235 = PHI(%1192, %1233)
                %1427 = PHI(%1175, %1424)
                %1265 = PHI(%1211, %1257)
                %1302 = PHI(%1180, %1294)
                %1165 = PHI(%1032, %1163)
                %1062 = PHI(#60, %715, %1032, %1055)
                %1163 = PHI(#60, %634, %715, %1116, %1162)
                %1475 = PHI(%1278, %1447, %1456, %1466)
                %1116 = PHI(%610, %1115)
                %1456 = PHI(%1175, %1249, %1449, %1450)
                %1280 = PHI(%1062, %1160, %1278)
                %1177 = PHI(%1062, %1175)
                %1281 = PHI(%1177, %1249)
                %1180 = PHI(%1084, %1171, %1177)
                %1136 = PHI(%1084, %1098, %1124, #19bdb)
                %1185 = PHI(%1136, %1183)
                %959 = PHI(#177, #181)
                %1207 = PHI(%1157, %1192, %1206)
                %1021 = PHI(%341, %943, %1017, %1018, %1020, #4ec, #50a, #528)
                %1124 = PHI(%1084, %1101, %1116, #636c5)
                %1332 = PHI(%1180, %1318, %1319)
                %910 = PHI(#312, #3ad)
                %1359 = PHI(%1124, %1357)
                %1384 = PHI(#60, %1116, %1157, %1364)
                %1175 = PHI(%1032, %1171)
                %1387 = PHI(#46f, #4c9, #4de, #4e7, #4fc, #505, #51a, #523)
                %1417 = PHI(%1364, %1415)
                %1257 = PHI(%1177, %1249)
                %1032 = PHI(#60, %1021, #474, #4ce)
                %1055 = PHI(%715, %942)
                %634 = MLOAD(%586)
                %1162 = PHI(#60, %715, %1062, %1160, #548)
                %1466 = PHI(%1265, %1459)
                %1278 = PHI(%1233, %1249, %1265)
                %610 = MOD(%1206, #10)
                %1115 = PHI(%577, %1114)
                %1449 = PHI(%1032, %1163, %1447)
                %1450 = PHI(%1175, %1424)
                %1249 = PHI(%943, %1211, %1236)
                %1160 = PHI(%586, %715, %1084, %1157)
                %1084 = PHI(%1062, %1076, #8e2c)
                %1098 = PHI(#0)
                %1206 = PHI(%629, %1114)
                %1157 = PHI(%1116, %1136, %1150)
                %943 = PHI(%715, %942)
                %1017 = PHI(%715, %942)
                %1020 = PHI(%715, %942)
                %1018 = PHI(%715, %942)
                %341 = MLOAD(#40)
                %1101 = PHI(#407, #41e)
                %1318 = PHI(%1084, %1157, %1317)
                %1319 = PHI(%1180, %1294)
                %942 = PHI(%586, %715)
                %586 = MLOAD(#40)
                %1459 = PHI(%1171, %1456)
                %1114 = PHI(#47fba, #a6f65)
                %577 = DIV(%1115, #10)
                %1236 = PHI(%1211, %1235)
                %1076 = PHI(#402, #419)
                %629 = DIV(%1206, #10)
                %1150 = PHI(#0, %574)
                %1317 = PHI(%1278, %1294, %1310)
                %574 = ADD(#1, %1150)
"""

# 解析输出，构建图并可视化
edges = parse_trace(trace)
graph = build_tree_from_edges(edges)
visualize_tree(graph, "output_tree_graph.png")

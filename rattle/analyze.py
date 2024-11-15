#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .recover import *

logger = logging.getLogger(__name__)


class UseDefGraph(object):
    value: StackValue

    def __init__(self, value: StackValue) -> None:
        self.value = value

    def dot(self) -> str:
        rv = ''
        rv += 'digraph G {\n'

        es = self.edges(self.value)

        for reader in self.value.readers():
            reader_s = str(reader).replace('%', '\\%')
            value_s = str(self.value).replace('%', '\\%')
            es.append(f"\"{value_s}\" -> \"{reader_s}\"")

        rv += '\n'.join(list(set(es)))
        rv += '\n}'

        return rv

    def edges(self, value) -> List[str]:
        rv = []
        writer = value.writer
        if writer is None:
            return []

        value_s = str(value).replace('%', '\\%')
        writer_s = str(writer).replace('%', '\\%')
        rv.append(f"\"{writer_s}\" -> \"{value_s}\"")

        for arg in writer:
            arg_s = str(arg).replace('%', '\\%')
            writer_s = str(writer).replace('%', '\\%')
            rv.append(f"\"{arg_s}\" -> \"{writer_s}\"")
            rv.extend(self.edges(arg))

        for reader in writer.return_value.readers():
            reader_s = str(reader).replace('%', '\\%')
            value_s = str(value).replace('%', '\\%')
            rv.append(f"\"{value_s}\" -> \"{reader_s}\"")

        return rv


class DefUseGraph(object):
    value: StackValue

    def __init__(self, value: StackValue) -> None:
        self.value = value

    def dot(self, filt=None) -> str:
        if filt is None:
            filt = lambda x: True

        rv = ''
        rv += 'digraph G {\n'

        es = self.edges(self.value, filt)

        for reader in self.value.readers():
            reader_s = str(reader).replace('%', '\\%')
            value_s = str(self.value).replace('%', '\\%')
            es.append(f"\"{value_s}\" -> \"{reader_s}\"")

        rv += '\n'.join(list(set(es)))
        rv += '\n}'

        return rv

    def edges(self, value, filt) -> List[str]:
        rv = []
        writer = value.writer
        if writer is None:
            return []

        value_s = str(value).replace('%', '\\%')
        writer_s = str(writer).replace('%', '\\%')
        rv.append(f"\"{writer_s}\" -> \"{value_s}\"")

        for reader in writer.return_value.readers():
            reader_s = str(reader).replace('%', '\\%')
            value_s = str(value).replace('%', '\\%')
            rv.append(f"\"{value_s}\" -> \"{reader_s}\"")

            if filt(reader):
                rv.extend(self.edges(reader.return_value, filt))

        return rv

class ControlFlowGraph:
    def __init__(self, function: SSAFunction) -> None:
        self.function = function
        self.edges = self.get_edges()
        self.block_dict = {block.offset: block for block in function}  # Create a dictionary for efficient block lookup

    def get_edges(self):
        edges = []
        for block in self.function:
            # Use block references directly instead of creating string IDs
            if block.fallthrough_edge:
                edges.append((block, block.fallthrough_edge))

            for edge in block.jump_edges:
                edges.append((block, edge))

        return edges

    def get_dominators(self):
        dominators = {block: set(self.function) for block in self.function}
        start_block = next(iter(self.function))
        dominators[start_block] = {start_block}
        changed = True

        while changed:
            changed = False
            for block in self.function:
                if block == start_block:
                    continue
                preds = [pred for pred, succ in self.edges if succ == block]
                new_doms = set(self.function) if not preds else set.intersection(*[dominators[pred] for pred in preds])
                new_doms.add(block)
                if new_doms != dominators[block]:
                    dominators[block] = new_doms
                    changed = True

        return dominators

    def get_immediate_dominators(self):
        dominators = self.get_dominators()
        idoms = {}
        for block in self.function:
            doms = dominators[block] - {block}
            idom = None
            for dom in doms:
                if all(other_dom == dom or other_dom not in dominators[block] for other_dom in doms):
                    idom = dom
                    break
            idoms[block] = idom if idom is not None else next(iter(dominators[block]))  # Ensure every block has an immediate dominator
        return idoms

    def dot(self) -> str:
        rv = 'digraph G {'
        rv += 'graph [fontname = "consolas"];'
        rv += 'node [fontname = "consolas"];'
        rv += 'edge [fontname = "consolas"];'

        name = self.function.desc()
        hash = f'Hash: {self.function.hash:#x}'
        offset = f'Start: {self.function.offset:#x}'
        arguments = f'Arguments: {self.function.arguments()}'
        storage = f'Storage: {self.function.storage}'

        function_desc = [name, hash, offset, arguments, storage]
        rv += f'ff [label="{{' + '\l'.join(function_desc) + '\l}}", shape="record" ];'

        edges = []

        for block in self.function:
            block_body = '\l'.join([f'{insn.offset:#x}: {insn}' for insn in block])
            block_body = block_body.replace('<', '\<').replace('>', '\>')
            rv += f'block_{block.offset:#x} [label="{block_body}\l", shape="record"] ;'

        for block, target_block in self.edges:
            rv += f'block_{block.offset:#x} -> block_{target_block.offset:#x};'

        rv += '}'
        return rv


class ProgramDependenceGraph:
    def __init__(self, function: SSAFunction):
        self.function = function
        self.cfg = ControlFlowGraph(function)
        self.control_edges = self._extract_control_dependence()
        self.data_edges = self._extract_data_dependence()

    def _extract_control_dependence(self):
        idoms = self.cfg.get_immediate_dominators()
        control_edges = []
        for block in self.function:
            idom = idoms.get(block)
            if idom is not None:
                control_edges.append((idom, block))
        return control_edges

    def _extract_data_dependence(self):
        data_edges = []
        definitions = {}

        for block in self.function:
            for insn in block:
                if insn.return_value is not None:
                    definitions[insn.return_value] = block

                used_vars = insn.arguments
                for var in used_vars:
                    if var in definitions:
                        data_edges.append((definitions[var], block))

        return data_edges

    def dot(self):
        rv = 'digraph PDG {'
        rv += 'graph [fontname = "consolas"];'
        rv += 'node [fontname = "consolas"];'
        rv += 'edge [fontname = "consolas"];'

        name = self.function.desc()
        hash = f'Hash: {self.function.hash:#x}'
        offset = f'Start: {self.function.offset:#x}'
        arguments = f'Arguments: {self.function.arguments()}'
        storage = f'Storage: {self.function.storage}'
        function_desc = [name, hash, offset, arguments, storage]
        rv += f'ff [label="{{' + '\l'.join(function_desc) + '\l}}", shape="record" ];'

        for block in self.function:
            block_body = '\l'.join([f'{insn.offset:#x}: {insn}' for insn in block])
            block_body = block_body.replace('<', '\<').replace('>', '\>')
            rv += f'block_{block.offset:#x} [label="{block_body}\l", shape="record"] ;'

        for edge in self.control_edges:
            rv += f'block_{edge[0].offset:#x} -> block_{edge[1].offset:#x} [label="control", color="blue"];'

        for edge in self.data_edges:
            rv += f'block_{edge[0].offset:#x} -> block_{edge[1].offset:#x} [label="data", color="green"];'

        rv += '}'
        return rv


class SystemDependenceGraph:
    def __init__(self, functions: List[SSAFunction]):
        self.function_pdgs = {func: ProgramDependenceGraph(func) for func in functions}
        self.call_edges = self._extract_call_dependence()
        self.parameter_edges = self._extract_parameter_dependence()

    def _extract_call_dependence(self):
        call_edges = []
        for caller, caller_pdg in self.function_pdgs.items():
            for block in caller:
                for insn in block:
                    if insn.insn.name in ('CALL', 'CALLCODE', 'DELEGATECALL'):
                        if isinstance(insn.arguments[1], ConcreteStackValue):
                            callee_hash = insn.arguments[1].concrete_value
                            callee = self._find_function_by_hash(callee_hash)
                            if callee:
                                call_edges.append((block, callee.blocks[0]))
        return call_edges

    def _extract_parameter_dependence(self):
        parameter_edges = []
        for caller, caller_pdg in self.function_pdgs.items():
            for block in caller:
                for insn in block:
                    if insn.insn.name in ('CALL', 'CALLCODE', 'DELEGATECALL'):
                        resolved_arg, _ = insn.arguments[1].resolve()
                        if hasattr(resolved_arg, 'concrete_value'):
                            callee_hash = resolved_arg.concrete_value
                            callee = self._find_function_by_hash(callee_hash)
                            if callee:
                                for i, arg in enumerate(insn.arguments[2:]):
                                    if arg in caller_pdg.data_edges:
                                        callee_param = f'param_{i}'
                                        parameter_edges.append((block, callee_param))
                                if insn.return_value:
                                    parameter_edges.append((callee.blocks[-1], block))
        return parameter_edges

    def _find_function_by_hash(self, hash_value):
        return next((func for func in self.function_pdgs if func.hash == hash_value), None)

    def dot(self):
        rv = 'digraph SDG {'
        rv += 'graph [fontname = "consolas"];'
        rv += 'node [fontname = "consolas"];'
        rv += 'edge [fontname = "consolas"];'

        for func, pdg in self.function_pdgs.items():
            name = func.desc().replace('<', '\<').replace('>', '\>')
            hash = f'Hash: {func.hash:#x}'
            offset = f'Start: {func.offset:#x}'
            arguments = f'Arguments: {func.arguments()}'
            storage = f'Storage: {func.storage}'

            node_name = name.replace('(', '').replace(')', '').replace(' ', '_')
            function_desc = [name, hash, offset, arguments, storage]
            rv += f'{node_name} [label="{{' + '\l'.join(function_desc) + '\l}}", shape="record" ];'

            for block in func:
                block_body = '\l'.join([f'{insn.offset:#x}: {insn}' for insn in block])
                block_body = block_body.replace('<', '\<').replace('>', '\>').replace('(', '\(').replace(')', '\)')
                rv += f'block_{block.offset:#x} [label="{block_body}\l", shape="record"] ;'

            for edge in pdg.control_edges:
                rv += f'block_{edge[0].offset:#x} -> block_{edge[1].offset:#x} [label="control", color="blue"];'

            for edge in pdg.data_edges:
                rv += f'block_{edge[0].offset:#x} -> block_{edge[1].offset:#x} [label="data", color="green"];'

            if hasattr(pdg, 'parameter_edges'):
                for edge in pdg.parameter_edges:
                    rv += f'{edge[0].offset:#x} -> {edge[1]} [label="param", color="orange"];'

        rv += '}'
        return rv




class CallBacktracking:
    def __init__(self, sdg: SystemDependenceGraph):
        self.sdg = sdg
        self.backtrack_results = self._perform_backtracking()

    def _perform_backtracking(self):
        backtrack_results = {}
        for func, pdg in self.sdg.function_pdgs.items():
            backtrack_paths = self._backtrack_calls_in_function(pdg)
            backtrack_results[func] = backtrack_paths
        return backtrack_results

    def _backtrack_calls_in_function(self, pdg: ProgramDependenceGraph):
        backtrack_paths = []
        for block in pdg.function:
            for insn in block:
                if insn.insn.name in ('CALL', 'CALLCODE', 'DELEGATECALL'):
                    visited = set()
                    path = []
                    self._backtrack(block, pdg, visited, path)
                    backtrack_paths.append(path)
        return backtrack_paths

    def _backtrack(self, block, pdg, visited, path):
        if block in visited:
            return
        visited.add(block)
        path.append(block)
        preds = [pred for pred, succ in pdg.control_edges + pdg.data_edges if succ == block]
        for pred in preds:
            self._backtrack(pred, pdg, visited, path)

    def get_backtrack_results(self):
        return self.backtrack_results

class CallBacktracking:
    def __init__(self, sdg: SystemDependenceGraph):
        self.sdg = sdg
        self.backtrack_results = self._perform_backtracking()

    def _perform_backtracking(self):
        backtrack_results = {}
        for func, pdg in self.sdg.function_pdgs.items():
            backtrack_paths = self._backtrack_calls_in_function(pdg)
            backtrack_results[func] = backtrack_paths
        return backtrack_results

    def _backtrack_calls_in_function(self, pdg: ProgramDependenceGraph):
        backtrack_paths = []
        for block in pdg.function:
            for insn in block:
                if insn.insn.name in ('CALL', 'CALLCODE', 'DELEGATECALL'):
                    visited = set()
                    path = []
                    self._backtrack(block, pdg, visited, path, exclude_calls=True)
                    backtrack_paths.append(path)
        return backtrack_paths

    def _backtrack(self, block, pdg, visited, path, exclude_calls=False):
        if block in visited:
            return
        visited.add(block)
        path.append(block)
        if exclude_calls:
            # 如果当前块中包含 CALL 指令，则跳过它
            for insn in block:
                if insn.insn.name in ('CALL', 'CALLCODE', 'DELEGATECALL') and block != path[0]:
                    return
        preds = [pred for pred, succ in pdg.control_edges + pdg.data_edges if succ == block]
        for pred in preds:
            self._backtrack(pred, pdg, visited, path, exclude_calls)

    def get_backtrack_results(self):
        return self.backtrack_results

    def dot(self):
        rv = ''
        for func, paths in self.backtrack_results.items():
            pdg = self.sdg.function_pdgs[func]  # 获取与函数关联的 PDG 对象
            for idx, path in enumerate(paths):
                func_name = func.desc().replace('<', '\<').replace('>', '\>')
                rv += f'digraph "{func_name}_CallBacktrack_{idx}" {{'
                rv += 'graph [fontname = "consolas"];'
                rv += 'node [fontname = "consolas"];'
                rv += 'edge [fontname = "consolas"];'

                if path:
                    start_block = path[0]
                    rv += f'block_{start_block.offset:#x} [shape="box", color="yellow"];'

                for i in range(len(path) - 1):
                    block = path[i]
                    next_block = path[i + 1]
                    rv += f'block_{block.offset:#x} -> block_{next_block.offset:#x} [label="backtrack", color="red"];'

                for block in path:
                    block_body = '\l'.join([f'{insn.offset:#x}: {insn}' for insn in block])
                    block_body = block_body.replace('<', '\<').replace('>', '\>')
                    rv += f'block_{block.offset:#x} [label="{block_body}\l", shape="record"] ;'

                # 继承控制依赖边和数据依赖边
                control_edges = [edge for edge in pdg.control_edges if edge[0] in path and edge[1] in path]
                data_edges = [edge for edge in pdg.data_edges if edge[0] in path and edge[1] in path]

                for edge in control_edges:
                    rv += f'block_{edge[0].offset:#x} -> block_{edge[1].offset:#x} [label="control", color="blue"];'
                for edge in data_edges:
                    rv += f'block_{edge[0].offset:#x} -> block_{edge[1].offset:#x} [label="data", color="green"];'

                rv += '}'
        return rv

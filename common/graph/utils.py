from typing import *

NodeName = Any


def inv(graph: Dict[NodeName, List[NodeName]]) -> Dict[NodeName, List[NodeName]]:
    inv_graph = {v: [] for v in graph}
    for u in graph:
        for v in graph[u]:
            if v not in inv_graph:
                inv_graph[v] = []
            inv_graph[v].append(u)
    return inv_graph


def tarjan(graph: Dict[NodeName, List[NodeName]]) -> List[List[NodeName]]:
    index = {}
    lowlink = {}
    stack = []
    result = []
    index_counter = [0]

    def strong_connect(node):
        index[node] = index_counter[0]
        lowlink[node] = index_counter[0]
        index_counter[0] += 1
        stack.append(node)

        for neighbor in graph[node]:
            if neighbor not in index:
                strong_connect(neighbor)
                lowlink[node] = min(lowlink[node], lowlink[neighbor])
            elif neighbor in stack:
                lowlink[node] = min(lowlink[node], index[neighbor])

        if lowlink[node] == index[node]:
            connected_component = []
            while True:
                neighbor = stack.pop()
                connected_component.append(neighbor)
                if neighbor == node:
                    break
            result.append(connected_component)

    for node in graph:
        if node not in index:
            strong_connect(node)

    return result


def contract_scc(sccs: List[List[NodeName]], graph: Dict[NodeName, List[NodeName]]) -> Dict[NodeName, List[NodeName]]:
    contracted_graph = {}

    node_to_scc = {}
    for scc_index, scc in enumerate(sccs):
        for node in scc:
            node_to_scc[node] = scc_index

    for node, neighbors in graph.items():
        scc_index = node_to_scc[node]
        if scc_index not in contracted_graph:
            contracted_graph[scc_index] = set()
        for neighbor in neighbors:
            neighbor_scc_index = node_to_scc[neighbor]
            if scc_index != neighbor_scc_index:
                contracted_graph[scc_index].add(neighbor_scc_index)

    return contracted_graph


def topological_sort(graph: Dict[NodeName, List[NodeName]]):
    def dfs_topo(graph, node, visited, stack):
        visited.add(node)
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                dfs_topo(graph, neighbor, visited, stack)
        stack.append(node)

    visited = set()
    stack = []
    for node in graph:
        if node not in visited:
            dfs_topo(graph, node, visited, stack)

    return stack[::-1]


def longest_path(graph: Dict[NodeName, List[NodeName]]):
    sccs = tarjan(graph)
    assert all(len(scc) == 1 for scc in sccs)  # dag is required
    topo = topological_sort(graph)

    inv_graph = {node: [] for node in graph}
    for node in graph:
        for child in graph[node]:
            if not child in inv_graph:
                inv_graph[child] = []
            inv_graph[child].append(node)

    max_anc_length = {}
    max_length = 0
    for node in topo:
        max_anc_length[node] = max([
            max_anc_length[parent] for parent in inv_graph[node]
        ]) + 1 if len(inv_graph[node]) > 0 else 1
        max_length = max(max_length, max_anc_length[node])
    return max_length


def bron_kerbosch(graph, R=set(), P=None, X=None):
    if P is None:
        P = set(graph.keys())
    if X is None:
        X = set()

    if not P and not X:
        yield R
    while P:
        v = P.pop()
        yield from bron_kerbosch(graph, R | {v}, P.intersection(graph[v]), X.intersection(graph[v]))
        X.add(v)

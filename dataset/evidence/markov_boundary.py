import networkx as nx
from itertools import combinations
from typing import *


def markov_boundary(graph: Dict[str, List[str]],
                    x: str,
                    t: Set[str],
                    y: Set[str],
                    ) -> Set[str]:
    # Intervention
    graph = {u: [v for v in graph[u] if v not in t] for u in graph}

    # Create a networkx.DiGraph
    g = nx.DiGraph()
    for u in graph:
        g.add_node(u)
    for u in graph:
        for v in graph[u]:
            g.add_edge(u, v)

    # 2^y
    _2_y = [set(Z) for r in range(0, len(y) + 1)
            for Z in combinations(y, r)]

    # Enumerate over all subsets of variables
    mb = y
    for z in _2_y:
        # Update minimal seperation set
        if len(z) < len(mb) and nx.d_separated(g, {x}, y - z, z):
            mb = z

    # The minimal seperation set is markov boundary
    return mb


def aux_graph(graph: Dict[str, List[str]],
              t: Set[str],
              c: Set[str],
              exo: Set[str] = None,
              itv_mode: str = 'normal',
              ):
    ch = {u: set() for u in graph}
    pa = {u: set() for u in graph}

    # Intervention
    if itv_mode == 'normal':
        graph = {u: [v for v in graph[u] if v not in t] for u in graph}
    elif itv_mode == 'endo_only':
        assert exo is not None
        graph = {
            u: graph[u] if u in exo
            else [v for v in graph[u] if v not in t]
            for u in graph
        }
    elif itv_mode == 'fake':
        graph = graph

    # Record parents and children
    for u in graph:
        for v in graph[u]:
            ch[u].add(v)
            pa[v].add(u)

    # Build conditional graph
    aux_graph = {u: set() for u in graph}

    def dfs_an(v):
        for u in pa[v]:
            if u in aux_graph[v]:
                continue
            aux_graph[v].add(u)
            dfs_an(u)
            dfs_de(u)

    def dfs_de(u):
        for v in ch[u]:
            if v in aux_graph[u]:
                continue
            aux_graph[u].add(v)
            dfs_de(v)

    for x in c:
        dfs_an(x)
        dfs_de(x)
    for x in c:  # collider
        bi = [u for u in pa[x] if u in aux_graph[x] and x in aux_graph[u]]
        for w in bi:
            aux_graph[w] |= pa[x]
    for u in aux_graph:  # make root
        for v in c:
            if v in aux_graph[u]:
                aux_graph[u].remove(v)

    return aux_graph


def fast_markov_boundary(aux_graph: Dict[str, List[str]],
                         x: str,
                         ) -> Set[str]:
    # Find the root ancestors
    inv_gragh = {u: [] for u in aux_graph}
    for u in aux_graph:
        for v in aux_graph[u]:
            inv_gragh[v].append(u)
    vis = {u: 0 for u in aux_graph}
    res = []

    # Search inverse graph
    def dfs(u):
        if vis[u] == 1:
            return
        vis[u] = 1
        if len(inv_gragh[u]) == 0:
            res.append(u)
        for v in inv_gragh[u]:
            dfs(v)

    dfs(x)
    return set(res) - {x}

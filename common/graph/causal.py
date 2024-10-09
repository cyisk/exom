from typing import *

from common.graph.utils import *


class DirectedMixedGraph:
    def __init__(self):
        self.directed_graph = {}
        self.bidirected_graph = {}

    def add_node(self, u: NodeName):
        if u not in self.directed_graph:
            self.directed_graph[u] = []
        if u not in self.bidirected_graph:
            self.bidirected_graph[u] = []

    def add_directed_edge(self, u: NodeName, v: NodeName):
        self.add_node(u)
        self.add_node(v)
        if v not in self.directed_graph[u]:
            self.directed_graph[u].append(v)

    def add_bidirected_edge(self, u: NodeName, v: NodeName):
        self.add_node(u)
        self.add_node(v)
        if u == v:
            return
        if v not in self.bidirected_graph[u]:
            self.bidirected_graph[u].append(v)
        if u not in self.bidirected_graph[v]:
            self.bidirected_graph[v].append(u)

    def augment(self, naming_rule: Callable = None) -> "AugmentedGraph":
        if naming_rule is None:
            def naming_rule(c2): return f'u_{"_".join(c2)}'
        aug_graph = AugmentedGraph()
        for u in self.nodes:
            for v in self.directed_graph[u]:
                aug_graph.add_endogenous_edge(u, v)
        for c2 in self.maximal_confounded_components:
            u = naming_rule(c2)
            for v in c2:
                aug_graph.add_exogenous_edge(u, v)
        return aug_graph

    def intervene(self, intervened: List[str]) -> "DirectedMixedGraph":
        causal_graph = DirectedMixedGraph()
        for u in self.nodes:
            causal_graph.add_node(u)
            for v in self.directed_subgraph[u]:
                if v in intervened:
                    continue
                self.add_directed_edge(u, v)
            for v in self.bidirected_subgraph[u]:
                if v in intervened or u in intervened:
                    continue
                self.add_bidirected_edge(u, v)

    @property
    def nodes(self) -> List[NodeName]:
        return list(self.directed_graph.keys())

    @property
    def directed_subgraph(self) -> Dict[NodeName, List[NodeName]]:
        return self.directed_graph

    @property
    def bidirected_subgraph(self) -> Dict[NodeName, List[NodeName]]:
        return self.bidirected_graph

    @property
    def maximal_confounded_components(self) -> List[Set[NodeName]]:
        cliques = list(bron_kerbosch(self.bidirected_graph))
        return cliques + [{v} for v in self.nodes if {v} not in cliques]

    @property
    def is_dmg(self) -> bool:
        return True

    @property
    def is_admg(self) -> bool:
        return self.augment().is_dag

    @property
    def is_dag(self) -> bool:
        for v in self.bidirected_graph:
            if len(self.bidirected_graph[v]) > 0:
                return False
        sccs = tarjan(self.directed_subgraph)
        return all(len(scc) == 1 for scc in sccs)

    @staticmethod
    def from_dict(directed_subgraph: Dict[NodeName, List[NodeName]],
                  bidirected_subgraph: Dict[NodeName, List[NodeName]]) -> "DirectedMixedGraph":
        causal_graph = DirectedMixedGraph()
        for u in directed_subgraph:
            for v in directed_subgraph[u]:
                causal_graph.add_directed_edge(u, v)
        for u in bidirected_subgraph:
            for v in bidirected_subgraph[u]:
                if u == v:
                    continue
                causal_graph.add_bidirected_edge(u, v)
        return causal_graph


class AugmentedGraph:
    def __init__(self):
        self.node_type = {}
        self.endo_graph = {}
        self.exo_graph = {}

    def add_node(self, u: NodeName, is_exogenous: bool = False):
        self.node_type[u] = is_exogenous
        if not is_exogenous:
            if u not in self.endo_graph:
                self.endo_graph[u] = []
        else:
            if u not in self.exo_graph:
                self.exo_graph[u] = []

    def add_endogenous_edge(self, u: NodeName, v: NodeName):
        self.add_node(u)
        self.add_node(v)
        if v not in self.endo_graph[u]:
            self.endo_graph[u].append(v)

    def add_exogenous_edge(self, u: NodeName, v: NodeName):
        self.add_node(u, is_exogenous=True)
        self.add_node(v)
        if v not in self.exo_graph[u]:
            self.exo_graph[u].append(v)

    def unaugment(self) -> DirectedMixedGraph:
        causal_graph = DirectedMixedGraph()
        for u in self.endogenous_nodes:
            for v in self.endo_graph[u]:
                causal_graph.add_directed_edge(u, v)
        for u in self.exogenous_nodes:
            for v1 in self.exo_graph[u]:
                for v2 in self.exo_graph[u]:
                    if v1 not in causal_graph.bidirected_subgraph:
                        continue
                    if v2 in causal_graph.bidirected_subgraph[v1]:
                        continue
                causal_graph.add_bidirected_edge(v1, v2)
        return causal_graph

    def intervene(self, intervened: List[str]) -> "AugmentedGraph":
        aug_graph = AugmentedGraph()
        for u in self.endogenous_nodes:
            aug_graph.add_node(u)
            for v in self.endo_graph[u]:
                if v in intervened:
                    continue
                aug_graph.add_endogenous_edge(u, v)
        for u in self.exogenous_nodes:
            aug_graph.add_node(u, is_exogenous=True)
            for v in self.exo_graph[u]:
                if v in intervened:
                    continue
                aug_graph.add_exogenous_edge(u, v)
        return aug_graph

    @property
    def endogenous_nodes(self) -> List[NodeName]:
        return [u for u in self.node_type if self.node_type[u] == False]

    @property
    def exogenous_nodes(self) -> List[NodeName]:
        return [u for u in self.node_type if self.node_type[u] == True]

    @property
    def endogenous_subgraph(self) -> Dict[NodeName, List[NodeName]]:
        return self.endo_graph

    @property
    def exogenous_subgraph(self) -> Dict[NodeName, List[NodeName]]:
        return self.exo_graph

    @property
    def graph(self) -> Dict[NodeName, List[NodeName]]:
        return {**self.exo_graph, **self.endo_graph}

    @property
    def is_dmg(self) -> bool:
        return False

    @property
    def is_admg(self) -> bool:
        return True

    @property
    def is_dag(self) -> bool:
        sccs = tarjan(self.endogenous_subgraph)
        return all(len(scc) == 1 for scc in sccs)

import torch as th

from common.graph.causal import DirectedMixedGraph
from common.scm import TensorSCM
from dataset.synthetic.regional_canonical.rcm import rcm

g_directed = {
    'x': ['y', 'w'],
    'y': [],
    'z': ['x', 'y', 'w'],
    'w': ['y'],
}
g_undirected = {
    'x': ['w'],
    'w': ['x']
}


cg = DirectedMixedGraph.from_dict(g_directed, g_undirected)


def fairness_xw_init(r_size: int = 50,
                     x_dim: int = 0,
                     y_dim: int = 0,
                     z_dim: int = 16,
                     w_dim: int = 0,
                     rc_seed: int = 0,
                     *args, **kwarg) -> TensorSCM:
    V_dims = {
        'x': th.Size((x_dim, )) if x_dim > 0 else th.Size(),
        'y': th.Size((y_dim, )) if y_dim > 0 else th.Size(),
        'z': th.Size((z_dim, )) if z_dim > 0 else th.Size(),
        'w': th.Size((w_dim, )) if w_dim > 0 else th.Size(),
    }
    return rcm(
        causal_graph=cg,
        V_dims=V_dims,
        r_size=r_size,
        rc_seed=rc_seed,
        name='fairness_xw',
    )

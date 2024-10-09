import functools
import os
import torch as th
from torch.distributions import Uniform
from typing import List, Dict

from common.graph.causal import *
from common.scm import Equation, TensorSCM, batchshape
from common.scm.eq import EquationWrapper


def ran(dim: th.Size):
    # Get n_elements for multidimensional shape
    return int(th.prod(th.tensor(dim)).item())


def flatten(dims: Dict[str, th.Size], **X):
    if len(X) == 0:
        return None, None
    # Flatten tensors into [batch_size, ran] according to dimension table
    Y = {}
    for x, x_val in X.items():
        Y[x] = x_val.reshape(-1, ran(dims[x]))
    return Y, batchshape(X, dims)


def unflatten(dims: Dict[str, th.Size], batch_size, **Y):
    # Unflatten tensors from [batch_size, ran] according to dimension table and batch_size
    X = {}
    for y, y_val in Y.items():
        X[y] = y_val.reshape([*batch_size, *dims[y]])
    return X


def rand_hyperboxes(r_size: int, u_size: int, rng: th.Generator):
    # Generate random hyperboxes sized [r_size, u_size, 2] from Uniform distribution
    rand_boxes = th.empty(
        [r_size, u_size, 2]
    ).uniform_(0, 1, generator=rng)
    # Swap last dimension if unordered
    return th.sort(rand_boxes, dim=-1).values


def rand_mappings(r_size: int, pV_ran: int, v_ran: int, rng: th.Generator):
    # Generate random mappings from Bernoulli distribution
    rand_mappings = th.empty(
        [r_size + 1] + [2]*(pV_ran) + [v_ran]
    ).bernoulli_(0.5, generator=rng)
    return rand_mappings


def find_hyperbox(hyperboxes: th.Tensor, *X):
    x = th.cat(X, dim=-1)
    assert x.dim() == 2
    r = hyperboxes.shape[0]  # [r_size, u_size, 2]
    b = x.shape[0]  # [batch_size, u_size]

    # hyperboxes[r, i, 0] <= x[b, i] <= hyperboxes[r, i, 1]
    # hyperboxes_[b, r, i, 0] <= x_[b, r, i] <= hyperboxes_[b, r, i, 1]
    x_ = x[:, None, :].expand(-1, r, -1)
    hyperboxes_ = hyperboxes[None, :, :, :].expand(b, -1, -1, -1)

    inbox = th.all((hyperboxes_[..., 0] <= x_) &
                   (x_ <= hyperboxes_[..., 1]), dim=-1)

    r = th.argmax(
        th.cat((
            inbox.to(th.int),
            th.ones((b, 1), device=x.device),  # Default case
        ), dim=1), dim=1,
    )
    return r


def map_by_indices(mappings, *indices):
    concatenated_indices = th.cat(indices, dim=1).to(th.int)
    indices_unbind = th.unbind(concatenated_indices, dim=1)
    return mappings[tuple(indices_unbind)]


class FunctionRCM:
    def __init__(self,
                 r_size: int,
                 pV: List[str],
                 pV_dims: Dict[str, th.Size],
                 pU: List[str],
                 v: str,
                 v_dim: th.Size,
                 rng: th.Generator = None,
                 ) -> None:
        # Random generator
        if rng is None:
            rng = th.Generator()
            rng.manual_seed(0)

        # Sizes
        self.pV_dims = pV_dims
        self.pU_dims = {u: th.Size() for u in pU}
        self.v_dim = v_dim
        pV_ran = sum(ran(dim) for dim in pV_dims.values())
        v_ran = ran(v_dim)
        u_size = len(pU)

        # Variables
        self.v = v
        self.pV = pV
        self.pU = pU

        # Generate random boxes
        self.hyperboxes = rand_hyperboxes(r_size, u_size, rng)

        # Generate random mapping from D_pV to D_v
        self.mappings = rand_mappings(r_size, pV_ran, v_ran, rng)

        # Cross device (constant) tensors
        self.hyperboxes_deviced = {}
        self.mappings_deviced = {}

    def load(self, tensors: Dict[str, th.Tensor]):
        self.hyperboxes = tensors['hyperboxes']
        self.mappings = tensors['mappings']

    def save(self) -> Dict[str, th.Tensor]:
        return {
            'hyperboxes': self.hyperboxes,
            'mappings': self.mappings,
        }

    def __call__(self) -> EquationWrapper:
        VU = ', '.join(self.pV + self.pU)
        Umap = '{' + ', '.join([f'\'{u}\': {u}' for u in self.pU]) + '}'
        Vmap = '{' + ', '.join([f'\'{v}\': {v}' for v in self.pV]) + '}'

        func_str = '\n'.join([
            f"def dummy_func(self, {VU}):",
            # Cross device tensors
            f"    device = {self.pU[0]}.device",
            f"    if device not in self.hyperboxes_deviced:",
            f"        self.hyperboxes_deviced[device] = self.hyperboxes.to({self.pU[0]}.device)",
            f"    if device not in self.mappings_deviced:",
            f"        self.mappings_deviced[device] = self.mappings.to({self.pU[0]}.device)",
            f"    hyperboxes = self.hyperboxes_deviced[device]",
            f"    mappings = self.mappings_deviced[device]",
            # RCM forward
            f"    U_, batch_size = flatten(self.pU_dims, **{Umap})",
            f"    V_, _ = flatten(self.pV_dims, **{Vmap})",
            f"    r = find_hyperbox(hyperboxes, *(U_.values())).unsqueeze(-1)",
            f"    i = [r] if V_ is None else [r, *(V_.values())]",
            f"    v = map_by_indices(mappings, *i)",
            f"    return unflatten({{self.v: self.v_dim}}, batch_size, **{{self.v: v}})[self.v]",
        ])
        locals = {}
        exec(func_str, globals(), locals)
        return Equation(v=self.v, pV=self.pV)(
            functools.partial(locals['dummy_func'], self)
        )


def rcm(
    causal_graph: DirectedMixedGraph,
    V_dims=Dict[str, th.Size],
    r_size: int = 20,
    rc_seed: int = 0,
    name: str = 'fairness',
) -> TensorSCM:
    real_path = os.path.dirname(os.path.realpath(__file__))
    save_path = os.path.join(real_path, 'rcm_saves', name + '.pt')
    if not os.path.exists(save_path):
        return rand_rcm(causal_graph, V_dims, r_size, rc_seed, True, name)
    else:
        return load_rcm(causal_graph, V_dims, r_size, name)


def rand_rcm(
    causal_graph: DirectedMixedGraph,
    V_dims: Dict[str, th.Size],
    r_size: int = 20,
    rc_seed: int = 0,
    save: bool = True,
    name: str = 'fairness',
) -> TensorSCM:
    # random generator
    rng = th.Generator()
    rng.manual_seed(rc_seed)

    aug_graph = causal_graph.augment()
    pV = inv(aug_graph.endogenous_subgraph)
    pU = inv(aug_graph.exogenous_subgraph)
    U = aug_graph.exogenous_nodes

    # Generate random functions
    randfs = []
    eqs = []
    saves = {}
    for v in V_dims:
        pV_dims = {pv: V_dims[pv] for pv in pV[v]}
        randf = FunctionRCM(  # Generate random function
            r_size, pV[v], pV_dims, pU[v], v, V_dims[v],
            rng=rng
        )
        randfs.append(randf)
        eqs.append(randf())
        saves[v] = randf.save()

    # Save if needed
    if save:
        real_path = os.path.dirname(os.path.realpath(__file__))
        save_path = os.path.join(real_path, 'rcm_saves', name + '.pt')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        th.save(saves, save_path)

    # Uniform exogeneous distributions
    exogeneous_distrs = {u: Uniform(low=0, high=1) for u in U}

    # Initialize scm
    scm = TensorSCM(
        equations=eqs,
        exogenous_distrs=exogeneous_distrs,
        name=name,
    )
    scm.randfs = randfs

    return scm


def load_rcm(
    causal_graph: DirectedMixedGraph,
    V_dims: Dict[str, th.Size],
    r_size: int = 20,
    name: str = 'fairness',
) -> TensorSCM:
    aug_graph = causal_graph.augment()
    pV = inv(aug_graph.endogenous_subgraph)
    pU = inv(aug_graph.exogenous_subgraph)
    U = aug_graph.exogenous_nodes

    # Generate random functions
    randfs = []
    eqs = []
    real_path = os.path.dirname(os.path.realpath(__file__))
    save_path = os.path.join(real_path, 'rcm_saves', name + '.pt')
    saves = th.load(save_path)
    for v in V_dims:
        pV_dims = {pv: V_dims[pv] for pv in pV[v]}
        randf = FunctionRCM(  # Generate random function
            r_size, pV[v], pV_dims, pU[v], v, V_dims[v],
        )
        randf.load(saves[v])
        randfs.append(randf)
        eqs.append(randf())

    # Uniform exogeneous distributions
    exogeneous_distrs = {u: Uniform(low=0, high=1) for u in U}

    # Initialize scm
    scm = TensorSCM(
        equations=eqs,
        exogenous_distrs=exogeneous_distrs,
        name=name,
    )
    scm.randfs = randfs

    return scm

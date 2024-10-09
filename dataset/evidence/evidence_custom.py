import torch as th
from typing import *

from dataset.evidence.evidence_joint import *
from dataset.evidence.markov_boundary import *


class EvidenceContextConcat(Evidence):
    # A EvidenceContextConcat provides concatenated evidence and masks of default evidence
    def __init__(self,
                 context_mode: List[str] = ['e', 't', 'w_e', 'w_t'],
                 *args,
                 **kwargs,
                 ) -> None:
        super().__init__(*args, **kwargs)

        assert set(context_mode).issubset(
            {'e', 't', 'e+t', 'w_e', 'w_t', 'w_e+w_t'}
        )
        self._context_mode = context_mode

    def get_context_feature(self, context_name: str) -> th.Tensor:
        if context_name == 'e':
            return self.e
        elif context_name == 't':
            return self.t
        elif context_name == 'e+t':
            e_t = th.zeros_like(self.e)
            e_t[self.w_e] = self.e[self.w_e]
            e_t[self.w_t] = self.t[self.w_t]
            return e_t
        elif context_name == 'w_e':
            return self.w_e.float()
        elif context_name == 'w_t':
            return self.w_t.float()
        elif context_name == 'w_e+w_t':
            return (self.w_e | self.w_t).float()

    @property
    def context_mode(self) -> List[str]:
        return self._context_mode

    @property
    def context(self) -> th.Tensor:
        # Concatenate context tensors
        return th.cat([
            self.get_context_feature(x) for x in self._context_mode
        ], dim=-1)

    @property
    def adjacency(self) -> th.Tensor | None:
        return None

    @staticmethod
    def context_features(scm: TensorSCM,
                         context_mode: List[str] = ['e', 't', 'w_e', 'w_t'],
                         ) -> int:
        return len(context_mode) * sum(scm.endogenous_features.values())

    @staticmethod
    def standardize(context: th.Tensor,
                    w_e: th.Tensor,
                    w_t: th.Tensor,
                    prior_mean: th.Tensor,
                    prior_std: th.Tensor,
                    scm: TensorSCM,
                    context_mode: List[str] = ['e', 't', 'w_e', 'w_t'],
                    ) -> th.Tensor:
        c_overline = context.clone()
        for i in range(len(context_mode)):
            start_i = i * w_e.size(-1)
            end_i = start_i + w_e.size(-1)
            ci = c_overline[..., start_i:end_i]
            if prior_mean is None:
                prior_mean = th.ones_like(ci)
            if prior_std is None:
                prior_std = th.zeros_like(ci)
            prior_std[prior_std < 1e-6] = 1
            ci = (ci - prior_mean) / prior_std
            if context_mode[i] == 'e':
                ci[~w_e] = 0
            elif context_mode[i] == 't':
                ci[~w_t] = 0
            elif context_mode[i] == 'e+t':
                ci[~(w_t | w_e)] = 0
            else:
                ci = c_overline[..., start_i:end_i]
            ci[ci.isnan() | ci.isinf()] = 0
            c_overline[..., start_i:end_i] = ci
        return c_overline


class EvidenceContextMasked(EvidenceContextConcat):
    # A BundleContextMasked provides concatenated evidence, but masks in context adjacency
    def __init__(self,
                 context_mode: List[str] = ['e', 't', 'w_e', 'w_t'],
                 mask_mode: List[str] = ['fc', 'fc', 'fc', 'fc'],
                 *args,
                 **kwargs,
                 ) -> None:
        super().__init__(context_mode=context_mode, *args, **kwargs)

        assert len(context_mode) == len(mask_mode)
        assert set(mask_mode).issubset({'fc', 'em', 'mb', 'mb1', 'mb2', 'ib'})
        self._mask_mode = mask_mode

    def get_exogenous_full_connected(self) -> th.Tensor:
        # [|U|, dim(V)]
        return th.ones((
            len(self.scm.exogenous_variables),
            sum(self.scm.endogenous_features.values()),
        )).bool()

    def get_exogenous_endo_masked(self,
                                  context_name: str = 'e',
                                  ) -> th.Tensor:
        if context_name in ['e', 'w_e']:
            w = self.w_e
        elif context_name in ['t', 'w_t']:
            w = self.w_t
        elif context_name in ['e+t', 'w_e+w_t']:
            w = self.w_e | self.w_t
        else:
            raise ValueError(f"Unsupported context name {context_name}")
        # [|U|, dim(V)]
        return w[None, :].expand(len(self.scm.exogenous_variables), -1)

    def get_exogenous_markov_boundary(self,
                                      mask_mode: str = 'mb',
                                      context_name: str = 'e',
                                      ) -> th.Tensor:
        assert context_name in\
            ['e', 't', 'e+t', 'w_e', 'w_t', 'w_e+w_t', 'e+h']
        # Augmented graph only
        g = self.scm._aug_graph_dict
        t = self.intervened
        o = {
            'e': self.observed,
            't': self.intervened,
            'e+t': self.observed | self.intervened,
            'w_e': self.observed,
            'w_t': self.intervened,
            'w_e+w_t': self.observed | self.intervened,

        }[context_name]

        # Auxillary graph
        exo = set(self.scm.exogenous_variables)
        mode = {
            'mb': 'normal',
            'mb1': 'endo_only',
            'mb2': 'fake'
        }[mask_mode]
        aux_g = aux_graph(g, t, o, exo, mode)

        # Get markov boundary for each exogenous variable
        mb = {}
        for u in self.scm.exogenous_variables:
            mbu = fast_markov_boundary(aux_g, u)
            mb[u] = mbu

        # Generate masks for variables
        dim_v = self.scm.endogenous_dimensions
        endo_mb = {u: set([v for v in mb[u] if v in dim_v])
                   for u in mb}  # Force endogenous
        adj = th.stack([  # padding for non-mb variables
            batch(pad_variable_mask(endo_mb[u], dim_v), dim_v)
            for u in self.scm.exogenous_variables
        ], dim=0)

        # [|U|, dim(V)]
        return adj

    @property
    def mask_mode(self) -> List[str]:
        return self._mask_mode

    @property
    def adjacency(self) -> th.Tensor | None:
        # Calculate adjacency
        adjs = []
        for i in range(len(self._context_mode)):
            c = self._context_mode[i]
            m = self._mask_mode[i]
            if m == 'fc':
                adjj = self.get_exogenous_full_connected()
            elif m == 'em':
                adjj = self.get_exogenous_endo_masked(c)
            elif m.startswith('mb'):
                adjj = self.get_exogenous_markov_boundary(m, c)
            else:
                raise ValueError(f"Unsupported masking mode {m}")
            adjs.append(adjj)

        # Concate adjacency
        adj = th.cat(adjs, dim=1)

        # Expand for feature size
        adjs.clear()
        for i, feat_u in enumerate(self.scm.exogenous_features.values()):
            adji = adj[i][None, :].expand(feat_u, -1)
            adjs.append(adji)

        adj = th.cat(adjs, dim=0)
        return adj

    @staticmethod
    def context_features(scm: TensorSCM,
                         context_mode: List[str] = ['e', 't', 'w_e', 'w_t'],
                         mask_mode: List[str] = ['fc', 'fc', 'fc', 'fc'],
                         ) -> int:
        assert len(context_mode) == len(mask_mode)
        return len(context_mode) * sum(scm.endogenous_features.values())

    @staticmethod
    def standardize(context: th.Tensor,
                    w_e: th.Tensor,
                    w_t: th.Tensor,
                    prior_mean: th.Tensor,
                    prior_std: th.Tensor,
                    scm: TensorSCM,
                    context_mode: List[str] = ['e', 't', 'w_e', 'w_t'],
                    mask_mode: List[str] = ['fc', 'fc', 'fc', 'fc'],
                    markov_boundary_preprocess: Any = False,
                    ) -> th.Tensor:
        assert len(context_mode) == len(mask_mode)
        return EvidenceContextConcat.standardize(
            context=context,
            w_e=w_e,
            w_t=w_t,
            prior_mean=prior_mean,
            prior_std=prior_std,
            scm=scm,
            context_mode=context_mode,
        )

import concurrent.futures
import torch as th
from typing import *

from dataset.evidence.evidence_custom import *
from dataset.evidence.batched_evidence_joint import *
from dataset.evidence.markov_boundary import *
from dataset.utils import *


class BatchedEvidenceContextConcat(BatchedEvidence):
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

        # Prepare contexts
        self.prepare_contexts()

    def prepare_contexts(self) -> None:
        # Concatenate context tensors
        self._context = th.cat([
            self.get_context_feature(x) for x in self._context_mode
        ], dim=-1)

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

    def __getitem__(self, index: int) -> Evidence:
        return EvidenceContextConcat(
            e=self._e_batched[index],
            t=self._t_batched[index],
            w_e=self._w_e_batched[index],
            w_t=self._w_t_batched[index],
        )

    @property
    def context_mode(self) -> List[str]:
        return self._context_mode

    def get_context(self, index: int) -> th.Tensor | None:
        return self._context[index]

    def get_adjacency(self, index: int) -> th.Tensor | None:
        return None

    @staticmethod
    def unbatched_type() -> Type[EvidenceLike]:
        return EvidenceContextConcat

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
        return EvidenceContextConcat.standardize(
            context=context,
            w_e=w_e,
            w_t=w_t,
            prior_mean=prior_mean,
            prior_std=prior_std,
            scm=scm,
            context_mode=context_mode,
        )


class BatchedEvidenceContextMasked(BatchedEvidenceContextConcat):
    # A BundleContextMasked provides concatenated evidence, but masks in context adjacency
    def __init__(self,
                 context_mode: List[str] = ['e', 't', 'w_e', 'w_t'],
                 mask_mode: List[str] = ['fc', 'fc', 'fc', 'fc'],
                 preprocess_markov_boundary: Any = False,
                 *args,
                 **kwargs,
                 ) -> None:
        super().__init__(context_mode=context_mode, *args, **kwargs)

        assert len(context_mode) == len(mask_mode)
        assert set(mask_mode).issubset({'fc', 'em', 'mb', 'mb1', 'mb2', 'ib'})
        self._mask_mode = mask_mode

        # Preprocess markov boundaries
        self._preprocess_markov_boundary = preprocess_markov_boundary
        self._markov_boundary_preprocessed = None
        if self._preprocess_markov_boundary:
            pass  # self.preprocess_markov_boundary(), to be continued

        # Prepare adjacencies
        self.prepare_adjacency()

    def prepare_adjacency(self) -> None:
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
                adjj = self.get_exogenous_markov_boundary()
            else:
                raise ValueError(f"Unsupported masking mode {m}")
            adjs.append(adjj)

        # Concate adjacency then repeat
        exo_features = th.tensor(list(self._scm.exogenous_features.values()))
        adjacency = th.cat(adjs, dim=2)
        self._adjacency = adjacency.repeat_interleave(exo_features, dim=1)

    def get_exogenous_full_connected(self) -> th.Tensor:
        # [|U|, dim(V)]
        return th.ones((
            len(self),
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
        return w[:, None, :].expand(-1, len(self.scm.exogenous_variables), -1)

    def get_exogenous_markov_boundary(self) -> th.Tensor:
        # If markov boundary not prepared, return dummy mask
        if self._markov_boundary_preprocessed is None:
            return self.get_exogenous_full_connected()

    def get_exogenous_endo_masked_of(self,
                                     context_name: str,
                                     w_t: th.Tensor,
                                     w_e: th.Tensor,
                                     ) -> th.Tensor:
        if context_name in ['e', 'w_e']:
            w = th.clone(w_e)
        elif context_name in ['t', 'w_t']:
            w = th.clone(w_t)
        elif context_name in ['e+t', 'w_e+w_t']:
            w = th.clone(w_e) | th.clone(w_t)
        else:
            raise ValueError(f"Unsupported context name {context_name}")
        # [|U|, dim(V)]
        return w[None, :].expand(len(self.scm.exogenous_variables), -1)

    def get_exogenous_markov_boundary_of(self,
                                         context_name: str,
                                         w_t: th.Tensor,
                                         w_e: th.Tensor,
                                         mask_mode: str = 'mb',
                                         ) -> th.Tensor:
        # Augmented graph only
        g = self.scm._aug_graph_dict

        # Indicator to set
        endo_features = self.scm.endogenous_features
        intervened = indicator_to_set(w_t, endo_features)
        observed = indicator_to_set(w_e, endo_features)

        t = intervened
        o = {
            'e': observed,
            't': intervened,
            'e+t': observed | intervened,
            'w_e': observed,
            'w_t': intervened,
            'w_e+w_t': observed | intervened,
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

    def get_exogenous_inverse_markov_boundary_of(self,
                                                 context_name: str,
                                                 w_t: th.Tensor,
                                                 w_e: th.Tensor,
                                                 ) -> th.Tensor:
        adj = self.get_exogenous_markov_boundary_of(context_name, w_t, w_e)
        return ~adj & self.get_exogenous_endo_masked_of(context_name, w_t, w_e)

    @property
    def mask_mode(self) -> List[str]:
        return self._mask_mode

    def get_adjacency(self, index: int) -> th.Tensor | None:
        # If markov boundary is not preprocessed, lazy prepare again
        endo_feats = sum(self._scm.endogenous_features.values())
        exo_features = th.tensor(list(self._scm.exogenous_features.values()))
        if self._markov_boundary_preprocessed is None:
            for i in range(len(self._context_mode)):
                c = self._context_mode[i]
                m = self._mask_mode[i]
                if not (m.startswith('mb')):
                    continue
                kwargs = dict(
                    context_name=c,
                    w_t=self.w_t[index],
                    w_e=self.w_e[index],
                    mask_mode=m,
                )
                adj = self.get_exogenous_markov_boundary_of(**kwargs)
                self._adjacency[index, :, endo_feats*i:endo_feats*(i+1)] = \
                    adj.repeat_interleave(exo_features, dim=0)

        # Return adjacency
        return self._adjacency[index, :, :]

    @staticmethod
    def unbatched_type() -> Type[EvidenceLike]:
        return EvidenceContextMasked

    @staticmethod
    def context_features(scm: TensorSCM,
                         context_mode: List[str] = ['e', 't', 'w_e', 'w_t'],
                         mask_mode: List[str] = ['fc', 'fc', 'fc', 'fc'],
                         markov_boundary_preprocess: Any = False,
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

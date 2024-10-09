import torch as th
from functools import partial
from torch.utils.data import Dataset, default_collate
from lightning import Trainer
from lightning.pytorch.callbacks import Callback
from typing import *

from common.scm import *
from dataset.evidence.evidence_sampler import *
from dataset.evidence.batched_evidence_sampler import *

"""
EvidenceDataset and BatchedEvidenceDataset will produce data like:

(u, w_j, e, w_e, t, w_t, context, adjacency)

where context and adjacency might be Nones, which are used in proposal sample learning
"""


class EvidenceDataset(Dataset):
    def __init__(self,
                 scm: TensorSCM,
                 sampler: EvidenceSampler,
                 size: int = 16384,
                 max_len_joint: int = 1,
                 ) -> None:
        self._scm = scm
        self._sampler = sampler
        self._size = size
        self._max_len_joint = max_len_joint
        self.flush()

        # prior mean and std
        us = self._scm.batched_sample((size, ))
        self._u_mean = us.mean(dim=0)
        self._u_std = us.std(dim=0)

    def flush(self) -> None:
        self._buffer = [None] * self._size

    def sample_evidence_pair(self) -> Tuple[th.Tensor, EvidenceJoint]:
        u, evidence_joint = self._sampler.sample(True)
        u = batch(u, self._scm.exogenous_dimensions)
        return u, evidence_joint

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, index):
        if self._buffer[index] is None:
            self._buffer[index] = self.sample_evidence_pair()
        u, evidence_joint = self._buffer[index]
        return (u, ) + self.make_evidence_data(evidence_joint, self._max_len_joint)

    @staticmethod
    def make_evidence_data(evidence_joint: EvidenceJoint, max_len: int):
        # Make tensor data
        joint_length = min(len(evidence_joint), max_len)
        joint_idcs = th.arange(max_len)
        stack = partial(evidence_joint.stack, max_len=max_len)
        return (
            joint_idcs < joint_length,  # w_j
            stack('e'),                 # e
            stack('w_e'),               # w_e
            stack('t'),                 # t
            stack('w_t'),               # w_t
            stack('context'),           # context
            stack('adjacency'),         # adjacency
        )

    def get_buffer(self, index):
        assert self._buffer_mode == 'lazy'
        self.__getitem__(index)
        return self._buffer[index]

    @property
    def mean(self) -> th.Tensor:
        return self._u_mean

    @property
    def std(self) -> th.Tensor:
        return self._u_std


class BatchedEvidenceDataset(Dataset):
    def __init__(self,
                 scm: TensorSCM,
                 sampler: BatchedEvidenceSampler,
                 size: int = 16384,
                 max_len_joint: int = 1,
                 ) -> None:
        self._scm = scm
        self._sampler = sampler
        self._size = size
        self._max_len_joint = max_len_joint
        self.flush()
        # prior mean and std
        us = self._scm.batched_sample((size, ))
        self._u_mean = us.mean(dim=0)
        self._u_std = us.std(dim=0)

    def flush(self) -> None:
        self._u, self._batched_evidence = self._sampler.batched_sample(
            batch_size=self._size,
            return_exogenous=True,
        )

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, index):
        # Make tensors
        return (
            self._u[index],                                     # u
            self._batched_evidence.w_j[index],                  # w_j
            self._batched_evidence.e[index],                    # e
            self._batched_evidence.w_e[index],                  # w_e
            self._batched_evidence.t[index],                    # t
            self._batched_evidence.w_t[index],                  # w_t
            self._batched_evidence.get_context(index),          # context
            self._batched_evidence.get_adjacency(index),        # adjacency
        )

    @property
    def mean(self) -> th.Tensor:
        return self._u_mean

    @property
    def std(self) -> th.Tensor:
        return self._u_std

    def save(self, path) -> None:
        th.save({
            'u': self._u,
            'e_batched': self._batched_evidence._batched_evidence._e_batched,
            't_batched': self._batched_evidence._batched_evidence._t_batched,
            'w_e_batched': self._batched_evidence._batched_evidence._w_e_batched,
            'w_t_batched': self._batched_evidence._batched_evidence._w_t_batched,
            'joint_length': self._batched_evidence._joint_length,
        }, path)

    def load(self, path) -> None:
        save = th.load(path)
        self._u = save['u']
        self._batched_evidence = BatchedEvidenceJoint(
            self._sampler._batched_evidence_type(
                scm=self._scm,
                e_batched=save['e_batched'].detach().to('cpu'),
                t_batched=save['t_batched'].detach().to('cpu'),
                w_e_batched=save['w_e_batched'].detach().to('cpu'),
                w_t_batched=save['w_t_batched'].detach().to('cpu'),
                **self._sampler._evidence_kwargs,
            ),
            save['joint_length']
        )


def evidence_collate_fn(batch: List[Any]):
    # Adjacency is [None]
    if batch[0][-1] is None:
        return default_collate([
            sample[:-1] for sample in batch
        ]) + [None]
    # Adjacency is [None]
    if any(adjacency is None for adjacency in batch[0][-1]):
        return default_collate([
            sample[:-1] for sample in batch
        ]) + [None]
    # Adjacency is not None
    else:
        return default_collate(batch)


class FlushCallback(Callback):
    def on_train_epoch_end(self, trainer: Trainer, _) -> None:
        dataset = trainer.train_dataloader.dataset
        assert hasattr(dataset, 'flush'), \
            f"Flush is not allow in {type(dataset).__name__}"
        dataset.flush()

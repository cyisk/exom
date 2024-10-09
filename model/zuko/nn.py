import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, BoolTensor
from typing import *

from dataset.utils import batch_expand


class BatchedMaskedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, **kwargs) -> None:
        super().__init__(in_features, out_features, bias)

    def forward(self, x: Tensor, adjacency: BoolTensor = None) -> Tensor:
        if adjacency is None:
            return super().forward(x)

        assert (self.out_features, self.in_features) == adjacency.shape[-2:]

        # Unbatched adjacency: adjacency = [out_features, in_features]
        if adjacency.dim() == 2:
            return F.linear(x, adjacency * self.weight, self.bias)

        # Batched adjacency: adjacency = [batch_size, out_features, in_features]
        if adjacency.dim() == 3:
            # x = [..., batch_size, in_features]
            assert x.size(-2) == adjacency.size(0)
            x = x[..., None, :]
            w = adjacency * self.weight[None, :, :]
            w = w[*([None]*(x.dim()-3)), ...]
            x = torch.matmul(x, w.transpose(-2, -1))[..., 0, :]
            if self.bias is not None:
                b = self.bias[*([None]*(x.dim()-2)), ...]
                return x + b
            else:
                return x


class BatchedMaskedMLP(nn.Sequential):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: Sequence[int] = (64, 64),
        activation: Callable[[], nn.Module] = None,
        **kwargs,
    ):
        super().__init__()

        self.hidden_features = hidden_features
        self.in_features = in_features
        self.out_features = out_features

        if activation is None:
            activation = nn.ReLU

        # Initialize each layer and mask
        last_features = in_features
        for i, features in enumerate((*self.hidden_features, self.out_features)):
            self.append(BatchedMaskedLinear(last_features, features, **kwargs))
            last_features = features
            self.append(activation())
        self.pop(-1)

        # Initialize unique and inverse
        self.unique = None
        self.inverse = None

    def set_unique(self, unique: Tensor) -> None:
        self.unique = unique

    def set_inverse(self, inverse: Tensor) -> None:
        self.inverse = inverse

    def forward(self, x: Tensor, adjacency: BoolTensor = None) -> Tensor:
        if adjacency is None:
            masks = [None] * (len(self.hidden_features) + 1)
        else:
            assert (self.out_features,
                    self.in_features) == adjacency.shape[-2:]

            # Unbatched mask: mask = [out_features, in_features]
            if adjacency.dim() == 2:
                masks = self.bake_masks(adjacency)

            # Batched mask: mask = [batch_size, out_features, in_features]
            if adjacency.dim() == 3:
                masks = self.batched_bake_masks(adjacency)

        # Forward each layer
        i = 0
        for layer in self:
            if isinstance(layer, BatchedMaskedLinear):
                x = layer(x, masks[i])
                i += 1
            else:
                x = layer(x)
        return x

    def bake_masks(self, adjacency: BoolTensor) -> List[BoolTensor]:
        if self.unique is None or self.inverse is None:
            adjacency, self.inverse = torch.unique(
                adjacency, dim=0, return_inverse=True
            )
        # When too many rows, unique is slow; it is recommanded to be prepared by user
        else:
            adjacency = adjacency[self.unique, :]

        # Copied from zuko.nn.MaskedMLP
        precedence = adjacency.float() @ adjacency.float().t() == adjacency.float().sum(dim=-1)
        masks = []
        for i, features in enumerate((*self.hidden_features, self.out_features)):
            if i > 0:
                mask = precedence[:, indices].bool()
            else:
                mask = adjacency.bool()
            if (~mask).all():
                raise ValueError(
                    "The adjacency matrix leads to a null Jacobian.")
            if i < len(self.hidden_features):
                reachable = mask.sum(dim=-1).nonzero().squeeze(dim=-1)
                indices = reachable[torch.arange(features) % len(reachable)]
                mask = mask[indices]
            else:
                mask = mask[self.inverse]
            masks.append(mask)
        return masks

    def batched_bake_masks(self, adjacency: BoolTensor) -> List[BoolTensor]:
        # Constants
        batch_size = adjacency.size(0)
        idc_batch = torch.arange(batch_size).to(adjacency.device)
        idc_out_features = torch.arange(self.out_features).to(adjacency.device)

        # Make dummy unique if not given
        if self.unique is None or self.inverse is None:
            self.unique = torch.arange(self.out_features)
            self.inverse = torch.arange(self.out_features)
        adjacency = adjacency[:, self.unique, :]

        # Precedence
        adjacency = adjacency.float()
        inout_edges = torch.bmm(adjacency, adjacency.transpose(-2, -1))
        in_edges = adjacency.sum(dim=-1)[:, None, :].expand(
            -1, self.out_features, -1
        )
        precedence = inout_edges == in_edges

        masks = []
        for i, features in enumerate((*self.hidden_features, self.out_features)):
            if i > 0:
                mask = precedence[idc_batch[:, None, None],
                                  idc_out_features[None, :, None],
                                  indices[:, None, :]].bool()
            else:
                mask = adjacency.bool()
            if (~mask).all(dim=(-1, -2)).any():
                raise ValueError(
                    "The adjacency matrix leads to a null Jacobian.")
            if i < len(self.hidden_features):
                # Reachable
                reachable = (mask.sum(dim=-1) != 0).int()
                len_reachable = reachable.sum(dim=-1)
                _, idc_reachable = torch.sort(
                    reachable, dim=-1, descending=True
                )
                # Indices
                idc_features = torch.arange(features).to(adjacency.device)
                map_features = idc_features[None, :] % len_reachable[:, None]
                indices = idc_reachable[idc_batch[:, None], map_features]
                # Masks
                mask = mask[idc_batch[:, None], indices]
            else:
                mask = mask[:, self.inverse]
            masks.append(mask)
        return masks

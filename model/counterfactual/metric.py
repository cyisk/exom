import torch as th

from dataset.utils import *


def log_w(log_pu: th.Tensor,
          log_pue: th.Tensor,
          sample_indicates: th.Tensor,
          self_normalized: bool = False,
          ) -> th.Tensor:
    assert log_pu.shape == log_pue.shape == sample_indicates.shape

    # If pu is 0, then w should be 0
    log_w = log_pu - log_pue
    log_w[log_pu.isinf()] = -th.inf
    log_w[log_pue.isinf()] = -th.inf

    # Self-normalized
    if self_normalized:
        n = th.full_like(log_w, log_w.size(0))
        return log_w - log_w.logsumexp(dim=0)[None, :] + th.log(n)
    return log_w


def effective_sample_proportion(sample_indicates: th.Tensor,
                                reduce: bool = True,
                                ) -> th.Tensor:
    # Equals to indicated rate
    esp = sample_indicates.float().mean(dim=0)
    if reduce:
        return esp.mean()
    else:
        return esp


def fails(sample_indicates: th.Tensor,
          min_effective_sample_proportion: float = 0,
          reduce: bool = True,
          ) -> th.Tensor:
   # Indicated rate <= min
    esp = sample_indicates.float().mean(dim=0)
    fails = (esp <= min_effective_sample_proportion).float()

    if reduce:  # Fail rate of this batch
        return fails.mean(dim=0)
    else:  # Fails
        return fails


def effective_sample_size(log_pu: th.Tensor,
                          log_pue: th.Tensor,
                          sample_indicates: th.Tensor,
                          reduce: bool = True,
                          ) -> th.Tensor:
    # Log normalized weight
    logw = log_w(log_pu, log_pue, sample_indicates)

    # Esitimate effective sample size
    logw[~sample_indicates] = -th.inf
    log_ess = 2 * logw.logsumexp(dim=0) - (2 * logw).logsumexp(dim=0)
    ess = log_ess.exp()

    if reduce:
        return ess.mean()
    else:
        return ess


def effective_sample_entropy(log_pue: th.Tensor,
                             sample_indicates: th.Tensor,
                             reduce: bool = True,
                             ) -> th.Tensor:
    # Effeice entropy
    ese = masked_reduce('mean', -log_pue, sample_indicates, dim=0)
    ese[ese.isnan()] = 0  # For failed case, use 0

    if reduce:
        return ese.mean()
    else:
        return ese

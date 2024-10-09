import abc
import copy
import torch as th
from lightning import LightningModule
from torch.distributions import Distribution
from typing import *

from zuko.distributions import NormalizingFlow, DiagNormal

from dataset.evidence import *
from model.counterfactual.utils import *
from model.counterfactual.metric import *

EvidenceLike = TypeVar('EvidenceLike', bound=Evidence)
EvalResult = Tuple[TensorDict, Dict[str, th.Size]]
BatchedEvalResult = Tuple[th.Tensor, Dict[str, th.Size]]


"""
IS sampling for joint counterfactuals,
designed to work with EvidenceDataset or BatchedEvidenceDataset
"""


class JointCounterfacutalEstimator(abc.ABC, LightningModule):
    def __init__(self,
                 scm: TensorSCM,
                 # Sampling & Inference
                 max_len_joint: int = 1,
                 indicator: Callable | Dict[str, Callable] = l1_indicator(),
                 eval_sample_size: int = 1_000,
                 val_estimate: bool = False,
                 # Standardization
                 prior_u_mean: th.Tensor = 0,
                 prior_u_std: th.Tensor = 1,
                 ) -> None:
        super().__init__()

        # Proxy scm or real scm
        self._scm = scm
        self._eval_sample_size = eval_sample_size
        self._indicator = indicator
        self._max_len_joint = max_len_joint
        self._indicator_eval = MaskedEndoEvaluation(self._scm, indicator)
        self._val_estimate = val_estimate

        # For exogenous standardization
        if not isinstance(prior_u_mean, th.Tensor):
            prior_u_mean = th.tensor(prior_u_mean)
        self.register_buffer('_prior_u_mean', prior_u_mean)
        if not isinstance(prior_u_std, th.Tensor):
            prior_u_std = th.tensor(prior_u_std)
        self.register_buffer('_prior_u_std', prior_u_std)
        self._standardize = StandardTransform(
            self._prior_u_mean, self._prior_u_std
        )

    def on_load_checkpoint(self, checkpoint: th.Dict[str, th.Any]) -> None:
        self._prior_u_mean = checkpoint['prior_u_mean']
        self._prior_u_std = checkpoint['prior_u_std']

    def on_save_checkpoint(self, checkpoint: th.Dict[str, th.Any]) -> None:
        checkpoint['prior_u_mean'] = self._prior_u_mean
        checkpoint['prior_u_std'] = self._prior_u_std

    def get_indicator(self) -> Callable | Dict[str, Callable]:
        return self._indicator

    def set_indicator(self, indicator: Callable | Dict[str, Callable]) -> None:
        self._indicator = indicator
        self._indicator_eval = MaskedEndoEvaluation(self._scm, indicator)

    def set_val_estimate(self, val_estimate: bool = True) -> None:
        self._val_estimate = val_estimate

    def proposal_distribution(self,
                              evidence: Evidence | EvidenceJoint,
                              ) -> Distribution:
        evidence_batch = self.make_evidence_batch(evidence)
        # Note: the distribution returned has batch size 1
        return self.batched_proposal_distribution(evidence_batch)

    def batched_proposal_distribution(self,
                                      evidence_batch: th.Tensor,
                                      ) -> Distribution:
        # Add standard transform
        self._standardize = self._standardize.to(self.device)
        return NormalizingFlow(self._standardize, self.proposal(evidence_batch))

    @abc.abstractmethod
    def proposal(self,
                 evidence_batch: th.Tensor,
                 ) -> Distribution:
        raise NotImplementedError

    def preprocess_batch(self, batch):
        self._standardize = self._standardize.to(self.device)
        return (self._standardize(batch[0]), *batch[1:])

    """Evalutaion: Validation & Testing"""

    def eval_step(self, batch) -> Dict[str, th.Tensor]:
        # Sampling from proposal distribution
        p = self.batched_proposal_distribution(batch[1:])
        u_hat, sample_indicates = self.batched_exogenous_sample(
            p, batch[1:], self._eval_sample_size
        )

        # Log probability
        log_pu = self._scm.to(self.device).batched_log_prob(u_hat)
        log_pue = p.log_prob(u_hat)

        # Effective sample proportion
        esp = effective_sample_proportion(sample_indicates, reduce=False)
        # Fails
        fail = fails(sample_indicates, 0.01, reduce=False)
        # Effective sample size
        ess = effective_sample_size(
            log_pu, log_pue, sample_indicates, reduce=False
        )
        # Effective sample entropy
        ese = effective_sample_entropy(log_pue, sample_indicates, reduce=False)
        # Log likelihood
        ll = log_pue.mean(dim=0)

        # Estimate
        if self._val_estimate:
            estimate = self.batched_estimate(
                p, batch[1:], self._eval_sample_size, True
            )
            return {
                'effective_sample_proportion': esp,
                'fails': fail,
                'effective_sample_size': ess,
                'effective_sample_entropy': ese,
                'log_likelihood': ll,
                'estimate': estimate,
                'u': batch[0],
            }
        else:
            return {
                'effective_sample_proportion': esp,
                'fails': fail,
                'effective_sample_size': ess,
                'effective_sample_entropy': ese,
                'log_likelihood': ll,
            }

    def eval_log(self, result):
        # Get reults
        esps = result['effective_sample_proportion']
        fails = result['fails']
        esss = result['effective_sample_size']
        eses = result['effective_sample_entropy']

        # Masks
        ess_mask = ((fails == 0) & ~esss.isnan())
        ese_mask = ((fails == 0) & ~eses.isnan())

        # Logging
        log_kwargs = dict(prog_bar=True)
        self.log('esp', esps.mean(), **log_kwargs)
        self.log('frate', fails.mean(), **log_kwargs)
        self.log('ess', masked_mean(esss, ess_mask, dim=0), **log_kwargs)
        self.log('ese', masked_mean(eses, ese_mask, dim=0), **log_kwargs)

    def eval_step_start(self, buffer_name: str):
        # Create buffer
        if not hasattr(self, buffer_name):
            setattr(self, buffer_name, [])
        # Clear buffer
        else:
            getattr(self, buffer_name).clear()

    def eval_step_end(self, result, buffer_name: str):
        getattr(self, buffer_name).append(result)

    def eval_epoch_end(self, buffer_name: str, del_buffer: bool = True):
        # Concat results
        buffer = getattr(self, buffer_name)
        assert len(buffer) > 0
        concat_result = {}
        for metric in buffer[0]:
            concat_result[metric] = th.cat([
                result[metric] for result in buffer
            ], dim=-1)

        # Logging
        self.eval_log(concat_result)

    def on_fit_start(self):
        self.val_logs = {}  # Save all val_buffer during traning

    def on_validation_epoch_start(self):
        self.eval_step_start('val_buffer')

    def validation_step(self, batch, batch_idx):
        result = self.eval_step(self.preprocess_batch(batch))
        self.eval_step_end(result, 'val_buffer')

    def on_validation_epoch_end(self):
        self.eval_epoch_end('val_buffer')

        # Log validation buffer when fitting
        if hasattr(self, 'val_logs'):
            self.val_logs[self.current_epoch] = copy.deepcopy(
                getattr(self, 'val_buffer')
            )

    def on_fit_end(self):
        # Save log on trainer
        setattr(self.trainer, 'val_logs', self.val_logs)
        delattr(self, 'val_logs')

    """def on_test_epoch_start(self):
        self.eval_step_start('test_buffer')

    def test_step(self, batch, batch_idx):
        result = self.eval_step(self.preprocess_batch(batch))
        self.eval_step_end(result, 'test_buffer')

    def on_test_epoch_end(self):
        self.eval_epoch_end('test_buffer')
        # Save buffer on trainer
        setattr(self.trainer, 'test_buffer', getattr(self, 'test_buffer'))
        delattr(self, 'test_buffer')"""

    def predict_step(self, batch):
        # Sampling from proposal distribution
        p = self.batched_proposal_distribution(batch[1:])
        u_hat, sample_indicates = self.batched_exogenous_sample(
            p, batch[1:], self._eval_sample_size
        )

        # Double sampling for robustness (still under testing)
        if hasattr(self, '_double_sampling') and self._double_sampling == True:
            u_center = u_hat.mean(dim=0)
            u_std = u_hat.std(dim=0)
            p = DiagNormal(
                u_center,
                th.max(th.ones_like(u_std) * 0.1, u_std),
            )
            u_hat, sample_indicates = self.batched_exogenous_sample(
                p, batch[1:], self._eval_sample_size
            )

        # Log probability
        log_pu = self._scm.to(self.device).batched_log_prob(u_hat)
        log_pue = p.log_prob(u_hat)

        return {
            'indicates': sample_indicates,
            'log_pu': log_pu,
            'log_pue': log_pue,
            'u_hat': u_hat,
        }

    """Operations on proposal distribution"""

    def proposal_eval(self,
                      masked_eval: MaskedEvaluation,
                      p: Distribution,
                      evidence: Evidence | EvidenceJoint,
                      sample_size: int = 1_000,
                      return_sampled_exo: bool = False,
                      ) -> EvalResult | Tuple[th.Tensor, EvalResult]:
        evidence_batch = self.make_evidence_batch(evidence)
        eval_ret = self.batched_proposal_eval(
            masked_eval=masked_eval,
            p=p, evidence_batch=evidence_batch, sample_size=sample_size,
            return_sampled_exo=return_sampled_exo,
        )
        if not return_sampled_exo:
            res, res_dim = eval_ret
            return unbatch(res, res_dim), res_dim
        else:
            u_hat, res, res_dim = eval_ret
            U_hat = unbatch(u_hat, self._scm.endogenous_dimensions)
            return U_hat, unbatch(res, res_dim), res_dim

    @th.no_grad()
    def batched_proposal_eval(self,
                              masked_eval: MaskedEvaluation,
                              p: Distribution,
                              evidence_batch: th.Tensor,
                              sample_size: int = 1_000,
                              return_sampled_exo: bool = False,
                              ) -> BatchedEvalResult | Tuple[th.Tensor, BatchedEvalResult]:
        """
        Evaluate endogenous of exogenous sampled from proposal distribution with ground truth
        a batch is sampled from EvidenceDataset or BatchedEvidenceDataset

        w_j: [batch_size, joint_size]
        e, w_e, t, w_t: [batch_size, joint_size, endo_features]
        """
        w_j, e, w_e, t, w_t, _, _ = evidence_batch
        assert w_j.dim() == 2
        assert e.dim() == w_e.dim() == t.dim() == w_t.dim() == 3

        # Sample exogenous from proposal distribution
        u_hat = p.sample((sample_size, ))
        assert u_hat.dim() == 3

        # Expand u_hat for each joint
        u_flat = u_hat[:, :, None, :].expand(-1, -1, w_j.size(1), -1)

        # Expand w_j, e, w_e, t, w_t for each sample
        w_j = w_j[None, :, :].expand(u_hat.size(0), -1, -1)
        e = e[None, :, :, :].expand(u_hat.size(0), -1, -1, -1)
        w_e = w_e[None, :, :, :].expand(u_hat.size(0), -1, -1, -1)
        t = t[None, :, :, :].expand(u_hat.size(0), -1, -1, -1)
        w_t = w_t[None, :, :, :].expand(u_hat.size(0), -1, -1, -1)

        # Masked select for valid (unmasked) joint
        u_flat = feature_masked_select(u_flat, w_j)
        e = feature_masked_select(e, w_j)
        w_e = feature_masked_select(w_e, w_j)
        t = feature_masked_select(t, w_j)
        w_t = feature_masked_select(w_t, w_j)

        # Inference from estimated exogenous
        e_hat = self._scm.to(self.device).batched_call(u_flat, t, w_t, w_e)

        # Masked evaluation
        res, res_dim = masked_eval.eval(e_hat, e, w_e, w_j)

        # Masked scatter for each joint
        # res: [sample_size, batch_size, "according to res_dim"]
        if not return_sampled_exo:
            return res, res_dim
        else:
            # u_hat: [sample_size, batch_size, exo_features]
            return u_hat, (res, res_dim)

    def exogenous_sample(self,
                         p: Distribution,
                         evidence: Evidence | EvidenceJoint,
                         sample_size: int = 1_000,
                         ) -> Tuple[TensorDict, th.Tensor]:
        evidence_batch = self.make_evidence_batch(evidence)
        u_hat, sample_indicates = self.batched_exogenous_sample(
            p, evidence_batch, sample_size
        )
        U_hat = unbatch(u_hat[0, ...], self._scm.endogenous_dimensions)
        return U_hat, sample_indicates

    @th.no_grad()
    def batched_exogenous_sample(self,
                                 p: Distribution,
                                 evidence_batch: th.Tensor,
                                 sample_size: int = 1_000,
                                 ) -> Tuple[th.Tensor, th.Tensor]:
        """
        Sample from proposal distribution, but only indicator = 1 is kept
        """
        # u_hat: [sample_size, batch_size, exo_features]
        # indicates: [sample_size, batch_size, "according to _"]
        u_hat, (indicates, _) = self.batched_proposal_eval(
            masked_eval=self._indicator_eval,
            p=p, evidence_batch=evidence_batch, sample_size=sample_size,
            return_sampled_exo=True,
        )
        if indicates.dim() > 2:
            # Only when all features in a sample of a batch indicates True, does it indicate True
            sample_indicates = indicates.all(
                dim=list(range(2, indicates.dim())))
        else:
            # If indicates is exactly sample wise
            sample_indicates = indicates
        # Support check
        log_prob = self._scm.batched_log_prob(u_hat)
        sample_indicates[th.isinf(log_prob)] = False
        # sample_indicates: [sample_size, batch_size]
        return u_hat, sample_indicates

    def estimate(self,
                 p: Distribution,
                 evidence: Evidence | EvidenceJoint,
                 sample_size: int = 1_000,
                 as_log: bool = False,
                 ) -> float:
        evidence_batch = self.make_evidence_batch(evidence)
        log_p = self.batched_estimate(p, evidence_batch, sample_size, as_log)
        return log_p[0].item()

    @ th.no_grad()
    def batched_estimate(self,
                         p: Distribution,
                         evidence_batch: th.Tensor,
                         sample_size: int = 1_000,
                         as_log: bool = False,
                         ) -> th.Tensor:
        # Sample
        u_hat, sample_indicates = self.batched_exogenous_sample(
            p, evidence_batch, sample_size
        )

        # Log prob from exogenous and estimated exogenous
        log_pu = self._scm.to(self.device).batched_log_prob(u_hat)
        log_pue = p.log_prob(u_hat)

        # Log w
        logw = log_w(log_pu, log_pue, sample_indicates)

        # Regularize
        logn = th.log(th.tensor(u_hat.size(0)))

        # Esitimate prob
        logw[~sample_indicates] = -th.inf
        log_p = logw.logsumexp(dim=0) - logn
        if th.all(th.isnan(log_p)):  # zero if no valid sample
            log_p = th.zeros_like(log_p)

        if as_log:
            return log_p
        else:
            return th.exp(log_p)

    def make_evidence_batch(self,
                            evidence: Evidence | EvidenceJoint,
                            unsqueeze_batch_size: bool = True,
                            ) -> List[th.Tensor]:
        if isinstance(evidence, Evidence):
            evidence_joint = EvidenceJoint([evidence])
        else:
            evidence_joint = evidence
        evidence_batch = EvidenceDataset.make_evidence_data(
            evidence_joint, self._max_len_joint
        )
        evidence_batch = list(evidence_batch)
        for i in range(len(evidence_batch)):
            if not isinstance(evidence_batch[i], th.Tensor):
                continue
            if unsqueeze_batch_size:
                # unsqueeze for batch size 1
                evidence_batch[i] = evidence_batch[i][None, :]
            evidence_batch[i] = evidence_batch[i].to(device=self.device)
        return tuple(evidence_batch)

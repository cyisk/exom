import torch as th
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from math import prod
from typing import *

from zuko.nn import MLP

from common.graph.causal import *
from common.scm import *
from model.proxy_scm.ncm.ncm import NCM
from model.proxy_scm.causal_nf.metric import maximum_mean_discrepancy
from dataset.utils import *


class GANNCM(LightningModule):
    def __init__(self,
                 # Structual Causal Model
                 causal_graph: DirectedMixedGraph,
                 endogenous_dimensions: Dict[str, th.Size],
                 endogenous_logits: Dict[str, int],  # binary only
                 exogenous_dimensions: Dict[str, th.Size],
                 exogenous_distribution_type: str = 'gaussian',
                 # Estimator
                 ncm_hidden_features: Sequence[int] = [64, 64],
                 critic_hidden_features: Sequence[int] = [64, 64],
                 # Learning
                 learning_rate: float = 2e-5,
                 n_critics: int = 5,
                 ) -> None:
        # This implementation only works on learning from observational distribution
        super().__init__()
        self.automatic_optimization = False
        self._learning_rate = learning_rate
        self._n_critics = n_critics
        self._endo_dimensions = {
            x: endogenous_dimensions[x]
            for x in sorted(endogenous_dimensions)
        }
        self._endo_features = {
            v: prod(dim) for v, dim in endogenous_dimensions.items()
        }
        self._endo_dimensions = {
            x: endogenous_dimensions[x]
            for x in sorted(endogenous_dimensions)
        }
        self._endo_features = {
            v: prod(dim) for v, dim in self._endo_dimensions.items()
        }

        # General graph
        assert isinstance(causal_graph, DirectedMixedGraph)
        assert causal_graph.is_admg  # Only work on admg
        self._causal_graph = causal_graph

        # Initialize NCM
        self._ncm = NCM(
            causal_graph=causal_graph,
            endogenous_dimensions=endogenous_dimensions,
            endogenous_logits=endogenous_logits,
            exogenous_dimensions=exogenous_dimensions,
            exogenous_distribution_type=exogenous_distribution_type,
            hidden_features=ncm_hidden_features,
        )

        # Critic
        """ # Vanilla GAN
        self._critic = nn.Sequential(MLP(
            in_features=sum(self._endo_features.values()),
            out_features=1,
            hidden_features=critic_hidden_features,
        ), nn.Sigmoid())
        """
        self._critic = MLP(
            in_features=sum(self._endo_features.values()),
            out_features=1,
            hidden_features=critic_hidden_features,
        )

    def configure_optimizers(self):
        g_optimizer = th.optim.RMSprop(
            self.parameters(),
            lr=self._learning_rate,
        )
        d_optimizer = th.optim.RMSprop(
            self.parameters(),
            lr=self._learning_rate,
        )
        return g_optimizer, d_optimizer

    def adversarial_loss(self, y_hat: th.Tensor, is_fake: bool = False):
        """ # Vanilla GAN
        if is_fake:
            return F.binary_cross_entropy(y_hat, th.zeros_like(y_hat))
        return F.binary_cross_entropy(y_hat, th.ones_like(y_hat))
        """
        if is_fake:
            return y_hat.mean()
        return -y_hat.mean()

    def training_step(self, batch, batch_idx):
        g_optimizer, d_optimizer = self.optimizers()

        # Ground truth
        batch_size = batch.size(0)
        v = batch
        # v = self.expand_for_logits(batch)

        if batch_idx % self._n_critics == 0:
            # Generator loss
            z = self._ncm.batched_noise(batch_size)
            v_hat = self._ncm(z)
            # g_loss = self.adversarial_loss(self._critic(v_hat), False)
            g_loss = self.adversarial_loss(self._critic(v_hat), False)
            self.log('g_loss', g_loss, on_step=True, prog_bar=True)
            self.manual_backward(g_loss)
            g_optimizer.step()
            g_optimizer.zero_grad()
            self.untoggle_optimizer(g_optimizer)

        # Critic loss
        z = self._ncm.batched_noise(batch_size)
        v_hat = self._ncm(z).detach()
        # d_loss_real = self.adversarial_loss(self._critic(v), False)
        # d_loss_fake = self.adversarial_loss(self._critic(v_hat), True)
        d_loss_real = self.adversarial_loss(self._critic(v), False)
        d_loss_fake = self.adversarial_loss(self._critic(v_hat), True)
        d_loss = d_loss_real + d_loss_fake + 10 * \
            self.compute_gradient_penalty(v, v_hat, 0.01)
        self.log('d_loss', d_loss, on_step=True, prog_bar=True)
        self.log('diff', th.abs(d_loss_real + d_loss_fake),
                 on_step=True, prog_bar=True)
        self.clip_gradients(d_optimizer, 0.01)
        self.manual_backward(d_loss)
        d_optimizer.step()
        d_optimizer.zero_grad()
        self.untoggle_optimizer(d_optimizer)

    def compute_gradient_penalty(self, v, v_hat, c):
        """Calculates the gradient penalty loss for WGAN GP"""
        alpha = th.rand((v.size(0), 1), device=self.device)
        interps = (alpha * v + ((1 - alpha) * v_hat)).requires_grad_(True)
        interps = interps.to(self.device)
        d_interps = self._critic(interps)
        fake = th.ones((v.size(0), 1)).to(self.device)
        # Get gradient w.r.t. interps
        gradients = th.autograd.grad(
            outputs=d_interps,
            inputs=interps,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1).to(self.device)
        gradients_norm = th.max(
            gradients, th.zeros_like(gradients)).norm(2, dim=1)
        gradient_penalty = ((gradients_norm - c) ** 2).mean()
        return gradient_penalty

    def validation_step(self, batch, batch_idx):
        # Estimated observational distribution
        v = batch

        # Samples
        z = self._ncm.batched_noise(batch.size(0))
        x = self._ncm.batched_call(z, soft=False).detach()
        mmd = maximum_mean_discrepancy(v.float(), x.float())
        diff = th.abs((x.int() == 1).float().sum(dim=0) / x.size(0) -
                      (v == 1).float().sum(dim=0) / v.size(0))
        # print(diff)

        self.log('mmd', mmd, on_step=False, on_epoch=True, prog_bar=True)
        self.log('ddf', diff.sum(), on_step=False,
                 on_epoch=True, prog_bar=True)
        return mmd

    def to(self, device: Optional[str | th.device | int] = None, *args, **kwargs):
        self._ncm = self._ncm.to(device, *args, **kwargs)
        return super().to(device=device, *args, **kwargs)

    """
    def expand_for_logits(self, v: th.Tensor) -> th.Tensor:
        return batch({
            x: F.one_hot(
                x_val.long(),
                num_classes=self._endo_logits[x],
            ).float()
            for x, x_val in unbatch(v, self._endo_dimensions).items()
        }, self._endo_dimensions)
    """

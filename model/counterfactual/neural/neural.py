from model.counterfactual.exo_match.exo_match import *


class NeuralIS(ExogenousMatch):
    def __init__(self,
                 # Structual Causal Model
                 scm: TensorSCM,
                 # Evidence Type
                 evidence_type: Type[EvidenceLike],
                 evidence_kwargs: Optional[Dict[str, Any]] = {},
                 # Density estimator
                 density_estimator_kwargs: Optional[Dict[str, Any]] = {},
                 base_distribution_type: str = 'gaussian',
                 base_distribution_kwargs: Optional[Dict[str, Any]] = {},
                 # Learning & Sampling & Inference
                 learning_rate: float = 1e-3,
                 max_len_joint: int = 1,
                 indicator: Callable | Dict[str, Callable] = l1_indicator(),
                 eval_sample_size: int = 1_000,
                 cold_starts: int = 0,
                 train_sample_size: int = 1_000,
                 # Standardization
                 prior_u_mean: th.Tensor = 0,
                 prior_u_std: th.Tensor = 1,
                 prior_update: bool = True,
                 prior_cold_starts: int = int(1e4),
                 prior_max_updates: int = 100,
                 ) -> None:
        super().__init__(
            scm=scm,
            evidence_type=evidence_type,
            evidence_kwargs=evidence_kwargs,
            density_estimator_type='nice',
            density_estimator_kwargs=density_estimator_kwargs,
            base_distribution_type=base_distribution_type,
            base_distribution_kwargs=base_distribution_kwargs,
            learning_rate=learning_rate,
            max_len_joint=max_len_joint,
            indicator=indicator,
            eval_sample_size=eval_sample_size,
            prior_u_mean=prior_u_mean,
            prior_u_std=prior_u_std,
            prior_update=prior_update,
            prior_cold_starts=prior_cold_starts,
            prior_max_updates=prior_max_updates,
        )
        print(train_sample_size)
        self._cold_starts = cold_starts
        self.automatic_optimization = False
        self._train_sample_size = train_sample_size

    def configure_optimizers(self):
        exom_optimizer = th.optim.AdamW(
            self.parameters(),
            lr=self._learning_rate,
            weight_decay=1e-6,
        )
        nis_optimizer = th.optim.AdamW(
            self.parameters(),
            lr=self._learning_rate,
            weight_decay=1e-6,
        )
        exom_scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=exom_optimizer, mode='min', factor=0.5, patience=5,
        )
        nis_scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=nis_optimizer, mode='min', factor=0.5, patience=5,
        )
        return ({
            'optimizer': exom_optimizer,
            'lr_scheduler': {
                'scheduler': exom_scheduler,
                'monitor': 'loss_exom',
                'frequency': 1,
            }
        }, {
            'optimizer': nis_optimizer,
            'lr_scheduler': {
                'scheduler': nis_scheduler,
                'monitor': 'nis_loss',
                'frequency': 1,
            }
        })

    def training_step(self, batch, batch_idx):
        # Copied from ExogenousMatch
        u, w_j, e, w_e, t, w_t, context, adjacency = self.preprocess_batch(
            batch
        )
        if self._prior_update:
            mask = feature_expand(w_j, th.Size((e.size(-1), )))
            self.update_prior(e, w_e & mask)
            self.update_prior(t, w_t & mask)
        context = self.standardize_context(context, w_e.bool(), w_t.bool())
        p = self.forward(context, adjacency, w_j)

        exom_optimizer, nis_optimizer = self.optimizers()
        if self.current_epoch >= self._cold_starts:
            # X^2 Loss
            u_hat, sample_indicates = self.batched_exogenous_sample(
                p, batch[1:], self._train_sample_size
            )
            logw = 2 * self._scm.to(self.device).batched_log_prob(u_hat)\
                - p.log_prob(u_hat).detach()
            logw[~sample_indicates] = -th.inf
            loss = (-p.log_prob(u_hat) * th.exp(logw)).mean()
            self.manual_backward(loss)
            exom_optimizer.step()
            exom_optimizer.zero_grad()
            self.untoggle_optimizer(exom_optimizer)
            self.log('nis_loss', loss, on_epoch=True,
                     on_step=False, prog_bar=True)
            self.log('nis_esp', sample_indicates.float().sum() / sample_indicates.nelement(), on_epoch=True,
                     on_step=False, prog_bar=True)
        else:
            loss = -p.log_prob(u).mean()
            self.manual_backward(loss)
            nis_optimizer.step()
            nis_optimizer.zero_grad()
            self.untoggle_optimizer(nis_optimizer)
            self.log('exom_loss', loss, on_epoch=True,
                     on_step=False, prog_bar=True)
        return loss

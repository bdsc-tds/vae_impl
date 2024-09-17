from typing import Any
import lightning.pytorch as pl
import torch
from abc import ABC, abstractmethod


class BaseVAE(ABC, pl.LightningModule):
    """An abstract class for variational autoencoders (VAE).

    This class cannot be used directly. Another class must inherit this class and implement the following methods:
        - `set_encoder`
        - `set_fc_mu`
        - `set_fc_log_var`
        - `set_decoder`
    """

    def __init__(self, use_kl_mc: bool = True, lr: float = 1e-3):
        super().__init__()

        self.save_hyperparameters()

        # encoder
        self.encoder = None

        # distribution parameters
        self.fc_mu = None
        self.fc_log_var = None

        # decoder
        self.decoder = None

        # for the gaussian likelihood
        self.log_scale = torch.nn.Parameter(torch.Tensor([0.0]))

        self.use_kl_mc = use_kl_mc
        self.lr = lr

    @abstractmethod
    def set_encoder(self, *args: Any, **kwargs: Any) -> torch.nn.Module:
        pass

    @abstractmethod
    def set_fc_mu(self, *args: Any, **kwargs: Any) -> torch.nn.Module:
        pass

    @abstractmethod
    def set_fc_log_var(self, *args: Any, **kwargs: Any) -> torch.nn.Module:
        pass

    @abstractmethod
    def set_decoder(self, *args: Any, **kwargs: Any) -> torch.nn.Module:
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def gaussian_likelihood(
            self,
            mean: float | torch.Tensor,
            logscale: float | torch.Tensor,
            sample: torch.Tensor,
    ) -> torch.Tensor:
        scale = torch.exp(logscale)
        dist = torch.distributions.Normal(mean, scale)
        log_pxz = dist.log_prob(sample)
        return log_pxz.sum(dim=(1, 2, 3))

    def kl_divergence_gaussian(
            self, mu: float | torch.Tensor, std: float | torch.Tensor
    ) -> torch.Tensor:
        """Assuming the latent random variables to follow Gaussian distribution.
        https://arxiv.org/abs/1312.6114

        Args:
            mu (float | torch.Tensor): a vector of means of Gaussian distributions for latent random variables
            std (float | torch.Tensor): a vector of std of Gaussian distributions for latent random variables

        Returns:
            torch.Tensor: KL divergence
        """
        return -0.5 * (1 + 2 * torch.log(std) - mu ** 2 - std ** 2).sum(-1)

    def kl_divergence_mc(
            self, z: torch.Tensor, mu: float | torch.Tensor, std: float | torch.Tensor
    ) -> torch.Tensor:
        # --------------------------
        # Monte Carlo KL divergence
        # --------------------------

        # prior distribution and probability
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        log_pz = p.log_prob(z)

        # approximation of posterior distribution and probability
        q = torch.distributions.Normal(mu, std)
        log_qzx = q.log_prob(z)

        # kl
        kl = log_qzx - log_pz
        kl = kl.sum(-1)

        return kl

    def get_latent(
            self, x: torch.Tensor = None, batch: tuple[torch.Tensor] = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        assert x is not None or batch is not None

        if batch is not None:
            x, y = batch

        x_encoded = self.encoder(x)
        mu, log_var = self.fc_mu(x_encoded), self.fc_log_var(x_encoded)

        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()  # reparameterization trick inside

        return z if batch is None else (z, y)

    def forward(
            self, x: torch.Tensor = None, batch: tuple[torch.Tensor] = None
    ) -> torch.Tensor:
        assert x is not None or batch is not None

        if batch is not None:
            x, _ = batch

        return self.decoder(self.get_latent(x=x))

    def training_step(self, batch: tuple[torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, _ = batch

        # encode x to get the mu and variance parameters
        x_encoded = self.encoder(x)
        mu, log_var = self.fc_mu(x_encoded), self.fc_log_var(x_encoded)

        # sample z from q
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()

        # decoded
        x_hat = self.decoder(z)

        # likelihood
        likelihood = self.gaussian_likelihood(x_hat, self.log_scale, x)

        # kl
        kl = (
            self.kl_divergence_mc(z, mu, std)
            if self.use_kl_mc
            else self.kl_divergence_gaussian(mu, std)
        )

        # elbo
        elbo = likelihood - kl
        elbo = elbo.mean()

        self.log_dict(
            {
                "training_loss": -elbo,
                "elbo": elbo,
                "kl": kl.mean(),
                "likelihood": likelihood.mean(),
            }
        )

        return -elbo

    def eval_perf(
            self, batch: tuple[torch.Tensor], batch_idx: int, log_prefix: str
    ) -> torch.Tensor:
        x, _ = batch
        x_hat = self.forward(x)
        likelihood = self.gaussian_likelihood(x_hat, self.log_scale, x).mean()

        self.log_dict({f"{log_prefix}_loss": -likelihood, "likelihood": likelihood})

        return -likelihood

    def on_validation_model_eval(self, *args, **kwargs):
        super().on_validation_model_eval(*args, **kwargs)
        # torch.set_grad_enabled(True)

    def validation_step(
            self, batch: tuple[torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        return self.eval_perf(batch, batch_idx, log_prefix="val")

    def on_test_model_eval(self, *args, **kwargs):
        super().on_test_model_eval(*args, **kwargs)
        # torch.set_grad_enabled(True)

    def test_step(self, batch: tuple[torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self.eval_perf(batch, batch_idx, log_prefix="test")

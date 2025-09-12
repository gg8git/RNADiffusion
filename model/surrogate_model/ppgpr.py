# ppgpr
import math
import gpytorch
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.priors import LogNormalPrior
from gpytorch.constraints.constraints import GreaterThan

from .base import DenseNetwork

# Multi-task Variational GP:
# https://docs.gpytorch.ai/en/v1.4.2/examples/04_Variational_and_Approximate_GPs/SVGP_Multitask_GP_Regression.html


class GPModel(ApproximateGP):
    def __init__(self, inducing_points, likelihood):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super(GPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.num_outputs = 1
        self.likelihood = likelihood

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def posterior(self, X, output_indices=None, observation_noise=False, *args, **kwargs) -> GPyTorchPosterior:
        self.eval()  # make sure model is in eval mode
        # self.model.eval()
        self.likelihood.eval()
        dist = self.likelihood(self(X))

        return GPyTorchPosterior(dist)


# gp model with deep kernel
class GPModelDKL(ApproximateGP):
    def __init__(self, inducing_points, likelihood, hidden_dims=(256, 256), ls_prior=False):
        feature_extractor = DenseNetwork(input_dim=inducing_points.size(-1), hidden_dims=hidden_dims).to(
            inducing_points.device
        )
        inducing_points = feature_extractor(inducing_points)
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super(GPModelDKL, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

        if ls_prior:
            dims = inducing_points.size(1)
            loc = 1.4 + math.log(dims) * 0.5
            scale = (1.73205 ** 2 + math.log(dims) * 0) ** 0.5
            lengthscale_prior=LogNormalPrior(loc=loc, scale=scale)
            lengthscale_constraint=GreaterThan(1e-4)
            self.covar_module = gpytorch.kernels.RBFKernel(ard_num_dims=dims, lengthscale_prior=lengthscale_prior, lengthscale_constraint=lengthscale_constraint)
            self.covar_module = gpytorch.kernels.ScaleKernel(self.covar_module)

        self.num_outputs = 1  # must be one
        self.likelihood = likelihood
        self.feature_extractor = feature_extractor

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def __call__(self, x, *args, **kwargs):
        x = self.feature_extractor(x)
        return super().__call__(x, *args, **kwargs)

    def posterior(self, X, output_indices=None, observation_noise=False, *args, **kwargs) -> GPyTorchPosterior:
        self.eval()  # make sure model is in eval mode
        # self.model.eval()
        self.likelihood.eval()
        dist = self.likelihood(self(X))

        return GPyTorchPosterior(dist)

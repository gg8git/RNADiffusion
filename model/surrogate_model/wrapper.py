from botorch.models.gpytorch import GPyTorchModel
from botorch.posteriors import GPyTorchPosterior

from .ppgpr import GPModelDKL

class BoTorchDKLModelWrapper(GPyTorchModel):
    def __init__(self, model: GPModelDKL):
        super().__init__()
        self.model = model
        self._num_outputs = 1

    @property
    def num_outputs(self) -> int:
        return self._num_outputs

    def posterior(self, X, output_indices=None, observation_noise=False, **kwargs) -> GPyTorchPosterior:
        return self.model.posterior(X, output_indices=output_indices, observation_noise=observation_noise)

    def eval(self):
        self.model.eval()
        return self

    def train(self, mode: bool = True):
        self.model.train(mode)
        return self

    def to(self, device):
        self.model = self.model.to(device)
        return self
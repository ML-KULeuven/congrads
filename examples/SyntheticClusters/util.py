import torch
from torch.nn import NLLLoss, Softmax
from torch.nn.modules.loss import _Loss

from congrads.networks.registry import MLPNetwork


class MLPNetworkWithSoftmax(MLPNetwork):
    def __init__(
        self,
        n_inputs,
        n_outputs,
        n_hidden_layers=3,
        hidden_dim=35,
        activation=None,
    ):
        super().__init__(n_inputs, n_outputs, n_hidden_layers, hidden_dim, activation)
        self.softmax = Softmax(dim=1)

    def forward(self, batch: torch.Tensor):
        output = super().forward(batch)
        output["output"] = self.softmax(output["output"])
        return output


class NNLLossFromProb(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction="mean", epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon
        self.criterion = NLLLoss(size_average=size_average, reduce=reduce, reduction=reduction)

    def forward(self, input: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
        log_probs = torch.log(torch.clamp(input, min=1e-8))
        return self.criterion(log_probs, target)

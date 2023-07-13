"""A module to house optimisers."""
from __future__ import annotations
import abc


class Optimiser(abc.ABC):
    """Abstract base class for optimisers."""

    def __init__(self, params: list) -> None:
        """Initialise an optimiser."""
        self.params = params

    @abc.abstractmethod
    def step(self) -> None:
        """Take a step in the optimiser."""
        pass

    def zero_grad(self) -> None:
        """Zero the gradients of the optimiser."""
        for param in self.params:
            param.grad = 0.0
        # TODO(leon):
        #   Set all other gradients to zero, including those that are not
        #   tracked by the optimiser, for all Scalars in the graph.


class SGDOptimiser(Optimiser):
    """Stochastic gradient descent."""

    def __init__(self, params: list, lr: float = 0.01, momentum: float = 0.0) -> None:
        """Initialise stochastic gradient descent."""
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.velocities = [0.0 for _ in params]

    def step(self) -> None:
        """Take a step in the optimiser."""
        for i, param in enumerate(self.params):
            if param.requires_grad:
                grad_step: float = self.lr * param.grad.data
                self.velocities[i] = self.momentum * self.velocities[i] + grad_step
                param.data -= self.velocities[i]

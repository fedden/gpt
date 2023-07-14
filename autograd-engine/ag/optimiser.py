"""A module to house optimisers."""
from __future__ import annotations
import abc

import ag


def no_grad(func):
    """Decorate a function to ensure no gradients are tracked."""

    def wrapper(*args, **kwargs):
        with ag.no_grad():
            return func(*args, **kwargs)

    return wrapper


class Optimiser(abc.ABC):
    """Abstract base class for optimisers."""

    def __init__(self, params: list[ag.Tensor]) -> None:
        """Initialise an optimiser."""
        self._check_params(params)
        self.params = params

    @staticmethod
    def _check_params(params: list[ag.Tensor]):
        """Ensure params are sensible."""
        assert all(isinstance(p, ag.Tensor) for p in params)
        assert all(p.requires_grad for p in params)

    @abc.abstractmethod
    def step(self) -> None:
        """Take a step in the optimiser."""
        pass

    def zero_grad(self) -> None:
        """Zero the gradients of the optimiser."""
        for param in self.params:
            param.zero_grad()


class SGDOptimiser(Optimiser):
    """Stochastic gradient descent."""

    def __init__(self, params: list, lr: float = 0.01, momentum: float = 0.0) -> None:
        """Initialise stochastic gradient descent."""
        super().__init__(params)
        self.lr: float = lr
        self.momentum: float = momentum
        self.velocities: list[ag.Tensor] = [ag.zeros_like(p) for p in params]

    @no_grad
    def step(self) -> None:
        """Take a step in the optimiser."""
        for i, param in enumerate(self.params):
            if param.requires_grad:
                grad_step: ag.Tensor = param.grad * self.lr
                self.velocities[i] = self.velocities[i] * self.momentum + grad_step
                param -= self.velocities[i]

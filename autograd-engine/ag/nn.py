"""Neural Network related modules."""
import abc
import math
from typing import Any

import ag
from ag.tensor import Parameter, Tensor


class Module(abc.ABC):
    """Abstract base class for modules."""

    @abc.abstractmethod
    def forward(self, *args, **kwargs) -> None:
        """Forward pass of the module."""

    def __call__(self, *args, **kwargs) -> Any:
        """Call the forward method."""
        return self.forward(*args, **kwargs)

    def parameter_list(self) -> list:
        """Return a list of parameters by recursive search."""
        params: list = []
        for attr in ag.utils.flatten(self.__dict__.values()):
            if isinstance(attr, Parameter):
                params.append(attr)
            elif isinstance(attr, Module):
                params += attr.parameter_list()
        return params


class Linear(Module):
    """Linear layer."""

    def __init__(self, in_features: int, out_features: int) -> None:
        """Initialise a linear layer."""
        bound: float = 1 / math.sqrt(in_features)
        self.weight = Parameter(
            ag.random.uniform(
                shape=(in_features, out_features),
                low=-bound,
                high=bound,
            )
        )
        self.bias = Parameter(
            ag.random.uniform(
                shape=(out_features,),
                low=-bound,
                high=bound,
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the module."""
        return x @ self.weight + self.bias


class Sequential(Module):
    """Sequential container."""

    def __init__(self, *args: Module) -> None:
        """Initialise a sequential container."""
        self.modules = args
        assert isinstance(self.modules, (list, tuple))
        assert all(isinstance(module, Module) for module in self.modules)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the module."""
        for module in self.modules:
            x = module(x)
        return x


class ReLU(Module):
    """ReLU activation function."""

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the module."""
        return x.relu()


class Tanh(Module):
    """Tanh activation function."""

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the module."""
        return x.tanh()

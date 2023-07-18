"""Neural Network related modules."""
import abc


class Module(abc.ABC):
    """Abstract base class for modules."""

    @abc.abstractmethod
    def forward(self, *args, **kwargs) -> None:
        """Forward pass of the module."""

    def __call__(self, *args, **kwargs) -> None:
        """Call the forward method."""
        self.forward(*args, **kwargs)

    def parameter_list(self) -> list:
        """Return a list of parameters."""
        return [p for p in self.__dict__.values() if isinstance(p, Parameter)]


class L

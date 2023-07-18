"""A basic autograd library for learning purposes."""
from __future__ import annotations
import math

import numpy as np
import pytest
import torch

import ag
from ag.tensor import Tensor


@pytest.mark.parametrize("axis", [0, 1, None])
def test_concatenate(axis: int) -> None:
    """Test concatenate."""
    np_a = np.arange(4).astype(float).reshape(2, 2)
    np_b = np.arange(4).astype(float).reshape(2, 2)
    ag_a = Tensor(np_a, requires_grad=True)
    ag_b = Tensor(np_b, requires_grad=True)
    np_c = np.concatenate((np_a, np_b), axis=axis)
    ag_c = ag.concatenate((ag_a, ag_b), axis=axis)
    assert np.allclose(np_c, ag_c.numpy()), f"\nnp\n{np_c}\n\n!=\n\nag\n{ag_c.numpy()}"


@pytest.mark.parametrize("shape,reps", [((2, 2), (2, 2)), ((2, 2), (1, 2))])
def test_tile(shape: tuple[int, ...], reps: tuple[int, ...]) -> None:
    """Test tile."""
    np_a = np.arange(np.prod(shape)).astype(float).reshape(shape)
    ag_a = Tensor(np_a, requires_grad=True)
    np_b = np.tile(np_a, reps=reps)
    ag_b = ag_a.tile(reps=reps)
    assert np.allclose(np_b, ag_b.numpy()), f"\nnp\n{np_b}\n\n!=\n\nag\n{ag_b.numpy()}"


def test_basic_elementwise_ops_and_broadcasting():
    """Test basic elementwise operations and broadcasting."""
    ag_a = Tensor([[1, 2, 3]], requires_grad=True)
    ag_b = Tensor(np.ones((3, 3)), requires_grad=True)
    (ag_a - 2) * (ag_b + 10)


def test_slicing() -> None:
    """Test slicing."""
    np_a = np.arange(9).reshape(3, 3)
    np_b = np_a[0]
    np_c = np_a[0, :]
    np_d = np_a[:, ::-1]
    ag_a = Tensor(np_a, requires_grad=True)
    ag_b = ag_a[0].numpy()
    ag_c = ag_a[0, :].numpy()
    ag_d = ag_a[:, ::-1].numpy()
    assert np.allclose(np_b, ag_b)
    assert np.allclose(np_c, ag_c)
    assert np.allclose(np_d, ag_d)


@pytest.mark.parametrize(
    "a_shape,b_shape",
    [
        ((10, 10, 2, 3), (10, 10, 3, 4)),
        ((1, 3), (3, 1)),
        ((2, 3), (3, 4)),
    ],
)
def test_matmul(a_shape: float, b_shape: float) -> None:
    """Test matrix multiplication."""
    a_size = np.prod(a_shape)
    b_size = np.prod(b_shape)
    np_a = np.random.uniform(size=(a_size,)).reshape(a_shape)
    np_b = np.random.uniform(size=(b_size,)).reshape(b_shape)
    ag_a = Tensor(np_a, requires_grad=True)
    ag_b = Tensor(np_b, requires_grad=True)
    np_c = np_a @ np_b
    ag_c = ag_a @ ag_b
    assert np.shape(np_c) == np.shape(ag_c.numpy())
    assert np.allclose(np_c, ag_c.numpy())


@pytest.mark.parametrize(
    "a_shape,b_shape",
    [
        ((2, 2), (2, 4)),
        ((2, 1), (1, 4)),
        ((1, 2, 3), (1, 3, 4)),
        ((1, 1, 2, 3), (1, 1, 3, 4)),
        ((2, 2), (2,)),
        ((1,), (1, 1, 2)),
    ],
)
def test_matmul_backprop(a_shape: tuple[int, ...], b_shape: tuple[int, ...]) -> None:
    """Test backpropagation against PyTorch's Tensor."""
    torch_a: torch.Tensor = torch.rand(a_shape, requires_grad=True)
    torch_b: torch.Tensor = torch.rand(b_shape, requires_grad=True)
    torch_c: torch.Tensor = torch_a @ torch_b
    torch_d: torch.Tensor = torch.mean(torch_c, dim=None)
    torch_d.backward()
    ag_a: Tensor = Tensor(torch_a.detach().numpy().copy(), requires_grad=True, name="a")
    ag_b: Tensor = Tensor(torch_b.detach().numpy().copy(), requires_grad=True, name="b")
    ag_c: Tensor = ag_a @ ag_b
    ag_c.name = "c"
    ag_d: Tensor = ag.mean(ag_c, axis=None)
    ag_d.name = "d"
    ag_d.backward()
    assert np.allclose(torch_d.detach().numpy(), ag_d.numpy()), f"{torch_d} != {ag_d}"
    assert np.allclose(
        torch_a.grad.numpy(), ag_a.grad.numpy()
    ), f"\ntorch:\n{torch_a.grad.numpy()}\n\n!=\n\nag:\n{ag_a.grad.numpy()}"


@pytest.mark.parametrize(
    "a_shape,b_shape",
    [
        ((2, 2), (2, 2)),
        ((2, 2), (2, 1)),
        ((2, 2), (1, 2)),
        ((2, 2), (1, 1)),
        ((2,), (2,)),
        ((1,), (1,)),
        ((1, 2, 2), (1, 1, 2)),
        ((2, 1), (1,)),
        ((2, 2), (2,)),
        ((2, 2, 2), (2,)),
    ],
)
def test_add_backprop(a_shape: tuple[int, ...], b_shape: tuple[int, ...]) -> None:
    """Test backpropagation against PyTorch's Tensor."""
    torch_a: torch.Tensor = torch.zeros(
        a_shape, dtype=torch.float64, requires_grad=True
    )
    torch_b: torch.Tensor = torch.arange(
        math.prod(b_shape), dtype=torch.float64, requires_grad=True
    ).reshape(b_shape)
    torch_c: torch.Tensor = torch_a + torch_b
    torch_d: torch.Tensor = torch.mean(torch_c, dim=None)
    torch_d.backward()
    ag_a: Tensor = Tensor(torch_a.detach().numpy().copy(), requires_grad=True, name="a")
    ag_b: Tensor = Tensor(torch_b.detach().numpy().copy(), requires_grad=True, name="b")
    ag_c: Tensor = ag_a + ag_b
    ag_c.name = "c"
    ag_d: Tensor = ag.mean(ag_c, axis=None)
    ag_d.name = "d"
    ag_d.backward()
    assert np.allclose(torch_c.detach().numpy(), ag_c.numpy()), f"{torch_c} != {ag_c}"
    assert np.allclose(torch_d.detach().numpy(), ag_d.numpy()), f"{torch_d} != {ag_d}"
    assert np.allclose(
        torch_a.grad.numpy(), ag_a.grad.numpy()
    ), f"\ntorch:\n{torch_a.grad.numpy()}\n\n!=\n\nag:\n{ag_a.grad.numpy()}"


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("input_size", [1, 2])
@pytest.mark.parametrize("layer_sizes", [[1], [2], [2, 3, 2]])
@pytest.mark.parametrize("output_size", [1, 2])
@pytest.mark.parametrize("activation_fn", ["sigmoid", "relu"])
@pytest.mark.parametrize("loss_fn", ["mse"])
def test_feedforward_network(
    batch_size: int,
    input_size: int,
    layer_sizes: list[int],
    output_size: int,
    activation_fn: str,
    loss_fn: str,
) -> None:
    """Test basic feedforward network."""

    def _forward_feedforward(weights, biases, activation_fn, x, y, loss_fn):
        activations = []
        n_layers: int = len(weights)
        for layer_i in range(n_layers):
            is_final_layer: bool = layer_i == n_layers - 1
            x = x @ weights[layer_i]
            if isinstance(x, ag.Tensor):
                x.name = f"weights_matmul_{layer_i}"
            activations.append(x)
            x = x + biases[layer_i]
            if isinstance(x, ag.Tensor):
                x.name = f"bias_add_{layer_i}"
            activations.append(x)
            if not is_final_layer:
                x = activation_fn(x)
                if isinstance(x, ag.Tensor):
                    x.name = f"activation_{layer_i}"
                activations.append(x)
        loss = loss_fn(y, x)
        if isinstance(x, ag.Tensor):
            loss.name = "loss"
        return loss, activations

    torch_activation_fns: dict[str, callable] = {
        "sigmoid": torch.sigmoid,
        "relu": torch.relu,
    }
    ag_activation_fns: dict[str, callable] = {
        "sigmoid": ag.sigmoid,
        "relu": ag.relu,
    }
    torch_loss_fns: dict[str, callable] = {
        "mse": torch.nn.functional.mse_loss,
    }
    ag_loss_fns: dict[str, callable] = {
        "mse": ag.loss.mse,
    }
    torch_weights: list[torch.Tensor] = []
    ag_weights: list[ag.Tensor] = []
    torch_biases: list[torch.Tensor] = []
    ag_biases: list[ag.Tensor] = []
    for layer_i, layer_size in enumerate(layer_sizes):
        is_first_layer: bool = layer_i == 0
        prev_layer_size: int = (
            input_size if is_first_layer else layer_sizes[layer_i - 1]
        )
        weights_shape: tuple[int, int] = (1, prev_layer_size, layer_size)
        bias_shape: tuple[int] = (layer_size,)
        torch_weights.append(
            torch.rand(weights_shape, requires_grad=True, dtype=torch.float64)
        )
        torch_biases.append(
            torch.rand(bias_shape, requires_grad=True, dtype=torch.float64)
        )
    final_weights_shape: tuple[int, int] = (1, layer_sizes[-1], output_size)
    final_bias_shape: tuple[int] = (output_size,)
    torch_weights.append(
        torch.rand(final_weights_shape, requires_grad=True, dtype=torch.float64)
    )
    torch_biases.append(
        torch.rand(final_bias_shape, requires_grad=True, dtype=torch.float64)
    )
    for layer_i, (torch_w, torch_b) in enumerate(zip(torch_weights, torch_biases)):
        torch_w.retain_grad()
        torch_b.retain_grad()
        ag_weights.append(
            Tensor(
                torch_w.detach().numpy().copy(),
                requires_grad=True,
                name=f"w_{layer_i}",
            )
        )
        ag_biases.append(
            Tensor(
                torch_b.detach().numpy().copy(),
                requires_grad=True,
                name=f"b_{layer_i}",
            )
        )
    torch_x: torch.Tensor = torch.rand((batch_size, input_size), dtype=torch.float64)
    torch_y: torch.Tensor = torch.rand((batch_size, output_size), dtype=torch.float64)
    ag_x: ag.Tensor = ag.Tensor(torch_x.detach().numpy().copy())
    ag_y: ag.Tensor = ag.Tensor(torch_y.detach().numpy().copy())
    torch_loss, torch_activations = _forward_feedforward(
        weights=torch_weights,
        biases=torch_biases,
        activation_fn=torch_activation_fns[activation_fn],
        loss_fn=torch_loss_fns[loss_fn],
        x=torch_x,
        y=torch_y,
    )
    torch_loss.backward()
    ag_loss, ag_activations = _forward_feedforward(
        weights=ag_weights,
        biases=ag_biases,
        activation_fn=ag_activation_fns[activation_fn],
        loss_fn=ag_loss_fns[loss_fn],
        x=ag_x,
        y=ag_y,
    )
    ag_loss.backward()
    for torch_a, ag_a in zip(torch_activations, ag_activations):
        try:
            assert np.allclose(torch_a.detach().numpy(), ag_a.numpy())
        except:
            breakpoint()
    try:
        assert np.isclose(torch_loss.detach().numpy(), ag_loss.numpy())
    except:
        print(torch_loss.detach().numpy(), "!=", ag_loss.numpy())
        breakpoint()
    for torch_w, torch_b, ag_w, ag_b in zip(
        torch_weights, torch_biases, ag_weights, ag_biases
    ):
        np.allclose(torch_w.grad.numpy(), ag_w.grad.numpy())
        np.allclose(torch_b.grad.numpy(), ag_b.grad.numpy())

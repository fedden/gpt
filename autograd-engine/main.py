"""A basic autograd library for learning purposes."""
from __future__ import annotations
from typing import Any

import numpy as np
import torch

import ag
from ag.tensor import Tensor


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


def test_matmul_backprop(a_shape: tuple[int, ...], b_shape: tuple[int, ...]) -> None:
    """Test backpropagation against PyTorch's Tensor."""
    torch_a: torch.Tensor = torch.rand(a_shape, requires_grad=True)
    torch_b: torch.Tensor = torch.rand(b_shape, requires_grad=True)
    torch_c: torch.Tensor = torch_a @ torch_b
    torch_d: torch.Tensor = torch.mean(torch_c, dim=None)
    torch_d.backward()
    ag_a: Tensor = Tensor(torch_a.detach().numpy().copy(), requires_grad=True)
    ag_b: Tensor = Tensor(torch_b.detach().numpy().copy(), requires_grad=True)
    ag_c: Tensor = ag_a @ ag_b
    ag_d: Tensor = ag.mean(ag_c, axis=None)
    ag_d.backward()
    assert np.allclose(torch_d.detach().numpy(), ag_d.numpy()), f"{torch_d} != {ag_d}"
    assert np.allclose(
        torch_a.grad.numpy(), ag_a.grad.numpy()
    ), f"\ntorch:\n{torch_a.grad.numpy()}\n\n!=\n\nag:\n{ag_a.grad.numpy()}"


def test_add_backprop(shape: tuple[int, ...]) -> None:
    """Test backpropagation against PyTorch's Tensor."""
    torch_a: torch.Tensor = torch.rand(shape, requires_grad=True)
    torch_b: torch.Tensor = torch.rand(shape, requires_grad=True)
    torch_c: torch.Tensor = torch_a + torch_b
    torch_d: torch.Tensor = torch.mean(torch_c, dim=None)
    torch_d.backward()
    ag_a: Tensor = Tensor(torch_a.detach().numpy().copy(), requires_grad=True)
    ag_b: Tensor = Tensor(torch_b.detach().numpy().copy(), requires_grad=True)
    ag_c: Tensor = ag_a + ag_b
    ag_d: Tensor = ag.mean(ag_c, axis=None)
    breakpoint()
    ag_d.backward()
    assert np.allclose(torch_d.detach().numpy(), ag_d.numpy()), f"{torch_d} != {ag_d}"
    assert np.allclose(
        torch_a.grad.numpy(), ag_a.grad.numpy()
    ), f"\ntorch:\n{torch_a.grad.numpy()}\n\n!=\n\nag:\n{ag_a.grad.numpy()}"



if __name__ == "__main__":
    test_add_backprop((2, 2))
    test_matmul_backprop((1, 1, 2, 3), (1, 1, 3, 4))
    test_basic_elementwise_ops_and_broadcasting()
    test_slicing()
    test_matmul((10, 10, 2, 3), (10, 10, 3, 4))
    test_matmul((1, 3), (3, 1))
    test_matmul((2, 3), (3, 4))
    print("All tests passed!")

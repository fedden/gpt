"""Test ops and graph."""
import traceback

import numpy as np
import pytest
import torch

import ag
from ag import Parameter, Scalar
from ag.ascii import render_as_tree


@pytest.mark.parametrize(
    "a", [-1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, np.pi, np.e, np.sqrt(2)]
)
@pytest.mark.parametrize(
    "b", [-1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, np.pi, np.e, np.sqrt(2)]
)
@pytest.mark.parametrize("verbose", [False])
def test_forward_backward_against_torch_diamond(
    a: float, b: float, verbose: bool
) -> None:
    """Test forward and backward pass against torch, with one node being used twice."""
    torch_a = torch.nn.Parameter(torch.tensor(a))
    torch_a.retain_grad()
    torch_b = torch.nn.Parameter(torch.tensor(b))
    torch_b.retain_grad()
    torch_c = torch_b + torch_a
    torch_c.retain_grad()
    torch_d = torch_a * torch_c
    torch_d.retain_grad()
    torch_d.backward()
    if verbose:
        print("----------------------------------------")
        print(f"a: {a}, b: {b}")
        print("----------------------------------------")
        print("TORCH")
        for t in [torch_a, torch_b, torch_c, torch_d]:
            grad = 0 if t.grad is None else t.grad.tolist()
            print(f"value: {t.data.tolist():12.8f}, grad: {grad:12.8f}")
    ag_a = Parameter(a, name="a")
    ag_b = Parameter(b, name="b")
    ag_c = ag_b + ag_a
    ag_c.name = "c"
    ag_d = ag_a * ag_c
    ag_d.name = "d"
    try:
        ag_d.backward()
    except Exception:
        render_as_tree(ag_d)
        # Print the stack trace nicely so we can see where the error
        # occurred, and then insert a breakpoint so we can inspect the
        # state of the program.
        traceback.print_exc()
        breakpoint()
        ag_d.backward()
    if verbose:
        print("AG")
        for s in [ag_a, ag_b, ag_c, ag_d]:
            print(f"value: {s.data:12.8f}, grad: {s.grad:12.8f}, name: {s.name}")
    assert np.isclose(
        ag_a.grad.numpy(), torch_a.grad.numpy()
    ), f"ag_a.grad ({ag_a.grad}) != torch_a.grad ({torch_a.grad.tolist()})"
    assert np.isclose(
        ag_d.numpy(), torch_d.detach().numpy(),
    ), f"ag_c.data ({ag_d.numpy()}) != torch_c.data ({torch_d.detach().numpy()})"


@pytest.mark.parametrize(
    "a", [-1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, np.pi, np.e, np.sqrt(2)]
)
@pytest.mark.parametrize(
    "b", [-1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, np.pi, np.e, np.sqrt(2)]
)
def test_forward_backward_against_torch(a: float, b: float) -> None:
    """Test forward and backward pass against torch."""
    torch_a = torch.nn.Parameter(torch.tensor(a))
    torch_b = torch.tensor(b)
    torch_c = torch_a * torch_b
    torch_d = torch_c + torch_b
    torch_e = torch.sigmoid(torch_d)
    torch_f = torch.tanh(torch_d)
    torch_g = torch.exp(torch_d)
    torch_h = torch_e + torch_f + torch_g

    torch_c.retain_grad()
    torch_d.retain_grad()
    torch_e.retain_grad()
    torch_f.retain_grad()
    torch_g.retain_grad()
    torch_h.retain_grad()
    torch_h.backward()
    print("TORCH")
    for t in [torch_a, torch_b, torch_c, torch_d, torch_e, torch_f, torch_g, torch_h]:
        grad = 0 if t.grad is None else t.grad.tolist()
        print(f"value: {t.data.tolist():12.8f}, grad: {grad:12.8f}")

    ag_a = Parameter(a, name="a")
    ag_b = Scalar(b, name="b")
    ag_c = ag_a * ag_b
    ag_c.name = "c"
    ag_d = ag_c + ag_b
    ag_d.name = "d"
    ag_e = ag.sigmoid(ag_d)
    ag_e.name = "e"
    ag_f = ag.tanh(ag_d)
    ag_f.name = "f"
    ag_g = ag.exp(ag_d)
    ag_g.name = "g"
    ag_h = ag_e + ag_f + ag_g
    ag_h.name = "h"
    try:
        ag_h.backward()
    except Exception:
        render_as_tree(ag_h)
        # Print the stack trace nicely so we can see where the error
        # occurred, and then insert a breakpoint so we can inspect the
        # state of the program.
        traceback.print_exc()
        ag_h.backward()
    print("AG")
    for s in [ag_a, ag_b, ag_c, ag_d, ag_e, ag_f, ag_g, ag_h]:
        print(f"value: {s.numpy():12.8f}, grad: {s.grad.numpy():12.8f}, name: {s.name}")
    assert np.isclose(ag_a.grad.numpy(), torch_a.grad.numpy())
    assert np.isclose(ag_h.numpy(), torch_h.detach().numpy())


@pytest.mark.parametrize("a", [-1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0])
@pytest.mark.parametrize("op", ["tanh", "sigmoid", "exp"])
@pytest.mark.parametrize("verbose", [False])
def test_forward_backward_against_torch_simple(
    a: float, op: str, verbose: bool,
) -> None:
    """Test forward and backward pass against torch."""
    torch_a = torch.nn.Parameter(torch.tensor(a))
    torch_a.retain_grad()
    torch_b = getattr(torch, op)(torch_a)
    torch_b.retain_grad()
    torch_c = torch_b + 1
    torch_c.retain_grad()
    torch_c.backward()
    if verbose:
        print("----------------------------------------")
        print(f"a: {a}, op: {op}")
        print("----------------------------------------")
        print("TORCH")
        for t in [torch_a, torch_b, torch_c]:
            grad = 0 if t.grad is None else t.grad.tolist()
            print(f"value: {t.data.tolist():12.8f}, grad: {grad:12.8f}")

    ag_a = Parameter(a, name="a")
    ag_b = getattr(ag, op)(ag_a)
    ag_b.name = "b"
    ag_c = ag_b + 1
    ag_c.name = "c"
    try:
        ag_c.backward()
    except Exception:
        render_as_tree(ag_c)
        # Print the stack trace nicely so we can see where the error
        # occurred, and then insert a breakpoint so we can inspect the
        # state of the program.
        traceback.print_exc()
        ag_c.backward()
    if verbose:
        print("AG")
        for t in [ag_a, ag_b, ag_c]:
            print(f"value: {t.data:12.8f}, grad: {t.grad:12.8f}, name: {t.name}")
    assert np.isclose(
        ag_a.grad.numpy(), torch_a.grad.numpy()
    ), f"For {op}, ag_a.grad ({ag_a.grad}) != torch_a.grad ({torch_a.grad.tolist()})"
    assert np.isclose(
        ag_c.numpy(), torch_c.detach().numpy()
    ), f"For {op}, ag_c ({ag_c.numpy()}) != torch_c ({torch_c.detach().numpy()})"

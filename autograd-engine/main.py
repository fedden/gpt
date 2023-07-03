"""A basic autograd library for learning purposes."""
from __future__ import annotations
from typing import List

import ag
from ag import Parameter, Scalar
from ag.loss import mse, binary_cross_entropy
from ag.optimiser import SGDOptimiser


def regression_test_forward_pass_perceptron() -> None:
    import numpy as np
    from sklearn.linear_model import Perceptron

    np.random.seed(42)
    X = np.random.uniform(size=(5, 2))
    y = np.random.randint(0, 2, size=(5,))
    clf = Perceptron(tol=1e-3, random_state=0)
    clf.fit(X, y)
    target_logit = clf.decision_function(X)
    w0, w1 = clf.coef_[0, :].tolist()
    w0 = Parameter(w0)
    w1 = Parameter(w1)
    b = Parameter(clf.intercept_[0])

    for i in range(len(X)):
        x0 = Scalar(X[i, 0])
        x1 = Scalar(X[i, 1])
        logit = x0 * w0 + x1 * w1 + b
        assert np.isclose(logit.data, target_logit[i])


def regression_test_train_linear_regression(
    max_n_epochs: int = 10000000,
    cost_delta_tol: float = 1e-14,
    lr: float = 0.01,
    verbose: bool = False,
) -> None:
    import numpy as np
    from sklearn.linear_model import LinearRegression

    np.random.seed(42)
    # Generate some data.
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    # Compare with sklearn - we should get the same coefficients, even though
    # they will use the analytical solution and we will use gradient descent.
    clf = LinearRegression(
        fit_intercept=True,
        copy_X=True,
        n_jobs=None,
    )
    clf.fit(X, y)
    # The coefficients we are trying to learn.
    w0 = Parameter(0.0)
    w1 = Parameter(0.0)
    b = Parameter(0.0)
    optimiser = SGDOptimiser([w0, w1, b], lr=lr)
    last_cost: float = 0.0
    for _ in range(max_n_epochs):
        loss_sum: float = 0.0
        for i in range(len(X)):
            x0 = Scalar(X[i, 0])
            x1 = Scalar(X[i, 1])
            y_hat = x0 * w0 + x1 * w1 + b
            loss = mse(Scalar(y[i]), y_hat) / len(X)
            loss.backward()
            loss_sum += loss.data
        # Cost is the average loss over all samples.
        cost: float = loss_sum / len(X)
        if verbose:
            print(
                f"cost={cost:.15f}, is {abs(cost - last_cost):.15f} < "
                f"{cost_delta_tol:.15f}?"
            )
        if abs(cost - last_cost) < cost_delta_tol:
            break
        optimiser.step()
        optimiser.zero_grad()
        last_cost = cost
    our_params: List[float] = [w0.data, w1.data, b.data]
    sklearn_params: List[float] = [*clf.coef_.tolist(), clf.intercept_.tolist()]
    if verbose:
        print(f"our_params: {our_params}")
        print(f"sklearn_params: {sklearn_params}")
    assert np.allclose(our_params, sklearn_params)


def regression_test_forward_pass_logistic_regression(
    max_n_epochs: int = 10000000,
) -> None:
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    np.random.seed(42)
    # Generate some data.
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.array([0, 0, 1, 1])
    clf = LogisticRegression(
        penalty=None,
        tol=1e-4,
        fit_intercept=True,
        random_state=0,
        solver="lbfgs",
        max_iter=max_n_epochs,
    )
    clf.fit(X, y)
    target_logits: np.ndarray = clf.decision_function(X)
    # Eqivalent to scipy.special.expit(target_logits)
    target_probs: np.ndarray = clf.predict_proba(X)[:, 1]
    # Copy the coefficients that have been learned.
    w0 = Parameter(clf.coef_[0, 0])
    w1 = Parameter(clf.coef_[0, 1])
    b = Parameter(clf.intercept_[0])
    # Check that the forward pass gives the same result as sklearn.
    for i in range(len(X)):
        x0 = Scalar(X[i, 0])
        x1 = Scalar(X[i, 1])
        logit = x0 * w0 + x1 * w1 + b
        assert ag.isclose(
            logit, target_logits[i]
        ), f"i={i}, logit={logit.data}, target_logit={target_logits[i]}"
        prob = ag.sigmoid(logit)
        assert ag.isclose(
            prob, target_probs[i]
        ), f"i={i}, prob={prob.data}, target_prob={target_probs[i]}"


def regression_test_train_logistic_regression(
    max_n_epochs: int = 10000000,
    cost_delta_tol: float = 1e-13,
    lr: float = 2.0,
    verbose: bool = True,
) -> None:
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    np.random.seed(42)
    # Generate some data.
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.array([0, 0, 1, 1])
    clf = LogisticRegression(
        penalty=None,
        tol=1e-4,
        fit_intercept=True,
        random_state=0,
        solver="lbfgs",
        max_iter=max_n_epochs,
    )
    clf.fit(X, y)
    # The coefficients we are trying to learn.
    w0 = Parameter(0.0, name="w0")
    w1 = Parameter(0.0, name="w1")
    b = Parameter(0.0, name="b")
    optimiser = SGDOptimiser([w0, w1, b], lr=lr)
    last_cost: float = 0.0
    for _ in range(max_n_epochs):
        loss_sum: float = 0.0
        for i in range(len(X)):
            x0 = Scalar(X[i, 0], name="x0")
            x1 = Scalar(X[i, 1], name="x1")
            activation = x0 * w0 + x1 * w1 + b
            activation.name = "activation"
            prob = ag.sigmoid(activation)
            prob.name = "prob"
            loss = binary_cross_entropy(y_true=Scalar(y[i], name="y"), y_pred=prob)
            loss.name = "loss"
            optimiser.zero_grad()
            try:
                loss.backward()
            except Exception:
                from ag.ascii import render_as_tree

                render_as_tree(loss)
                # Print the stack trace nicely so we can see where the error
                # occurred, and then insert a breakpoint so we can inspect the
                # state of the program.
                import traceback

                traceback.print_exc()
                breakpoint()
                loss.backward()
            optimiser.step()
            loss_sum += loss.data
        # Cost is the average loss over all samples.
        cost: float = loss_sum / len(X)
        if verbose:
            print(
                f"cost={cost:.15f}, is {abs(cost - last_cost):.15f} < "
                f"{cost_delta_tol:.15f}?"
            )
        if abs(cost - last_cost) < cost_delta_tol:
            break
        last_cost = cost
    our_params: List[float] = [w0.data, w1.data, b.data]
    sklearn_params: List[float] = [*clf.coef_[0].tolist(), clf.intercept_[0].tolist()]
    if verbose:
        print(f"our_params: {our_params}")
        print(f"sklearn_params: {sklearn_params}")
    assert np.allclose(our_params, sklearn_params)


def regression_test_forward_backward_against_torch_diamond(
    a: float, b: float, verbose: bool = True
) -> None:
    import torch
    import numpy as np

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
        from ag.ascii import render_as_tree

        render_as_tree(ag_d)
        # Print the stack trace nicely so we can see where the error
        # occurred, and then insert a breakpoint so we can inspect the
        # state of the program.
        import traceback

        traceback.print_exc()
        breakpoint()
        ag_d.backward()
    if verbose:
        print("AG")
        for t in [ag_a, ag_b, ag_c, ag_d]:
            print(f"value: {t.data:12.8f}, grad: {t.grad:12.8f}, name: {t.name}")
    assert np.isclose(
        ag_a.grad, torch_a.grad.tolist()
    ), f"ag_a.grad ({ag_a.grad}) != torch_a.grad ({torch_a.grad.tolist()})"
    assert np.isclose(
        ag_d.data, torch_d.data.tolist()
    ), f"ag_c.data ({ag_d.data}) != torch_c.data ({torch_d.data.tolist()})"


def regression_test_forward_backward_against_torch(a: float, b: float) -> None:
    import torch
    import numpy as np

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
        from ag.ascii import render_as_tree

        render_as_tree(ag_h)
        # Print the stack trace nicely so we can see where the error
        # occurred, and then insert a breakpoint so we can inspect the
        # state of the program.
        import traceback

        traceback.print_exc()
        ag_h.backward()
    print("AG")
    for t in [ag_a, ag_b, ag_c, ag_d, ag_e, ag_f, ag_g, ag_h]:
        print(f"value: {t.data:12.8f}, grad: {t.grad:12.8f}, name: {t.name}")
    assert np.isclose(ag_a.grad, torch_a.grad.tolist())
    assert np.isclose(ag_h.data, torch_h.data.tolist())


def regression_test_forward_backward_against_torch_simple(
    a: float, op: str, verbose: bool = True
) -> None:
    import torch
    import numpy as np

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
        from ag.ascii import render_as_tree

        render_as_tree(ag_c)
        # Print the stack trace nicely so we can see where the error
        # occurred, and then insert a breakpoint so we can inspect the
        # state of the program.
        import traceback

        traceback.print_exc()
        breakpoint()
        ag_c.backward()
    if verbose:
        print("AG")
        for t in [ag_a, ag_b, ag_c]:
            print(f"value: {t.data:12.8f}, grad: {t.grad:12.8f}, name: {t.name}")
    assert np.isclose(
        ag_a.grad, torch_a.grad.tolist()
    ), f"For {op}, ag_a.grad ({ag_a.grad}) != torch_a.grad ({torch_a.grad.tolist()})"
    assert np.isclose(
        ag_c.data, torch_c.data.tolist()
    ), f"For {op}, ag_c.data ({ag_c.data}) != torch_c.data ({torch_c.data.tolist()})"


if __name__ == "__main__":
    regression_test_forward_backward_against_torch_diamond(2.0, 3.0)
    regression_test_forward_backward_against_torch_simple(1.0, "sigmoid")
    regression_test_forward_backward_against_torch_simple(1.0, "exp")
    regression_test_forward_backward_against_torch_simple(2.0, "exp")
    regression_test_forward_backward_against_torch_simple(-2.0, "exp")
    regression_test_forward_backward_against_torch_simple(1.0, "tanh")
    regression_test_forward_backward_against_torch_simple(2.0, "tanh")
    regression_test_forward_backward_against_torch_simple(-2.0, "tanh")
    regression_test_forward_backward_against_torch(1.0, 2.0)
    regression_test_forward_backward_against_torch(-1.0, 2.0)
    regression_test_forward_backward_against_torch(1.0, -2.0)
    regression_test_forward_backward_against_torch(-1.0, -2.0)
    regression_test_forward_pass_perceptron()
    regression_test_forward_pass_logistic_regression()
    regression_test_train_logistic_regression()
    regression_test_train_linear_regression()

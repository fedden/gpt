"""Test we can train/replicate basic linear models."""
from typing import Any, List

import numpy as np
from sklearn.linear_model import Perceptron, LinearRegression, LogisticRegression

import ag
from ag import Parameter, Scalar
from ag.loss import mse, binary_cross_entropy
from ag.optimiser import SGDOptimiser


def test_forward_pass_perceptron() -> None:
    """Test forward pass of a perceptron."""
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


def test_train_linear_regression(
    max_n_epochs: int = 10000000,
    cost_delta_tol: float = 1e-14,
    lr: float = 0.01,
    verbose: bool = False,
) -> None:
    """Test we can train a linear regression model."""
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
    w0 = ag.Tensor(0.0, requires_grad=True)
    w1 = ag.Tensor(0.0, requires_grad=True)
    b = ag.Tensor(0.0, requires_grad=True)
    optimiser = SGDOptimiser([w0, w1, b], lr=lr)
    last_cost: float = 0.0
    for _ in range(max_n_epochs):
        optimiser.zero_grad()
        loss_sum: float = 0.0
        for i in range(len(X)):
            x0 = ag.Tensor(X[i, 0])
            x1 = ag.Tensor(X[i, 1])
            y_hat = x0 * w0 + x1 * w1 + b
            loss = mse(ag.Tensor(y[i]), y_hat) / len(X)
            try:
                loss.backward()
            except Exception:
                from ag.ascii import render_as_tree

                render_as_tree(loss)
                breakpoint()
                loss.backward()
            loss_sum += loss.numpy()
        # Cost is the average loss over all samples.
        cost: float = loss_sum / len(X)
        breakpoint()
        if verbose:
            print(
                f"cost={cost:.15f}, is {abs(cost - last_cost):.15f} < "
                f"{cost_delta_tol:.15f}?"
            )
        if abs(cost - last_cost) < cost_delta_tol:
            break
        optimiser.step()
        last_cost = cost
    our_params: List[float] = [w0.numpy(), w1.numpy(), b.numpy()]
    sklearn_params: List[float] = [*clf.coef_.tolist(), clf.intercept_.tolist()]
    if verbose:
        print(f"our_params: {our_params}")
        print(f"sklearn_params: {sklearn_params}")
    assert np.allclose(our_params, sklearn_params, atol=0.001)


def test_forward_pass_logistic_regression(
    max_n_epochs: int = 10000000,
) -> None:
    """Test forward pass of a logistic regression model."""
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


def test_train_logistic_regression(
    max_n_epochs: int = 10000000,
    cost_delta_tol: float = 1e-13,
    lr: float = 0.1,
    verbose: bool = True,
) -> None:
    """Test we can train a logistic regression model."""

    def forward(x0: Any, x1: Any, w0: Parameter, w1: Parameter, b: Parameter) -> Any:
        x0 = Scalar(x0, name="x0")
        x1 = Scalar(x1, name="x1")
        logit = x0 * w0 + x1 * w1 + b
        logit.name = "logit"
        prob = ag.sigmoid(logit)
        prob.name = "prob"
        return prob

    np.random.seed(42)
    # Generate some data.
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.array([0, 0, 1, 1])
    # The coefficients we are trying to learn.
    w0 = Parameter(0.0, name="w0")
    w1 = Parameter(0.0, name="w1")
    b = Parameter(0.0, name="b")
    optimiser = SGDOptimiser([w0, w1, b], lr=lr, momentum=0.01)
    last_cost: float = 0.0
    for _ in range(max_n_epochs):
        loss_sum: float = 0.0
        for i in range(len(X)):
            prob = forward(x0=X[i, 0], x1=X[i, 1], w0=w0, w1=w1, b=b)
            loss = binary_cross_entropy(y_true=Scalar(y[i], name="y"), y_pred=prob)
            loss.name = "loss"
            optimiser.zero_grad()
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
        last_cost = cost
        all_close: bool = True
        for i in range(len(X)):
            prediction = forward(x0=X[i, 0], x1=X[i, 1], w0=w0, w1=w1, b=b) > 0.5
            all_close = all_close and ag.isclose(prediction, Scalar(y[i], name="y"))
        if all_close:
            break
    our_params: List[float] = [w0.data, w1.data, b.data]
    if verbose:
        print(f"our_params: {our_params}")
        for i in range(len(X)):
            prob = forward(x0=X[i, 0], x1=X[i, 1], w0=w0, w1=w1, b=b)
            prediction = prob > 0.5
            print(f"prob={prob.data}, prediction={prediction.data}, y={y[i]}")

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
        assert ag.isclose(logit, target_logits[i]), (
            f"i={i}, logit={logit.data}, target_logit={target_logits[i]}"
        )
        prob = ag.sigmoid(logit)
        assert ag.isclose(prob, target_probs[i]), (
            f"i={i}, prob={prob.data}, target_prob={target_probs[i]}"
        )


def regression_test_train_logistic_regression(
    max_n_epochs: int = 10000000,
    cost_delta_tol: float = 1e-14,
    lr: float = 0.01,
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
            prob = ag.sigmoid(x0 * w0 + x1 * w1 + b)
            loss = binary_cross_entropy(Scalar(y[i]), prob) / len(X)
            loss.backward()
            loss_sum += loss.data
        # Cost is the average loss over all samples.
        cost: float = loss_sum / len(X)
        if verbose:
            print(
                f"cost={cost:.15f}, is {abs(cost - last_cost):.15f} < "
                f"{cost_delta_tol:.15f}?"
            )
            breakpoint()
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
    breakpoint()
    assert np.allclose(our_params, sklearn_params)


if __name__ == "__main__":
    regression_test_forward_pass_perceptron()
    regression_test_forward_pass_logistic_regression()
    regression_test_train_logistic_regression()
    regression_test_train_linear_regression()

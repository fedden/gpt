import ag

from sklearn.datasets import make_regression


def main():
    seed: int = 42
    n_features: int = 2
    n_epochs: int = 100
    lr: float = 0.01
    momentum: float = 0.0
    ag.random.seed(seed)
    np_x, np_y = make_regression(
        n_samples=100,
        n_features=n_features,
        noise=0,
        random_state=seed,
    )
    ag_x: ag.Tensor = ag.Tensor(np_x)
    ag_y: ag.Tensor = ag.Tensor(np_y)
    ag_x[:, 0]
    ag_network: ag.nn.Module = ag.nn.Sequential(
        ag.nn.Linear(n_features, 5),
        ag.nn.Tanh(),
        ag.nn.Linear(5, 5),
        ag.nn.Tanh(),
        ag.nn.Linear(5, 1),
    )
    ag_optimiser: ag.optimiser.Optimiser = ag.optimiser.SGDOptimiser(
        params=ag_network.parameter_list(),
        lr=lr,
        momentum=momentum,
    )
    for _ in range(n_epochs):
        indices = ag.random.permutation(len(ag_x))
        breakpoint()
        ag_y_hat: ag.Tensor = ag_network(ag_x[indices])
        ag_loss: ag.Tensor = ag.loss.mse(ag_y[indices], ag_y_hat)
        ag_optimiser.zero_grad()
        ag_loss.backward()
        ag_optimiser.step()
        print(ag_loss.numpy())



if __name__ == "__main__":
    main()

import ag

from sklearn.datasets import make_regression


def main():
    seed: int = 42
    n_features: int = 2
    n_epochs: int = 100
    lr: float = 0.01
    momentum: float = 0.0
    batch_size: int = 10
    ag.random.seed(seed)
    np_x, np_y = make_regression(
        n_samples=100,
        n_features=n_features,
        noise=0,
        random_state=seed,
    )
    ag_x: ag.Tensor = ag.Tensor(np_x)
    ag_y: ag.Tensor = ag.Tensor(np_y)
    ag_network: ag.nn.Module = ag.nn.Sequential(
        ag.nn.Linear(n_features, 4),
        ag.nn.Tanh(),
        ag.nn.Linear(4, 1),
    )
    ag_optimiser: ag.optimiser.Optimiser = ag.optimiser.SGDOptimiser(
        params=ag_network.parameter_list(),
        lr=lr,
        momentum=momentum,
    )
    for _ in range(n_epochs):
        indices = ag.random.permutation(len(ag_x))
        loss: float = 0.0
        n_batches: int = len(ag_x) // batch_size
        for batch_start in range(0, len(ag_x), batch_size):
            batch_indices = indices[batch_start:batch_start + batch_size]
            ag_x_batch: ag.Tensor = ag_x[batch_indices]
            ag_y_batch: ag.Tensor = ag_y[batch_indices]
            ag_y_hat: ag.Tensor = ag_network(ag_x_batch)
            ag_loss: ag.Tensor = ag.loss.mse(ag_y_batch, ag_y_hat)
            ag_optimiser.zero_grad()
            ag_loss.backward()
            ag_optimiser.step()
            loss += ag_loss.numpy() / n_batches
        print(f"loss: {loss:.4f}")



if __name__ == "__main__":
    main()

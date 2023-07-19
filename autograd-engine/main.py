import ag

from sklearn.datasets import make_regression


def main():
    seed: int = 42
    n_features: int = 2
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
        ag.nn.Linear(n_features, 5),
        ag.nn.Tanh(),
        ag.nn.Linear(5, 5),
        ag.nn.Tanh(),
        ag.nn.Linear(5, 1),
    )
    breakpoint()
    ag_network(ag_x[:10, :]).numpy()




if __name__ == "__main__":
    main()

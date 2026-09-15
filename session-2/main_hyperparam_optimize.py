import torch
from ray import tune

from dataset import MyDataset
from model import MyModel
from utils import accuracy

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def train_single_epoch(...):
    # TODO: Implement training loop
    raise NotImplementedError


def eval_single_epoch(...):
    # TODO: Implement evaluation loop
    raise NotImplementedError


def train_model(config):
    # Ray will populate the "config" dictionary automatically,
    # according to the defined search space.

    my_dataset = MyDataset(...)
    my_model = MyModel(...).to(device)

    for epoch in range(config["epochs"]):
        train_single_epoch(...)
        eval_single_epoch(...)
    
    return {
        "val_loss": val_loss
    }


if __name__ == "__main__":

    analysis = tune.run(
        train_model,
        metric="val_loss",
        mode="min",
        num_samples=5,
        resources_per_trial={"cpu": 1, "gpu": 1 if torch.cuda.is_available() else 0},
        config={
            "hyperparam_1": tune.uniform(1, 10),
            "hyperparam_2": tune.grid_search(["relu", "tanh"]),
        })

    print("Best hyperparameters found were: ", analysis.best_config)

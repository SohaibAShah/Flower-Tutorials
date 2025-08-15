"""my-first-app: A Flower / PyTorch app."""

from operator import ge
from typing import List, Tuple
from flwr.common import Context, ndarrays_to_parameters, Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from datasets import load_dataset
from torch.utils.data import DataLoader

import torch
from my_first_app.task import Net, get_weights, set_weights, test, train, get_transforms

def get_evaluate_fn(testloader, device):
    """Return a callback that evaluates the global model"""

    def evaluate(server_round, parameters_ndarrays, config):
        """Evaluate global model using provided centralized test set"""
        net = Net()
        set_weights(net, parameters_ndarrays)
        net.to(device)
        loss, accuracy = test(net, testloader, device)
        return loss, {"cen_accuracy": accuracy}

    return evaluate


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """A function that aggregates metrics"""
    # Initialize aggregated metrics
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    total_examples = sum(num_examples for num_examples, _ in metrics)

    return {
        "accuracy": sum(accuracies) / total_examples if total_examples > 0 else 0.0,
    }

def on_fit_config(server_round: int) -> Metrics:
    """Adjust learning rate based on current round"""

    lr = 0.01
    if server_round > 2:
        lr = 0.05
    return {"lr": lr}


def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    # Initialize model parameters
    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)

    #Load global test set
    testset = load_dataset("zalando-datasets/fashion_mnist")["test"]

    testloader = DataLoader(testset.with_transform(get_transforms()), batch_size=32)

    # Define strategy
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
        evaluate_metrics_aggregation_fn=weighted_average,
        on_fit_config_fn=on_fit_config,
        evaluate_fn=get_evaluate_fn(
            testloader,
            device="cuda:0"
        ),
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)

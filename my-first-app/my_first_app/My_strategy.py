


from logging import config
from flwr.common import Metrics, FitRes, Parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from typing import List, Tuple
from my_first_app.task import Net, set_weights
import torch, json
import wandb, os
from datetime import datetime


class CustomFedAvg(FedAvg):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.results_to_save = {}

        # Start a new wandb run to track this script.
        name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        wandb.init(
            # Set the wandb entity where your project will be logged (generally your team name).
            #entity="my-awesome-team-name",
            # Set the wandb project where this run will be logged.
            project="flower-simulation-tutorial",

            name = f"custom-strategy-{name}",
        )

    def aggregate_fit(self, 
                      server_round: int,
                      results: list[tuple[ClientProxy, FitRes]],
                      failures: list[tuple[ClientProxy, FitRes] | BaseException]
                      ) -> tuple[Parameters | None, dict[str, bool | bytes | float | int, str]]:
    # Custom aggregation logic
        parameters_aggredated, metrices_aggregated = super().aggregate_fit(server_round, results, failures)

        # Custom aggregation logic
        if parameters_aggredated is not None:
            # Convert parameters to ndarrays
            ndarrays = parameters_to_ndarrays(parameters_aggredated)

        # instantiate model
        model = Net()
        set_weights(model, ndarrays)
        # Save global model in the standard PyTorch way
        # Ensure global_model directory exists
        os.makedirs("global_model", exist_ok=True)
        
        # Save global model in the global_model folder
        model_path = os.path.join("global_model", f"global_model_round_{server_round}.pth")
        torch.save(model.state_dict(), model_path)

        return parameters_aggredated, metrices_aggregated
    

    def evaluate(self, 
                 server_round: int, 
                 parameters: Parameters
                 ) -> tuple[float, dict[str, bool | bytes | float | int, str]] | None:
        # Custom evaluation logic
        loss, metrics = super().evaluate(server_round, parameters)

        my_results = {"loss": loss, **metrics}


        self.results_to_save[server_round] = my_results

        # Save metrics as json
        with open("results.json", "w") as f:
            json.dump(self.results_to_save, f, indent=4)

        #Log to w&b
        wandb.log(my_results, step=server_round)

        return loss, metrics
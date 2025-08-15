


from logging import config
from flwr.common import Metrics, FitRes, Parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from typing import List, Tuple
from my_first_app.task import Net, set_weights
import torch, json


class CustomFedAvg(FedAvg):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.results_to_save = {}

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
        torch.save(model.state_dict(), f"global_model_round_{server_round}.pth")

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

        return loss, metrics
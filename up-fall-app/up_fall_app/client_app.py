"""up-fall-app: A Flower / PyTorch app."""

from typing_extensions import OrderedDict
import torch

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from up_fall_app.task import SensorModel, get_weights, load_data, set_weights, test, train
from load_data import loadData, splitForClients


# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(self, cid, net, trainloader, valloader):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        # NEW: Print when a client starts training
        print(f"✅ [Client {self.cid}] Starting training for round {config['server_round']}...")

        self.set_parameters(parameters)
        
        # Train and get the list of losses
        losses = train(self.net, self.trainloader, epochs=3)
        
        # Also get the local validation accuracy
        _, val_acc = test(self.net, self.valloader)

        # NEW: Announce completion of training
        print(f"✅ [Client {self.cid}] Finished training.")
        
        # Return parameters, dataset size, and our custom metrics dictionary
        return self.get_parameters(config={}), len(self.trainloader.dataset), {
            "train_losses": losses, 
            "val_acc": val_acc
        }

    def evaluate(self, parameters, config):
        # NEW: Print when a client starts evaluation
        print(f"[Client {self.cid}] Starting evaluation...")

        self.set_parameters(parameters)
        loss, accuracy = test(self.net, self.valloader)
        
        # NEW: Print the client's evaluation results
        print(f"✅ [Client {self.cid}] Evaluation results - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

        return float(loss), len(self.valloader.dataset), {"accuracy": float(accuracy)}


# Flower ClientApp
app = ClientApp(
    client_fn=FlowerClient,
)

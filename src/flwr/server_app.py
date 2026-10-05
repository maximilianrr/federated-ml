import json
import torch
import torch.nn as nn
import os

from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from src.flwr.test import ModelTester
from src.flwr.model import VisionModel

class FederatedServerManager: 

    def __init__(self, model_class, global_test_loader, input_size, server_rounds=5, fraction_eval=0.2, num_classes=5) -> None:
            
            """
            Create a Flower ClientApp for federated learning.
    
            Args:
                model_class (nn.Module): The PyTorch model class to be used.
                global_test_loader (DataLoader): DataLoader for the global test data.
                input_size (int): The size of the input features.
                server_rounds (int): The number of rounds to run the server. Defaults to 5.
                fraction_eval (float): The fraction of clients to evaluate. Defaults to 0.2.
                num_classes (int): The number of output classes. Defaults to 5.
            Returns:
                ServerApp: A Flower ServerApp instance configured for federated learning.
            """
    
            self.model_class = model_class
            self.global_test_loader = global_test_loader
            self.input_size = input_size
            self.server_rounds = server_rounds
            self.fraction_eval = fraction_eval
            self.num_classes = num_classes
            self.history_metrics = {}
    
            # Instantiate Flower's ClientApp
            self.app = ServerApp()
            self.app.main()(self.main)


    def main(self, grid: Grid, context: Context) -> None:
        num_rounds_raw = context.run_config.get("num-server-rounds", self.server_rounds)
        num_rounds = int(num_rounds_raw)

        fraction_raw = context.run_config.get("fraction-evaluate", self.fraction_eval)
        fraction_evaluate = float(fraction_raw)

        # initialize the global model dynamically 
        global_model = self.model_class(input_size=self.input_size, num_classes=self.num_classes)
        arrays = ArrayRecord(global_model.state_dict())

        # create own history tracker
        self.history_metrics = {
            "metrics_centralized": {
                "test_accuracy": [],
                "test_loss": []
            }
        }

        strategy = FedAvg(
            fraction_evaluate=fraction_evaluate
        )
        
        # Start strategy, run FedAvg for `num_rounds`
        result = strategy.start(
            grid=grid,
            initial_arrays=arrays,
            train_config=ConfigRecord(),
            num_rounds=num_rounds,
            evaluate_fn=self.global_evaluate,
        )
    
        # Save final model to disk
        print("\nSaving final model to disk...")
        os.makedirs("outputs", exist_ok=True)
        
        state_dict = result.arrays.to_torch_state_dict()
        torch.save(state_dict, f"final_model_fedavg.pt")

        # Save metrics JSON directly from the tracker
        with open(f"outputs/global_fedavg_baseline.json", "w") as f:
            json.dump(self.history_metrics, f)


    # the evaluate function closure has access to global_test_loader
    def global_evaluate(self, server_round: int, parameters: ArrayRecord):
        if self.global_test_loader is None:
            return None
        
        model = self.model_class(input_size=self.input_size, num_classes=self.num_classes)
        model.load_state_dict(parameters.to_torch_state_dict())
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        tester = ModelTester()

        criterion = nn.CrossEntropyLoss()
        test_loss, test_acc = tester.test_fn(model, criterion, self.global_test_loader, device)

        # Update the history metrics
        self.history_metrics["metrics_centralized"]["test_accuracy"].append([server_round, test_acc])
        self.history_metrics["metrics_centralized"]["test_loss"].append([server_round, test_loss])

        metrics = {"test_loss": test_loss, "test_accuracy": test_acc}
        return MetricRecord(metrics)


server_manager = FederatedServerManager(
    model_class=VisionModel, 
    global_test_loader=TEST_LOADER, 
    input_size=128
)

server_app = server_manager.app
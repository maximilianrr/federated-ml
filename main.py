from src.model import VisionModel
from src.flwr.train import ModelTrainer as train
from src.flwr.client_app import FederatedClientManager
from src.flwr.server_app import FederatedServerManager
import torch 


def main():
    """
    The main function to run the federated learning application
    """

    # Instantiate custom class with all required arguments
    client_manager = FederatedClientManager(
        model_class=VisionModel,
        criterion=torch.nn.CrossEntropyLoss(),
        learning_rate=0.01,
        train_loaders=TRAIN_LOADERS,
        test_loaders=TEST_LOADERS,
        input_size=128
    )

    # Expose the actual flwr ClientApp instance as `app`
    app = client_manager.app

    server_manager = FederatedServerManager(
        model_class=VisionModel, 
        global_test_loader=TEST_LOADER, 
        input_size=128)
    server_app = server_manager.app

if __name__ == "__main__": 
    main()
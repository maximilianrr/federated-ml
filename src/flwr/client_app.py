import torch
from torch import optim

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from flwr.train import ModelTrainer
from flwr.test import ModelTester
from src.flwr.model import VisionModel

class FederatedClientManager: 

    def __init__(self, model_class, criterion, learning_rate, train_loaders, test_loaders, input_size, num_classes=5, is_dp=False) -> None:
        """
        Create a Flower ClientApp for federated learning.

        Args:
            model_class (nn.Module): The PyTorch model class to be used.
            train_loaders (list): List of DataLoaders for training data for each client.
            test_loaders (list): List of DataLoaders for test data for each client.
            input_size (int): The size of the input features.
            num_classes (int): The number of output classes. Defaults to 5.
        Returns:
            ClientApp: A Flower ClientApp instance configured for federated learning.
        """

        self.model_class = model_class
        self.criterion = criterion
        self.learning_rate = learning_rate
        self.train_loaders = train_loaders
        self.test_loaders = test_loaders
        self.input_size = input_size
        self.num_classes = num_classes
        self.is_dp = is_dp

        # Instantiate Flower's ClientApp
        self.app = ClientApp()
        self.app.train()(self.train)
        self.app.evaluate()(self.evaluate)


    def train(self, msg: Message, context: Context):
        """Train the model on local data."""
        
        # Access state using `self.`
        model = self.model_class(input_size=self.input_size, num_classes=self.num_classes)
        # retrieve generic record
        arrays_record = msg.content["arrays"]
        # check if type of arrays_record matches requirements
        if isinstance(arrays_record, ArrayRecord):
            model.load_state_dict(arrays_record.to_torch_state_dict())
        else:
            # Fallback/error handling if something went terribly wrong with the payload
            raise TypeError("Expected an ArrayRecord under the key 'arrays'")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)

        partition_id = context.node_config["partition-id"]
        train_loader = self.train_loaders[partition_id]

        raw_mu = msg.content["config"].get("proximal_mu", 0.0)
        if isinstance(raw_mu, (float, int, str)):
            proximal_mu = float(raw_mu)
        else:
            proximal_mu = 0.0

        # instantiate trainer class
        trainer = ModelTrainer()

        # Assuming train_fn_dp and train_fn are imported globally
        if self.is_dp:
            train_loss = trainer.train_fn_dp(model, self.criterion, optimizer, train_loader, device, proximal_mu=proximal_mu)
        else:
            train_loss = trainer.train_fn(model, self.criterion, optimizer, train_loader, device, proximal_mu=proximal_mu)

        metrics = {"train_loss": train_loss, "num-examples": len(train_loader.dataset)}

        return Message(
            content=RecordDict({
                "arrays": ArrayRecord(model.state_dict()),
                "metrics": MetricRecord(metrics),
            }),
            reply_to=msg
        )

    def evaluate(self, msg: Message, context: Context):
        """Evaluate the model on local data."""
        
        model = self.model_class(input_size=self.input_size, num_classes=self.num_classes)
        # retrieve generic record
        arrays_record = msg.content["arrays"]
        # check if type of arrays_record matches requirements
        if isinstance(arrays_record, ArrayRecord):
            model.load_state_dict(arrays_record.to_torch_state_dict())
        else:
            # Fallback/error handling if something went terribly wrong with the payload
            raise TypeError("Expected an ArrayRecord under the key 'arrays'")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        partition_id = context.node_config["partition-id"]
        test_loader = self.test_loaders[partition_id]

        tester = ModelTester()

        # Assuming test_fn is imported globally
        eval_loss, eval_acc = tester.test_fn(model, self.criterion, test_loader, device)

        metrics = {"eval_loss": eval_loss, "eval_accuracy": eval_acc, "num-examples": len(test_loader.dataset)}

        return Message(
            content=RecordDict({
                "metrics": MetricRecord(metrics),
            }),
            reply_to=msg
        )


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
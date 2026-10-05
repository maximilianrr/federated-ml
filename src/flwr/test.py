import torch

class ModelTester: 

    def test_fn(self, model, criterion, test_loader, device): 
        """
        Test the model on the given test data.

        Args:
            model (nn.Module): The PyTorch model to be tested.
            test_loader (DataLoader): DataLoader for the test data.
            device (torch.device): The device to run the testing on (CPU or GPU).
        Returns:
            float: The accuracy of the model on the test data.
        """

        model.eval()  # Set the model to evaluation mode
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():  # Disable gradient calculation for testing
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)  # Forward pass
                loss = criterion(outputs, labels)  # Compute the loss
                total_loss += loss.item()

                # calculate the accuracy of the model on the test data
                _, predicted = torch.max(outputs.data, 1)  # Get the index of the max log-probability
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        total_loss /= len(test_loader)

        accuracy = 100 * correct / total
        # print(f"Average loss on the test data: {total_loss:.4f}")
        # print(f"Accuracy of the model on the test data: {accuracy:.2f}%")
        return total_loss, accuracy
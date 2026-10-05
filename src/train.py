import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelTrainer(): 
    def train_fn(self, model: nn.Module, criterion, optimizer, train_loader, device, num_epochs=5, proximal_mu=0.0):
        """
        Train the model on the given data.
    
        Args:
            model (nn.Module): The vision model to be trained.
            train_loader (DataLoader): DataLoader for the training data.
            device (torch.device): The device to run the training on (CPU or GPU).
            proximal_mu (float): The proximal parameter for the training.
        Returns:
            float: The average training loss over all epochs.
        """

        # Set the model to training mode
        model.train()
        global_params = None

        if proximal_mu > 0:
            global_params = [param.detach().clone() for param in model.parameters()]

        train_loss = 0.0

        for epoch in range(num_epochs):
            running_loss = 0.0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad() 
                outputs = model(inputs) 
                loss = criterion(outputs, labels)

                # Apply FedProx penalty only if mu > 0 
                if proximal_mu > 0:
                    assert global_params is not None
                    proximal_term = 0.0
                    for local_weights, global_weights in zip(model.parameters(), global_params):
                        proximal_term += torch.square(local_weights - global_weights).sum()
                    # Add the proximal term to the loss
                    loss += (proximal_mu / 2) * proximal_term  

                loss.backward()  # Backward pass
                optimizer.step()  # Update the weights

                running_loss += loss.item()

            train_loss = running_loss / len(train_loader)

            # debugging print, not needed in fl as it might get too noisy
            #print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

        return train_loss



    def train_fn_dp(self, model: nn.Module, criterion, optimizer, train_loader, device, num_epochs=5, proximal_mu=0.0, clip_norm=1.0, noise_std=0.001):
        """
        Train the model on the given data with differential privacy.

        Args:
            model (nn.Module): The PyTorch model to be trained.
            train_loader (DataLoader): DataLoader for the training data.
            device (torch.device): The device to run the training on (CPU or GPU).
            num_epochs (int): The number of epochs to train the model.
            proximal_mu (float): The proximal term coefficient for FedProx.
            clip_norm (float): The norm to which gradients are clipped.
            noise_std (float): The standard deviation of the noise added for differential privacy.
        Returns:
            float: The average training loss over all epochs.
        """

        # Set the model to training mode
        model.train()
        global_params = None

        if proximal_mu > 0:
            global_params = [param.detach().clone() for param in model.parameters()]

        train_loss = 0.0

        for epoch in range(num_epochs):
            running_loss = 0.0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad() 

                outputs = model(inputs) 
                loss = criterion(outputs, labels)

                # Apply FedProx penalty only if mu > 0 
                if proximal_mu > 0:
                    assert global_params is not None
                    proximal_term = 0.0
                    for local_weights, global_weights in zip(model.parameters(), global_params):
                        proximal_term += torch.square(local_weights - global_weights).sum()
                    # Add the proximal term to the loss
                    loss += (proximal_mu / 2) * proximal_term

                loss.backward() 

                # Clip gradients for differential privacy
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)

                # Add noise to gradients for differential privacy
                with torch.no_grad():
                    for param in model.parameters():
                        if param.grad is not None:
                            noise = torch.normal(mean=0, std=noise_std, size=param.grad.size()).to(device)
                            param.grad += noise


                optimizer.step()  # Update the weights

                running_loss += loss.item()

            train_loss = running_loss / len(train_loader)

        return train_loss
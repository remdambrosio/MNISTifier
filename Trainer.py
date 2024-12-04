# Title: Trainer.py
# Authors: Rem D'Ambrosio
# Created: 2024-12-02
# Description: initial structure by Mayur Ingole

import torch  
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from CNN import CNN


class Trainer():  
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


    def train_model(self, epochs):
        """
        Trains CNN model
        """
        learn_rate = 0.001
        batch_size = 64

        train_dataset = datasets.MNIST(root="dataset/", download=True, train=True, transform=transforms.ToTensor())
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

        model = CNN().to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learn_rate)

        for epoch in range(epochs):
            running_loss = 0.0
            print(f"===== Epoch {epoch + 1}/{epochs} =====")
            for data, targets in train_loader:
                data = data.to(self.device)
                targets = targets.to(self.device)

                scores = model(data)
                loss = criterion(scores, targets)
                running_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()

                optimizer.step()
            print(f"Average Loss: {running_loss / len(train_loader):.4f}")
        return model


# ==================================================================================================
# I/O FUNCTIONS
# ==================================================================================================    


    def save_model(self, model, path):
        """
        Saves trained CNN model model to file
        """
        torch.save(model.state_dict(), path)
        return
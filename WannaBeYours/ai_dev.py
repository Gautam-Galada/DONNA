import random
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10, ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import io
import os
import json

class AIDev:
    def __init__(self, learning_rate=0.001, batch_size=64, num_epochs=2, hidden_size=512, image_size=(28, 28), input_channels=1, random_seed=42):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.hidden_size = hidden_size
        self.image_size = image_size
        self.input_channels = input_channels
        self.random_seed = random_seed
        self.model = None
        self.train_loader = None
        self.criterion = None
        self.optimizer = None
        self.set_random_seed(random_seed)

    def set_random_seed(self, seed):
        self.random_seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    class SimpleNet(nn.Module):
        def __init__(self, hidden_size, input_size):
            super().__init__()
            self.flatten = nn.Flatten()
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 10)
            )

        def forward(self, x):
            x = self.flatten(x)
            return self.linear_relu_stack(x)

    def modelinit(self, dataset_choice, data_dir=None):
        if dataset_choice == "mnist":
            self.image_size = (28, 28)
            self.input_channels = 1
            transform = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            train_dataset = MNIST(root='./data', train=True, transform=transform, download=True)
        elif dataset_choice == "cifar10":
            self.image_size = (32, 32)
            self.input_channels = 3
            transform = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            train_dataset = CIFAR10(root='./data', train=True, transform=transform, download=True)
        else:
            self.input_channels = 3
            transform = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            train_dataset = ImageFolder(root=data_dir, transform=transform)

        self.train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
        input_size = self.image_size[0] * self.image_size[1] * self.input_channels
        self.model = self.SimpleNet(self.hidden_size, input_size)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return self.model

    def train(self):
        self.model.train()
        epoch_losses = []
        for epoch in range(self.num_epochs):
            epoch_loss = 0
            for _, (data, targets) in enumerate(self.train_loader):
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            epoch_loss /= len(self.train_loader)
            epoch_losses.append(epoch_loss)
            print(f'Epoch {epoch + 1}, Loss: {epoch_loss:.4f}')
        return epoch_losses

    def preprocess_image(self, imgbytes):
        image_stream = io.BytesIO(imgbytes)
        image = Image.open(image_stream).convert('RGB' if self.input_channels == 3 else 'L')
        transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        image = transform(image).unsqueeze(0)
        return image

    def test(self, image_bytes):
        image_tensor = self.preprocess_image(image_bytes)
        self.model.eval()
        res = self.model(image_tensor)
        predicted_class = res.argmax(dim=1).item()
        print(f'Predicted class: {predicted_class}')
        return predicted_class

    def datatype(self):
        return "images"

    def plot_losses(self, epoch_losses):
        plt.figure(figsize=(10, 5))
        plt.plot(epoch_losses, marker='o', linestyle='-', color='b')
        plt.title('Training Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig('training_loss_plot.png')
        plt.close()

    def save_log(self, hyperparameters, final_loss, random_seed):
        log = {
            "hyperparameters": hyperparameters,
            "final_loss": final_loss,
            "random_seed": random_seed
        }
        with open('training_logs.json', 'a') as log_file:
            log_file.write(json.dumps(log) + '\n')

    def show_logs(self):
        if not os.path.exists('training_logs.json'):
            return []

        with open('training_logs.json', 'r') as log_file:
            logs = log_file.readlines()
            logs = [json.loads(log) for log in logs]
            return logs

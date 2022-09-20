print("Script based on https://github.com/BrianPulfer/PapersReimplementations/blob/master/vit/vit_torch.py")
print("This script is not meant to be used as a solution, but rather as a working example to comprehensively examine building blocks of a vision transformer implemenation and how to train it.")

import numpy as np

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from torchvision.transforms import ToTensor
from torchvision.datasets.mnist import MNIST

import matplotlib.pyplot as plt

from building_blocks import patchify, get_positional_embeddings, MyViT, MyViTBlock

np.random.seed(0)
torch.manual_seed(0)

# Hyperparameters
N_EPOCHS = 5
LR = 0.005


def main():
    # Loading data
    transform = ToTensor()

    train_set = MNIST(root='/scratch-local/cdtemp/richard/datasets', train=True, download=True, transform=transform)
    test_set = MNIST(root='/scratch-local/cdtemp/richard/datasets', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, shuffle=True, batch_size=128)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=128)

    # Define model and device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")
    model = MyViT(chw=(1, 28, 28), n_patches=7, n_blocks=2, hidden_d=8, n_heads=2, out_d=10).to(device)

    # Train loop
    optimizer = Adam(model.parameters(), lr=LR)
    criterion = CrossEntropyLoss()

    for epoch in tqdm(range(N_EPOCHS), desc="Training"):
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} in training", leave=False):

            # unpack batch and assign devices
            x, y = batch
            x, y = x.to(device), y.to(device)

            # make prediction and calculate loss
            y_hat = model(x)
            loss = criterion(y_hat, y)
            train_loss += loss.detach().cpu().item() / len(train_loader)

            optimizer.zero_grad() # Initialize Gradient
            loss.backward() # Back Propagation
            optimizer.step() # Gradient Descent

        print(f"Epoch {epoch + 1}/{N_EPOCHS} loss: {train_loss:.2f}")

    # Test loop
    with torch.no_grad():
        correct, total = 0, 0
        test_loss = 0.0

        for batch in tqdm(test_loader, desc="Testing"):
            x, y = batch
            x, y = x.to(device), y.to(device)

            y_hat = model(x)
            loss = criterion(y_hat, y)
            test_loss += loss.detach().cpu().item() / len(test_loader)

            correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
            total += len(x)
        print(f"Test loss: {test_loss:.2f}")
        print(f"Test accuracy: {correct / total * 100:.2f}%")


if __name__ == '__main__':

    # Visualize Positional Embeddings
    plt.imshow(get_positional_embeddings(100, 300), cmap='hot', interpolation='nearest')


    # Train ViT on toy dataset
    main()

    plt.show()
import argparse
import os
from typing import Union

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from MNISTNet import MNISTNet

#ArgumentParser instance
parser = argparse.ArgumentParser(description='Settings and hyperparameters')
parser.add_argument('--use_cuda', type=bool, default=False, help='Determine CUDA training, default is false')
parser.add_argument('--batch_size', type=int, default=100, help='Input batch size, default = 100')
parser.add_argument('--lr', type=float, default=0.01, help='Input learning rate, default = 0.01')
parser.add_argument('--n_epochs', type=int, default=20, help='Input number of epochs, default = 20')
args = parser.parse_args()

#determine whether GPU training is enabled
device = torch.device("cuda" if args.use_cuda else "cpu")

#declaring dataset and result paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(ROOT_DIR, 'data')
results_path = os.path.join(ROOT_DIR, 'results')

#creating paths if not existing
if not os.path.exists(data_path):
    os.mkdir(data_path)
if not os.path.exists(results_path):
    os.mkdir(results_path)


def calculate_mean_std() -> Union[float, float]:
    '''Calculating mean and standard deviadion of the dataset. Required for normalisation'''
    trans = transforms.Compose([transforms.ToTensor()])
    train_set = dataset.MNIST(root=data_path, train=True, transform=trans, download=True)
    train_loader = DataLoader(train_set, batch_size=len(train_set))
    train_dataset_array = next(iter(train_loader))[0].numpy()
    mean = train_dataset_array.mean()
    std = train_dataset_array.std()
    return mean, std


def train(epoch: int, model: nn.Module, optimizer: optim.SGD):
    '''training loop + preparing data for loss plot'''
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
                   100. * batch_idx / len(train_loader), loss.item()))
        train_losses.append(loss.item())
        train_counter.append(
            (batch_idx * args.batch_size) + ((epoch - 1) * len(train_loader.dataset)))
        torch.save(model.state_dict(), os.path.join(results_path, 'model.pth'))


def test(model: nn.Module):
    '''test function + preparing data for loss plot'''
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    accuracy.append(100. * correct / len(test_loader.dataset))
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

#data preprocessing
mean, std = calculate_mean_std()
trans = transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize((mean,), (std,))])
train_set = dataset.MNIST(root=data_path, train=True, transform=trans, download=True)
test_set = dataset.MNIST(root=data_path, train=False, transform=trans, download=True)

#loading data to DataLoader class instances
train_loader = torch.utils.data.DataLoader(
    dataset=train_set,
    batch_size=args.batch_size,
    shuffle=True)
test_loader = torch.utils.data.DataLoader(
    dataset=test_set,
    batch_size=args.batch_size,
    shuffle=False)

#preparing lists for loss plot and accuracy plot
train_losses = []
train_counter = []
test_losses = []
test_counter = [i * len(train_loader.dataset) for i in range(args.n_epochs)]
accuracy = []
accuracy_counter = [i for i in range(1, args.n_epochs + 1)]

#initiating model and optimizer instances
model = MNISTNet().to(device)
optimizer = optim.SGD(model.parameters(), lr=args.lr)

#train loop
for epoch in range(1, args.n_epochs + 1):
    train(epoch, model, optimizer)
    test(model)

#preparing and saving loss plot
fig = plt.figure()
plt.plot(train_counter, train_losses, color='blue')
plt.scatter(test_counter, test_losses, color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training examples in millions')
plt.ylabel('negative log likelihood loss')
fig.savefig(os.path.join(ROOT_DIR, 'loss_plot.png'), dpi=fig.dpi)

#preparing and saving accuracy plot
fig2 = plt.figure()
plt.plot(accuracy_counter, accuracy, color='green')
plt.xlabel('number of epochs')
plt.ylabel('accuracy score')
ticks = [i for i in range(0, args.n_epochs + 1, 5)]
plt.xticks(ticks)
fig2.savefig(os.path.join(ROOT_DIR, 'acc_plot.png'), dpi=fig.dpi)

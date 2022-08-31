import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision.datasets import MNIST
import torchvision.transforms as transforms

# hyper parameters
PATH_TO_DATA = './data'
PATH_TO_SAVE_MODEL = './lenet.pth'
N_EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 1e-3

# reproducibility
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# use cuda if possible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# transform image to normalized tensor
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])

# dataset MNIST:
# http://yann.lecun.com/exdb/mnist/
trainset = MNIST(root=PATH_TO_DATA, train=True, download=True, transform=transform)  # use transform
testset = MNIST(root=PATH_TO_DATA, train=False, download=True, transform=transform)  # use transform

# dataloader
trainloader = DataLoader(dataset=trainset, batch_size=BATCH_SIZE, shuffle=True)  # shuffle only train
testloader = DataLoader(dataset=testset, batch_size=BATCH_SIZE, shuffle=False)


# convolutional neural network
class LeNet(nn.Module):
    """
    LeNet architecture:
    http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf
    """

    def __init__(self):
        super().__init__()
        # convolutional layers
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv_2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)

        # linear layers
        self.linear_1 = nn.Linear(16*4*4, 120)
        self.linear_2 = nn.Linear(120, 84)
        self.linear_3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.avg_pool2d(torch.tanh(self.conv_1(x)), 2)
        x = F.avg_pool2d(torch.tanh(self.conv_2(x)), 2)

        x = x.view(-1, 16*4*4)  # flatten before linear layers
        x = torch.tanh(self.linear_1(x))
        x = torch.tanh(self.linear_2(x))
        x = self.linear_3(x)
        return x


# init model
model = LeNet().to(device)

# init criterion to compute loss and optimizer to update parameters
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# train loop on epoch
def train_epoch(model, dataloader, criterion, optimizer, device=device):
    avg_loss = 0  # calculate average loss
    avg_accuracy = 0  # calculate average accuracy

    model.train()  # use model for training (actual for dropout, batch normalization, etc.)
    for inputs, labels in dataloader:  # iterate over training dataloader
        inputs, labels = inputs.to(device), labels.to(device)  # transfer data to device

        # forward pass
        outputs = model(inputs)  # use model for prediction
        loss = criterion(outputs, labels)  # compute loss using prediction and true labels

        # backward pass
        loss.backward()  # propagate model parameters gradients w.r.t loss
        optimizer.step()  # update parameters
        optimizer.zero_grad()  # zero gradients

        # calculate batch accuracy
        with torch.no_grad():  # disable gradient calculations for accuracy
            _, pred = torch.max(outputs, dim=-1)
            accuracy = torch.sum(pred == labels)

        avg_loss += loss.item()
        avg_accuracy += accuracy.item()

    # divide by the number of elements in dataset
    avg_loss /= len(dataloader.dataset)
    avg_accuracy /= len(dataloader.dataset)

    return avg_loss, avg_accuracy


# validation loop on epoch
def validate_epoch(model, dataloader, criterion, device=device):
    avg_loss = 0  # calculate average loss
    avg_accuracy = 0  # calculate average accuracy

    model.eval()  # use model for testing (actual for dropout, batch normalization, etc.)
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        with torch.no_grad():  # disable gradient calculations
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # calculate batch accuracy
            _, pred = torch.max(outputs, dim=-1)
            accuracy = torch.sum(pred == labels)

        avg_loss += loss.item()
        avg_accuracy += accuracy.item()

    # divide by the number of elements in dataset
    avg_loss /= len(dataloader.dataset)
    avg_accuracy /= len(dataloader.dataset)

    return avg_loss, avg_accuracy


# train / validation loop
for epoch in range(N_EPOCHS):
    # train
    train_loss, train_accuracy = train_epoch(model, trainloader, criterion, optimizer, device=device)

    # validate
    val_loss, val_accuracy = validate_epoch(model, testloader, criterion, device=device)

    # display
    print(f'epoch [{epoch+1}/{N_EPOCHS}]')
    print(f'loss (train/test): {train_loss:.4f}/{val_loss:.4f}')
    print(f'accuracy (train/test): {train_accuracy:.4f}/{val_accuracy:.4f}\n')

# save model (state dict)
torch.save(model.state_dict(), PATH_TO_SAVE_MODEL)

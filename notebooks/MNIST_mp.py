import os
import random
import pathlib
import time

import numpy as np

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torchvision
from torchvision import transforms

from torch.utils.data import DataLoader

import torch.nn as nn

# print(torch.backends.cuda.is_built())
# print(torch.__version__)

MNIST_FOLDER = r'C:\Users\User\Documents\Progs\Projects\MNIST'
MNIST_NOTEBOOKS_FOLDER = os.path.join(MNIST_FOLDER, 'notebooks')
MNIST_MODELS_FOLDER = os.path.join(MNIST_FOLDER, 'models')
MNIST_RESULTS_FOLDER = os.path.join(MNIST_FOLDER, 'results')

MEAN = 0.1307
STD = 0.3081

class SimpleCNN(nn.Module):
    def __init__(self, num_classes = 10):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(kernel_size=3, in_channels=1, out_channels=32, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.maxpool_1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_2 = torch.nn.Conv2d(kernel_size=3, in_channels=32, out_channels=64, stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.maxpool_2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_3 = torch.nn.Conv2d(kernel_size=3, in_channels=64, out_channels=128, stride=1, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(128)
        self.maxpool_3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.relu = torch.nn.ReLU()

        self.lin_1 = torch.nn.Linear(in_features=1152, out_features=256)
        self.lin_2 = torch.nn.Linear(in_features=256, out_features=num_classes)

        self.dropout_conv = torch.nn.Dropout(p=0.1)
        self.dropout_lin = torch.nn.Dropout(p=0.2)
    def forward(self, x):

        x = self.maxpool_1(self.relu(self.bn1(self.conv_1(x)))) # (batch_size, 1, 28, 28) -> (batch_size, 32, 14, 14)
        x = self.dropout_conv(x)
        x = self.maxpool_2(self.relu(self.bn2(self.conv_2(x)))) # (batch_size, 32, 14, 14) -> (batch_size, 64, 7, 7)
        x = self.dropout_conv(x)
        x = self.maxpool_3(self.relu(self.bn3(self.conv_3(x)))) # (batch_size, 64, 7, 7) -> (batch_size, 128, 3, 3)

        x = x.view(x.size(0), -1)  # (batch_size, 128, 3, 3) -> (batch_size, 128*3*3 = 1152)

        x = self.relu(self.lin_1(x))  # (batch_size, 1152) -> (batch_size, 256)
        x = self.dropout_lin(x)
        x = self.lin_2(x)               # (batch_size, 256) -> (batch_size, 10)

        return x



random_rotation = transforms.Compose([
    transforms.RandomAffine(degrees=20, translate=(0.1, 0.1)),  # Случайный поворот
    transforms.ToTensor(),                 # Преобразование в тензор [1, 28, 28], [0.0, 1.0]
    transforms.Normalize((MEAN,), (STD,))
])

no_rotation = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((MEAN,), (STD,))
])

class Transformed_Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        
        if self.transform is not None:
            image = self.transform(image)
        
        return image, label

    def __len__(self):
        return len(self.dataset)

def save_result(result, folder, name):
    current_file_path = os.path.join(folder, name)
    with open(current_file_path, mode="wb") as outfile_current:
        result = np.array(result)
        result.tofile(outfile_current)

if __name__ == '__main__':
    lr = 10**(-3)
    batch_size = 512
    num_epochs = 20
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    loss_fn = nn.CrossEntropyLoss()

    train_data = torchvision.datasets.MNIST("./", train=True, download=True)
    test_data = torchvision.datasets.MNIST("./", train=False, download=True)

    train_dataset = Transformed_Dataset(train_data, transform=no_rotation)
    train_dataset_rotated = Transformed_Dataset(train_data, transform=random_rotation)

    test_dataset = Transformed_Dataset(test_data, transform=no_rotation)
    #test_dataset_rotated = Transformed_Dataset(test_data, transform=random_rotation)

    train_dataloader = torch.utils.data.DataLoader(train_dataset_rotated, batch_size=batch_size, shuffle=True, num_workers=8)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    number_of_train_batches = len(train_dataloader)
    number_of_test_batches = len(test_dataloader)

    print_every_train = max(1, number_of_train_batches // 20) # Количество батчей для вывода (каждые 20%)
    print_every_test = max(1, number_of_test_batches // 20)

    best_accuracy = 0
    best_epoch = 0

    start_time = time.time()

    for i in range(num_epochs):
        
        train_loss = 0
        train_labels = []
        train_true_labels = []

        for j, (X, target) in enumerate(train_dataloader):
            X = X.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            preds = model(X)
            loss_value = loss_fn(preds, target)
            loss_value.backward()
            optimizer.step()

            train_loss = train_loss + loss_value.item()
            train_labels.extend(preds.argmax(axis=1).cpu().numpy().tolist())
            train_true_labels.extend(target.cpu().numpy().tolist())

            # if (j+1) % print_every_train == 0:
            #     percent_complete = ((j+1) / number_of_train_batches) * 100
            #     print(f'Epoch: {i}, {percent_complete:.1f}%, Loss: {train_loss / (j+1):.4f}, Time: {time.time() - start_time:.2f} sec')

        train_loss /= number_of_train_batches
        train_accuracy = np.mean(np.array(train_labels) == np.array(train_true_labels))

        save_result(train_labels, folder=MNIST_RESULTS_FOLDER, name=f'epoch_{i}_train_labels_rotated.npy')
        save_result(train_true_labels, folder=MNIST_RESULTS_FOLDER, name=f'epoch_{i}_train_true_labels_rotated.npy')

        print(f'TRAIN: epoch = {i}, train_loss = {train_loss:.4f}, accuracy = {train_accuracy:.4f}, Time: {time.time() - start_time:.2f} sec')

        test_loss = 0
        test_labels = []
        test_true_labels = []

        with torch.no_grad():
            for j, (X, target) in enumerate(test_dataloader):
                X = X.to(device)
                target = target.to(device)
                
                preds = model(X)
                loss_value = loss_fn(preds, target)
                
                test_loss += loss_value.item()
                test_labels.extend(preds.argmax(axis=1).cpu().numpy().tolist())
                test_true_labels.extend(target.cpu().numpy().tolist())
            
            test_loss /= number_of_test_batches
            test_accuracy = np.mean(np.array(test_labels) == np.array(test_true_labels))
        
            save_result(test_labels, folder=MNIST_RESULTS_FOLDER, name=f'epoch_{i}_test_labels_rotated.npy')
            save_result(test_true_labels, folder=MNIST_RESULTS_FOLDER, name=f'epoch_{i}_test_true_labels_rotated.npy')

            print(f'TEST: epoch =  {i}, test_loss = {test_loss:.4f}, accuracy = {test_accuracy:.4f}, Time: {time.time() - start_time:.2f} sec')

            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                best_epoch = i
                torch.save(model.state_dict(), os.path.join(MNIST_MODELS_FOLDER, 'best_model_rotated.pth'))
        
        scheduler.step()
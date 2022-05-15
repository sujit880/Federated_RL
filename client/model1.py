# License: Apache
# Author: Sujit Chowdhury

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

from torch.utils.data import DataLoader, Subset
from torchvision.transforms import Compose, ToTensor, Resize
from sklearn.model_selection import train_test_split
from torchvision.datasets import ImageFolder

cudnn.benchmark = True
plt.ion()   # interactive mode

ALIAS = "Resnet18"

def train_test_dataset(dataset, val_split=0.25):
    train_idx, test_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets ={}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['test'] = Subset(dataset, test_idx)
    return datasets

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def load_data(data_dir, num_workers=2, batch_size=4):
    dataset = ImageFolder(data_dir, transform=Compose([Resize((224,224)), ToTensor()]))
    print("len: ", len(dataset))
    datasets = train_test_dataset(dataset)
    print(len(datasets['train']))
    print(datasets['train'][0][1])
    dataloaders = {x: DataLoader(datasets[x], batch_size=batch_size,
                                             shuffle=True, num_workers=num_workers)
              for x in ['train', 'test']}
    dataset_sizes = {x: len(datasets[x]) for x in ['train', 'test']}
    class_names = dataset.classes
    print("Classes: ", class_names)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("\nDevice: ", device)
    return dataloaders, dataset_sizes, class_names, device


def train_model(model, dataloaders, dataset_sizes, device, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            # print("inside 1st for")
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            # print("before 2nd for")
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                # print("inside 2nd for1")
                inputs = inputs.to(device)
                labels = labels.to(device)
                # print("inside 2nd for")
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def get_model(model_name):
    print("\nModel: ", model_name)
    switcher = {
        'Resnet18': models.resnet18(pretrained=True), 
        'resnet18': models.resnet18(pretrained=True),
        'alexnet': models.alexnet(pretrained=True),
        'vgg16' : models.vgg16(pretrained=True),
        'squeezenet': models.squeezenet1_0(pretrained=True),
        'densenet': models.densenet161(pretrained=True),
        'inception': models.inception_v3(pretrained=True),
        'googlenet': models.googlenet(pretrained=True),
        
        1: models.resnet18(pretrained=True),
        2: models.resnet18(pretrained=True),
    }
    ALIAS = model_name
    return switcher.get(model_name, None)

def get_model_ft(model_name, model):
    print("\nModel: ", model_name)
    switcher = {
        'Resnet18': model_ft.fc, 
        'resnet18': model_ft.fc,
        'alexnet': model_ft.classifier[0],
        'vgg16' :model_ft.classifier[0],
        'densenet': model.classifier[0],
    }
    return switcher.get(model_ft, None)


def model_finetune(model, class_names, device):
    
    model_ft = get_model(model_name=model)
    
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    if model == "resnet18":   
        num_ftrs = model_ft.fc.in_features     
        model_ft.fc = nn.Linear(num_ftrs, len(class_names))
    
    if model == 'vgg16':
        num_ftrs= model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, len(class_names))

    if model =='densenet':
        num_ftrs= model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, len(class_names))
    model_ft = model_ft.to(device)
    
    print("number of features: ", num_ftrs)
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    return model_ft, criterion, optimizer_ft, exp_lr_scheduler



if __name__ == '__main__':
    print("in model1 main\n")
    dataset_path = "./Dataset/"
    data_dir_name = input("Enter the name of datset: Ex. hospital-1: ")
    data_dir = dataset_path + data_dir_name
    dataloaders, dataset_sizes, class_names, device = load_data(data_dir=data_dir)

    print(dataset_sizes, device)

    model_ft, criterion, optimizer_ft, exp_lr_scheduler = model_finetune(class_names, device)

    model_ft = train_model(model_ft, dataloaders, dataset_sizes, device, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=5)

    print("\n\nState_dict: *********************\n", model_ft.state_dict())



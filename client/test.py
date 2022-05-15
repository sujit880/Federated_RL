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
import modman_layered as modman
import sys
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import Compose, ToTensor, Resize
from sklearn.model_selection import train_test_split
from torchvision.datasets import ImageFolder

cudnn.benchmark = True
plt.ion()   # interactive mode

import model1
P= print
ip_address = "172.16.26.73"  # server macine ip address
# API endpoint
# URL = "http://"+ip_address+":5500/api/model/"
URL = "http://localhost:5500/api/model/"
ALIAS = sys.argv[1]

if __name__ == '__main__':
    dataset_path =  "./"+input("Which dataset to run: Ex. ") +"/"
    model_path = "./Models/"
    data_dir_name = "hospital-"+sys.argv[2] #input("Enter the name of datset: Ex. data-1: ")
    model_name = ALIAS #input("Enter the name of model Ex. resnet18/vgg16/densenet: ")
    data_dir = dataset_path + data_dir_name

    dataloaders, dataset_sizes, class_names, device = model1.load_data(data_dir=data_dir)

    print(dataset_sizes, device)

    model_ft, criterion, optimizer_ft, exp_lr_scheduler = model1.model_finetune(model_name, class_names, device)
    

    ##############################################
    # Fetch Initial Model Params (If Available)
    ##############################################

    # reply = modman.send_model_params(
    #         URL, modman.convert_tensor_to_list(pie.Q.state_dict()), PIE_PARAMS.LR, ALIAS)
    # print("Response:",reply)

    while modman.get_model_lock(URL):  # wait if model updation is going on
        print("Waiting for Model Lock Release.")

    global_params, n_push, log_id, is_available = modman.fetch_params(URL+'get', list(model_ft.state_dict().keys()))
    print("After fetch Params")



    if is_available:
        P("Model exist")
        P("Loading Q params .....")
        P("Number Push: ", n_push)
        model_ft.load_state_dict(modman.convert_list_to_tensor(global_params))
        model_ft.eval()
       
    else:
        P("Setting model for server")
        lst_params = modman.convert_tensor_to_list(model_ft.state_dict())
        global_params, n_push, log_id, Iteration = modman.send_model_params(
            URL, lst_params, 0.001, ALIAS)
        del lst_params
        del global_params
        P("Number Push: ", n_push)
        
        # print(reply)
    n_steps=n_push
    
    print("Test model: ", model1.ALIAS)
    # torch.save(model_ft, model_path+model_name+".pt" )
    epoch =0
    while True:
        model_ft = model1.train_model(model_ft, dataloaders, dataset_sizes, device, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=n_push)
        torch.save(model_ft, model_path+model_name+".pt" )
        print("\n\nState_dict: *********************\n", model_ft.state_dict().keys())
        lst_params = modman.convert_tensor_to_list(model_ft.state_dict())
        reply = modman.send_local_update(URL + 'post_params',
                 lst_params,
                 epoch+1, ALIAS)
        print(reply, "\n Local update sent")
        epoch=epoch+n_push
        while modman.get_model_lock(URL):
            print("Waiting for Model Lock Release.")

        # Get Updated Model Params from Server
        global_params, n_push,_, is_available = modman.fetch_params(URL + 'get', list(model_ft.state_dict().keys()))
        model_ft.load_state_dict(modman.convert_list_to_tensor(global_params))
        model_ft.eval()
        del global_params
        if epoch>1000:
            print("Trained 1000 epochs")
            break;
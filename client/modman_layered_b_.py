import torch
import requests
from os import getpid
import csv
import glob
from pathlib import Path
import re
import numpy as np
debug = False
# Fetch Latest Model Params (StateDict)
def fetch_params(url: str, layers: list):
   
    # Send GET request
    params = {}
    npush=None
    logs_id = None
    for x in layers:
        body = {
        'layer_name': x,
        'pid': getpid()
        }
        r = requests.get(url=url, json=body)
        print("Reply", r)
        # Extract data in json format
        data = r.json()
        npush, logs_id = data['npush'], data['logs_id']
        # Check for Iteration Number (-1 Means, No model params is present on Server)
        if data['iteration'] == -1:
            return {}, data['npush'], data['logs_id'], False
        else:
            if debug:
                print("Global Iteration", data['iteration'])
            params[x]= data['lr_params']
        del data
    if (len(params.keys()) == len(layers)):        
        return params, npush, logs_id, True
    else:
        return {}, npush, logs_id, False
# remove send gradient method as we are not dealing with gradients in FL

# Send Trained Model Params (StateDict)

# Get Model Lock
def get_model_lock(url: str) -> bool:
    # Send GET request
    r = requests.get(url=url + 'getLock')

    # Extract data in json format
    data = r.json()
    print("Lock data:->", data['lock'])

    return data['lock'] 

def send_local_update(url: str, params: dict, bid: dict, train_count: int, model_id: str):
    
    keys = list(params.keys())
    values = list(params.values())
    # P(keys)
    data=None
    for i in range (len(keys)):
        dict_ = {}
        dict_[keys[i]] = values[i]
        # print(dict_.keys()) 
        body = {
        'model_id': model_id,
        'model': dict_,
        'bid': bid[keys[i]],
        'layer_length' : len(keys),
        'send_length' : i+1,
        'update_count' : train_count,
        'pid': getpid()
        }
            # Send POST request
        r = requests.post(url=url, json=body)
        del dict_
        # Extract data in json format
        data = r.json()
        print(data['Message'])
    return (["iteration ", data['iteration'], "No of clients perticipant in the updation", data['n_clients'], data['Message']])    
        

    


def send_model_params(url: str, params: dict, lr: float, model_id: str):
    keys = list(params.keys())
    values = list(params.values())
    # P(keys)
    data=None
    for i in range (len(keys)):
        dict_ = {}
        dict_[keys[i]] = values[i]
        print(dict_.keys()) 
        body = {
        'model_id': model_id,
        'model': dict_,
        'learning_rate': lr,
        'layer_length' : len(keys),
        'send_length' : i+1,
        'pid': getpid()
        }
            # Send POST request
        r = requests.post(url=url+'set', json=body)
        del dict_
        print("reply: ", r)
        # Extract data in json format
        data = r.json()
        print(data['Message'])
    return data #['n_push'], data['log_id'],     

# Convert State Dict List to Tensor
def convert_list_to_tensor(params: dict) -> dict:
    params_ = {}
    for key in params.keys():
        params_[key] = torch.tensor(params[key], dtype=torch.float32)

    return params_


# Convert State Dict Tensors to List
def convert_tensor_to_list(params: dict) -> dict:
    params_ = {}
    for key in params.keys():
        params_[key] = params[key].tolist()

    return params_

def csv_writer(path, data):
    f = open(path, 'a')

    # create the csv writer
    writer = csv.writer(f)

    # write a row to the csv file
    for x in data:
        writer.writerow(x)

    # close the file
    f.close()


def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        suffix = path.suffix
        path = path.with_suffix('')
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = Path(f"{path}{sep}{n}{suffix}")  # update path
    dir = path if path.suffix == '' else path.parent  # directory
    if not dir.exists() and mkdir:
        dir.mkdir(parents=True, exist_ok=True)  # make directory
    return path

def calculate_score(global_params: dict, local_params: dict):
    score=None
    keys = list(global_params.keys())
    values1 = list(global_params.values())
    values2 = list(local_params.values())
    bid={}
    print("length of keys", len(keys))
    for i in range (len(keys)):
        b_score=0.0
        list1 = np.array(values1[i].tolist())
        list2 = np.array(values2[i].tolist())
        shape1 = list1.shape
        shape2 = list2.shape
        if len(shape1)==len(shape2):
            # print("Length of list 1: ", len(shape1))
            # print(list1)
            # for j in range(len(shape1)):
            #     # print(list1[j].shape)
            #     for k in range(list1[j]):
            #         b_score +=abs(list1[j][k]-list2[j][k])
            #     print("bid score:", b_score)

            if(len(shape1)>1):
                for j in range(shape1[0]):
                    # print(list1[j].shape)
                    for k in range(len(list1[j])):
                        # print(j,k)
                        # print(list1[j][k],list2[j][k])
                        b_score +=abs(list1[j][k] - list2[j][k])
                    # print("bid score:", b_score)
            else:
                for j in range(shape1[0]):
                    b_score +=abs(list1[j] - list2[j])
                    # print("bid score:", b_score)
        bid[keys[i]]=b_score
    print("bids: ", bid)
    return bid
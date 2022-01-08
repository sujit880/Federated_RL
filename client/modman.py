import torch
import requests
from os import getpid

debug = False
# Fetch Latest Model Params (StateDict)
def fetch_params(url: str):
    body = {
        'pid': getpid()
    }
    # Send GET request
    r = requests.get(url=url, json=body)
    print("Reply", r)
    # Extract data in json format
    data = r.json()

    # Check for Iteration Number (-1 Means, No model params is present on Server)
    if data['iteration'] == -1:
        return {}, data['npush'], False
    else:
        if debug:
            print("Global Iteration", data['iteration'])
        return data['params'], data['npush'], True
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

def send_local_update(url: str, params: dict, train_count: int):
    body = {
        'model': params,
        'pid': getpid(),
        'update_count': train_count
    }

    # Send POST request
    r = requests.post(url=url, json=body)

    # Extract data in json format
    data = r.json()
    return data

def send_model_params(url: str, params: dict, lr: float):
    body = {
        'model': params,
        'learning_rate': lr,
        'pid': getpid()
    }

    # Send POST request
    r = requests.post(url=url+'set', json=body)

    # Extract data in json format
    data = r.json()

    return data

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

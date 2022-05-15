from pickle import GLOBAL
from flask import Flask, Response, jsonify, request, render_template
import modman
from datetime import datetime, timedelta
import json
import torch

# Client class to manage updates
class CParamas:
    client_key = None
    model_id = None
    epochs = None
    params= {}
    params_length = None
    received_length = 0
    # mem_size = None

class G_model:
    model_id = None
    model_length = None
    received_length = 0
    iteration = None
    steps = None
    epochs = None
    model= {}
    lr= None
    lock = None

# GLOBAL VARS
# CENTRAL_MODEL = {}
# LEARNING_RATE = 0.001
GLOBAL_MODEL = G_model()
COMPLETE = True
ITERATION = -1
ALL_PARAMS={}
SCORES={}
LEARNING_RATE = None
U_TIME_STAMP = None
WTS=30  # time in seconds to wait if not received the params from all the clients
N_PUSH = 100
N_CLIENTS = 1 
UPDATE_COUNT = 0
MODEL_NAME = 'experiment_01'
Log = True
log_path = './server/logs/'
log_file = 'server_logs.csv'
path = str(modman.increment_path(path=log_path+log_file,exist_ok=False,mkdir=True))
log_id= path[len(log_path)-2:-4] 
modman.csv_writer(path=path, data=[['Log for server']])
now = datetime.now
UMT=[] # Updating model time

    

# LOCK VAR
MODEL_LOCK = False
MODEL_COMPLETE = False 

#Start  Supporting functions
def register(add):
    global N_CLIENTS
    global Log
    print("key:-> ", add, " Registered")
    if Log:
        modman.csv_writer(path=path, data=[["Client with key:-> ", add, " Registered"]])
    SCORES[add] = 1
    N_CLIENTS = len(SCORES)
def add_score(add):
    if add in SCORES:
        return SCORES[add]
    else:
        register(add)
        return 1
def correctness(client_params):
    if SCORES[client_params.client_key] >0 and client_params.params != None:
        return True, [client_params.params, client_params.epochs]
    else:
        return False, []
def collect_params():
    global ALL_PARAMS
    global SCORES
    global N_CLIENTS
    all_params=[]
    keys=[]
    for x in ALL_PARAMS.values():
        valid, params = correctness(x)
        if valid:
            print("valid Params")
            SCORES[x.client_key] +=1
            all_params.append(params)
            keys.append(x.client_key)
        else:
            print("Invalid Params")
            SCORES[x.client_key] -=1
    akeys=SCORES.keys()
    dkeys=akeys-keys
    # print(dkeys)
    for key in dkeys:
        print("key:-> ", key, " Deleted")
        if Log:
            modman.csv_writer(path=path, data=[["Client with key:-> ", key, " Disconnected"]])
        del SCORES[key]
    N_CLIENTS = len(SCORES)
    return all_params
    
def update_model(list_of_params):
    global ITERATION
    global ALL_PARAMS
    
    # Update ITERATION
    ITERATION += 1

    # Apply Gradients and Update CENTRAL MODEL
    F_MODEL = modman.Federated_average(list_of_params)
    # # ALL_PARAMS = {}
    # print("Deleting all params after federated average")
    # dkeys=list(ALL_PARAMS.keys())
    # for key in dkeys:
    #     print("key:-> ", key, " Deleted")
    #     del ALL_PARAMS[key]
    # # PRINT RESPONSE
    # print('iteration :', ITERATION,' Updated Model Params.')
    return F_MODEL
    
#End Supporting functions

app = Flask(__name__)


@app.route("/")
def hello():
    return "Param Server"


@app.route('/api/model/get', methods=['GET'])
def get_model():
    print("inside get model")
    data = request.get_json()
    global GLOBAL_MODEL
    global LEARNING_RATE
    global ITERATION
    global N_PUSH
    global MODEL_NAME
    layer_name= data['layer_name']
    if layer_name not in GLOBAL_MODEL.model.keys():
        payload = {
        'lr_params': {},
        'npush': N_PUSH,
        'learning_rate': LEARNING_RATE,
        'iteration': -1,
        'model_name': MODEL_NAME,
        'logs_id':log_id
        }
        return jsonify(payload)
    else:
        print(f'Sending Global model to: Pid = {data["pid"]} : {request.remote_addr}')
        payload = {
            'lr_params': GLOBAL_MODEL.model[layer_name].tolist(),
            'npush': N_PUSH,
            'learning_rate': LEARNING_RATE,
            'iteration': ITERATION,
            'model_name': MODEL_NAME,
            'logs_id':log_id
        }

        return jsonify(payload)

@app.route('/api/model/getLock', methods=['GET'])
def get_lock():
    print("inside lock")
    global MODEL_LOCK
    global MODEL_COMPLETE
    payload = {
        'model_name': MODEL_NAME,
        'lock': MODEL_LOCK,
        'complete': MODEL_COMPLETE
    }

    return jsonify(payload)


@app.route('/api/model/setComplete', methods=['POST'])
def set_complete():
    global MODEL_COMPLETE
    MODEL_COMPLETE = True

    params = request.get_json()

    # Save Model
    with open(f'../server/models/{MODEL_NAME}.json', 'w') as f:
        json.dump(params['model'], f)

    return jsonify({'Message': 'Model Set To Complete.'})


@app.route('/api/model/set', methods=['POST'])
def set_model():
    print("inside set model")
    params = request.get_json()

    global GLOBAL_MODEL
    global ITERATION
    global LEARNING_RATE
    global COMPLETE
    print(
        f'Got Model Params from Client ID = {params["pid"]} IP Address = {request.remote_addr}, Iteration: {ITERATION}, Complete? {COMPLETE}')
    if ITERATION>0:
        return jsonify({'iteration': ITERATION, 'Message': 'Error Model Already exist.'})

    # Update ITERATION
    if COMPLETE:
        ITERATION += 1

    # Set CENTRAL MODEL params
    set_model = params['model']
    MODEL_LOCK=True
    LEARNING_RATE = params['learning_rate']
    if ITERATION == 0:
        for key, value in set_model.items():
            layer_name, Layer_count, total_length= key, params['send_length'], params['layer_length']
            print("set layer: ", layer_name, Layer_count, total_length)
            if(GLOBAL_MODEL.received_length == 0 and  Layer_count==1):
                COMPLETE = False
                GLOBAL_MODEL.model_id = params['model_id']
                GLOBAL_MODEL.model[layer_name]= torch.Tensor(value)
                GLOBAL_MODEL.model_length = total_length
                GLOBAL_MODEL.received_length += 1
                GLOBAL_MODEL.lr = LEARNING_RATE
                GLOBAL_MODEL.iteration = 0
                GLOBAL_MODEL.epochs = 0
                GLOBAL_MODEL.steps = N_PUSH
                GLOBAL_MODEL.lock = True
                return jsonify({'iteration': ITERATION, 'Message': 'first Layer set.'})
            if(GLOBAL_MODEL.received_length+1 ==  total_length and Layer_count == total_length):
                GLOBAL_MODEL.model[layer_name]= torch.Tensor(value)
                GLOBAL_MODEL.received_length = Layer_count
                GLOBAL_MODEL.lock = False
                COMPLETE = True

            elif(GLOBAL_MODEL.received_length+1 ==  Layer_count):
                GLOBAL_MODEL.model[layer_name]= torch.Tensor(value)
                GLOBAL_MODEL.received_length = Layer_count
            
            else:
                return jsonify({'iteration': -1, 'Message': 'Wrong chunck send aborted setup.'})



    MODEL_LOCK=False
    # RETURN RESPONSE
    return jsonify({'iteration': ITERATION, 'Message': 'Model Params layer Set.'})



@app.route('/api/model/post_params', methods=['POST'])
def post_params():
    print("inside post params")
    global ALL_PARAMS
    global GLOBAL_MODEL
    global U_TIME_STAMP #updating time stamp
    global WTS #waiting time stamp
    global N_CLIENTS
    global N_PUSH
    global UPDATE_COUNT
    global ITERATION
    global MODEL_LOCK
    global COMPLETE

    update_params = request.get_json()

    c_key=request.remote_addr+":"+str(update_params["pid"])
    print("\nadd score:->",add_score(c_key))
    if c_key in ALL_PARAMS.keys():
        c_params = CParamas()
        c_params.client_key=c_key
        c_params.epochs = update_params['update_count']
        c_params.model_id = update_params['model_id']
        model_chunck = update_params['model']
        ALL_PARAMS[c_key]=c_params
    c_params = ALL_PARAMS[c_key]
    for key, value in model_chunck.items():
        layer_name, Layer_count, total_length= key, update_params['layer_length'], update_params['send_length']
            
        if(c_params.received_length == 0 and  Layer_count==1):
            COMPLETE = False
            c_params.model_id = update_params['model_id']
            c_params.params[layer_name]= torch.Tensor(value)
            c_params.received_length += 1
            c_params.params_length = total_length
        if(c_params.received_length+1 ==  total_length and Layer_count == total_length):
            c_params.params[layer_name]= torch.Tensor(value)
            c_params.received_length = Layer_count
            COMPLETE = True

        elif(c_params.received_length+1 ==  Layer_count):
            c_params.params[layer_name]= torch.Tensor(value)
            c_params.received_length = Layer_count
            
        else:
            return jsonify({'iteration': -1, 'Message': 'Wrong chunck send aborted posting params.'})

    # c_params.mem_size=update_params['mem_size']
    # c_params.iteration=update_params['iteration']

    print(f'Got Parameters chunck {c_params.received_length}/ {c_params.params_length} from Client ID = {update_params["pid"]} IP Address = {request.remote_addr}')

    # Storing params
    ALL_PARAMS[c_key]=c_params
    # Set Global Update Count
    UPDATE_COUNT += update_params['update_count']

    if (len(ALL_PARAMS))==1:
        U_TIME_STAMP=datetime.now()+timedelta(seconds=WTS)

    # Execute Federated Averaging if Accumulated Params is full
    
    if len(ALL_PARAMS)==N_CLIENTS or U_TIME_STAMP<datetime.now() and COMPLETE:   # U_TIME_STAMP<datetime.now() or
        sumt= now() # start time of model updation 
        data=[]
        MODEL_LOCK = True
        print("Model lock...")
        print("Updating global model with clients params: ", len(ALL_PARAMS))
        list_of_params =   collect_params()
        if(len(list_of_params)>0): 
            data.append([f'\nUpdating global parameters with params from {len(list_of_params)} client.'])
            set_model = update_model(list_of_params=list_of_params)
            
            for key, value in set_model.items():
                GLOBAL_MODEL.model = torch.Tensor(value)
            
            # Empty Accumulated Params
            ALL_PARAMS={}
            print("Cleared All Params: ", len(ALL_PARAMS))
            # Save Model
            with open(f'./models/{MODEL_NAME}.json', 'w') as f:
                json.dump(modman.convert_tensor_to_list(CENTRAL_MODEL), f)
            # RETURN RESPONSE
            eumt = now()
            UMT.append(eumt-sumt)
            print('Updation time:->', eumt-sumt)
            data.append([f'Time taken for updation:-> {eumt-sumt}'])
            data.append([f'iteration: {ITERATION} Updated Global Model Params Complete.'])
            if Log:
                modman.csv_writer(path=path, data=data)
            MODEL_LOCK = False
            print("Release lock...")
            return jsonify({'iteration': ITERATION, 'n_clients':len(list_of_params), 'Message': 'Updated Global Model Params.'})
        else: 
            eumt = now()
            UMT.append(eumt-sumt)
            print('Updation time:->', eumt-sumt)
            print('Could not update the model due to receiving invalid params.')
            data.append(f'iteration: {ITERATION} Error! Global Model Params Updation Couldn\'t Complete.')
            if Log:
                modman.csv_writer(path=path, data=data)
            MODEL_LOCK = False
            print("Release lock...")
            return jsonify({'iteration': ITERATION, 'n_clients':len(list_of_params), 'Message': 'Could not update the model due to receiving invalid params.'})
        
    # RETURN RESPONSE
    return jsonify({'iteration': ITERATION,'n_clients':-1, 'Message': 'Collected Model Params.'})


if __name__ == "__main__":
    #app.run(debug=True, port=5500)

    # for listening to any network
    app.run(host="0.0.0.0", debug=False, port=5500)

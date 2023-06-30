import numpy as np
import os, time
import torch
import torch.nn as nn
import all_inputs
from architecture import *
from dataset import *
import optuna

######################################################
#INPUTS

sim = all_inputs.sim
n = all_inputs.n
num = all_inputs.num
prefix = all_inputs.prefix


seed = all_inputs.seed
f_prop = all_inputs.f_prop
f_prop_norm = all_inputs.f_prop_norm
f_params = all_inputs.f_params
features = all_inputs.features
batch_size = all_inputs.batch_size
num_workers = all_inputs.num_workers
g = all_inputs.g
h = all_inputs.h
fout = all_inputs.fout
fbesttrial = all_inputs.fbesttrial
f_off = all_inputs.f_off

######################################################

#testing the Optuna model

file = open(fbesttrial)
contents = file.readlines()
trial_num = int(contents[0])
parameters = {}
for i in range(1,len(contents)):
  a,b = str(contents[i]).split(' ')
  parameters.update({"{}".format(a) : float(b)})

print(f"Best trial is trial number : {trial_num}")
fmodel = '/mnt/ceph/users/cchawak/saved_models/%s_cosmo%dgal_%d_%d_%s.pt'%(sim,n,num,trial_num,prefix)

def test_model(parameters, device, g, h, fmodel, fout):

    # generate the architecture
    model = define_model(parameters).to(device)

    # load best-model, if it exists
    print('Loading model...')
    if os.path.exists(fmodel):  
        model.load_state_dict(torch.load(fmodel))

    else:  
        raise Exception('model doesnt exists!!!')

    # define the matrix containing the true & predicted value of the parameters + errors
    params      = len(g)
    results     = np.zeros((test_points, 3*params), dtype=np.float32)
    
    # test the model
    test_loss1, test_loss = torch.zeros(len(g)).to(device), 0.0
    test_loss2, points    = torch.zeros(len(g)).to(device), 0
    true_value = []
    model.eval()
    for x, y in test_loader:
        with torch.no_grad():
            bs    = x.shape[0]         #batch size
            x     = x.to(device)       #maps
            y     = y.to(device)[:,g]  #parameters
            p     = model(x)           #NN output
            y_NN  = p[:,g]             #posterior mean
            e_NN  = p[:,h]             #posterior std
            loss1 = torch.mean((y_NN - y)**2,                axis=0)
            loss2 = torch.mean(((y_NN - y)**2 - e_NN**2)**2, axis=0)
            loss  = torch.mean(torch.log(loss1) + torch.log(loss2))
            test_loss1 += loss1*bs
            test_loss2 += loss2*bs
            results[points:points+bs,0*params:1*params] = y.cpu().numpy()
            results[points:points+bs,1*params:2*params] = y_NN.cpu().numpy()
            results[points:points+bs,2*params:3*params] = e_NN.cpu().numpy()
            points     += bs
    test_loss = torch.log(test_loss1/points) + torch.log(test_loss2/points)
    test_loss = torch.mean(test_loss).item()
    print('Test loss:', test_loss)

    # denormalize results here
    minimum = np.array([0.1, 0.6, 0.25, 0.25, 0.5, 0.5])[g]
    maximum = np.array([0.5, 1.0, 4.00, 4.00, 2.0, 2.0])[g]
    results[:,0*params:1*params] = results[:,0*params:1*params]*(maximum-minimum)+minimum
    results[:,1*params:2*params] = results[:,1*params:2*params]*(maximum-minimum)+minimum
    results[:,2*params:3*params] = results[:,2*params:3*params]*(maximum-minimum)
    
    # save results to file
    np.savetxt(fout, results)




# get the data
test_loader = create_dataset('test', seed, f_prop, f_off, 
                                    f_prop_norm, f_params, features, 
                                    batch_size, shuffle=False, 
                                    num_workers=num_workers)

test_points = 0
for x,y in test_loader:  
  test_points += x.shape[0]

# use GPUs if available
if torch.cuda.is_available():
  print("CUDA Available")
  device = torch.device('cuda:0')
else:
  print('CUDA Not Available')
  device = torch.device('cpu')
  
test_model(parameters, device, g, h, fmodel, fout)
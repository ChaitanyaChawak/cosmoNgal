import optuna
import torch
import torch.nn as nn
import all_inputs
from architecture import *
from dataset import *
from time import time


class Objective(object):
    def __init__(self, max_layers, max_neurons_layers, device, epochs, batch_size):

      self.max_layers         = max_layers
      self.max_neurons_layers = max_neurons_layers
      self.device             = device
      self.epochs             = epochs
      self.batch_size         = batch_size

    def __call__(self, trial):


        # generate the architecture
        n_layers = trial.suggest_int("n_layers", 1 , self.max_layers)
        
        params = {'n_layers': n_layers}

        for i in range (n_layers):
            n_hidden = trial.suggest_int("n_hidden{}".format(i), 1, self.max_neurons_layers)
            dropoutprob = trial.suggest_float("dropoutprob_{}".format(i), 0, 1)
            params.update({"n_hidden{}".format(i) : n_hidden})
            params.update({"dropoutprob_{}".format(i) : dropoutprob})

        model = define_model(params).to(self.device)


        # get the weight decay and learning rate values
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        wd = trial.suggest_float("wd", 1e-8, 1e0,  log=True)

        # define the optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.5, 0.999), 
                                weight_decay=wd)

        # getting the train and val data
        train_loader = create_dataset('train', seed, f_prop, f_off, 
                                            f_prop_norm, f_params, features, 
                                            self.batch_size, shuffle=True, 
                                            num_workers=num_workers)
        valid_loader = create_dataset('valid', seed, f_prop, f_off, 
                                            f_prop_norm, f_params, features,
                                            self.batch_size, shuffle=False, 
                                            num_workers=num_workers)

        # train/validate model
        min_valid = 1e40
        for epoch in range(self.epochs):

            # do training
            train_loss1, train_loss = torch.zeros(len(g)).to(device), 0.0
            train_loss2, points     = torch.zeros(len(g)).to(device), 0
            model.train()
            for x, y in train_loader:
                bs   = x.shape[0]         #batch size
                x    = x.to(device)       #maps
                y    = y.to(device)[:,g]  #parameters
                p    = model(x)           #NN output
                y_NN = p[:,g]             #posterior mean
                e_NN = p[:,h]             #posterior std
                loss1 = torch.mean((y_NN - y)**2,                axis=0)
                loss2 = torch.mean(((y_NN - y)**2 - e_NN**2)**2, axis=0)
                loss  = torch.mean(torch.log(loss1) + torch.log(loss2))
                train_loss1 += loss1*bs
                train_loss2 += loss2*bs
                points      += bs
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            train_loss = torch.log(train_loss1/points) + torch.log(train_loss2/points)
            train_loss = torch.mean(train_loss).item()

            # do validation
            valid_loss1, valid_loss = torch.zeros(len(g)).to(device), 0.0
            valid_loss2, points     = torch.zeros(len(g)).to(device), 0
            model.eval()
            for x, y in valid_loader:
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
                    valid_loss1 += loss1*bs
                    valid_loss2 += loss2*bs
                    points     += bs
            valid_loss = torch.log(valid_loss1/points) + torch.log(valid_loss2/points)
            valid_loss = torch.mean(valid_loss).item()

            # save best model if found
            if valid_loss<min_valid:  
                min_valid = valid_loss
                torch.save(model.state_dict(), '/mnt/ceph/users/cchawak/saved_models/%s_cosmo%dgal_%d_%d_%s.pt'%(sim,n,num,trial.number,prefix)) 

        return min_valid


#############################################################
#INPUTS

n = all_inputs.n
num = all_inputs.num

sim = all_inputs.sim
f_off = all_inputs.f_off
f_prop = all_inputs.f_prop
prefix = all_inputs.prefix
f_params = all_inputs.f_params
f_prop_norm = all_inputs.f_prop_norm
seed = all_inputs.seed
num_workers = all_inputs.num_workers
fout = all_inputs.fout
fbesttrial = all_inputs.fbesttrial

max_layers = all_inputs.max_layers
max_neurons_layers = all_inputs.max_neurons_layers
features = all_inputs.features
input_size = all_inputs.input_size
output_size = all_inputs.output_size

batch_size = all_inputs.batch_size
epochs = all_inputs.epochs
g = all_inputs.g
h = all_inputs.h

study_name = all_inputs.study_name
n_trials = all_inputs.n_trials
n_startup_trials = all_inputs.n_startup_trials
storage = all_inputs.storage

#############################################################


# use GPUs if available
if torch.cuda.is_available():
  device = torch.device('cuda:0')  
  print("CUDA Available", torch.cuda.get_device_name())
else:
  print('CUDA Not Available')
  device = torch.device('cpu')


# using the Optuna magic
start_time = time()

# define the optuna study and optimize it
objective = Objective(max_layers, max_neurons_layers, device, epochs, batch_size)
sampler = optuna.samplers.TPESampler(n_startup_trials=n_startup_trials)
study = optuna.create_study(direction = "minimize", study_name=study_name, sampler=sampler, storage=storage, load_if_exists=True)
study.optimize(objective, n_trials)
  
print("Study statistics: ")

print("Best trial:")
trial = study.best_trial
trial_num = study.best_trial.number

print("  Value: ", trial.value)

print("  Params: ")
for key, value in trial.params.items():
  print("    {}: {}".format(key, value))

params = trial.params
model = define_model(params).to(device)
print(model)

f = open(fbesttrial, 'w+')
f.write('%d\n'%(study.best_trial.number))
for key, value in trial.params.items():
  f.write("{} {}\n".format(key, value))
f.close()

end_time = time() 

time_taken = end_time - start_time #this is in seconds
hours, rest =   divmod(time_taken, 3600)
minutes, seconds = divmod(rest, 60) 
print (f"\nTotal time taken is : {hours} hrs {minutes} mins {seconds} sec")
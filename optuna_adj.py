#### chi^2 adjustment hail mary

import optuna
import numpy as np
import all_inputs

######################################################
#INPUTS

fbesttrial = all_inputs.fbesttrial
study_name = all_inputs.study_name
storage = all_inputs.storage
num = all_inputs.num
prefix = all_inputs.prefix
sim = all_inputs.prefix
n = all_inputs.n

######################################################

study = optuna.load_study(study_name, storage)
trials = study.get_trials()
valuess = []

    

for i in range(len(trials)):
    if trials[i].value != None: valuess.append(trials[i].value)
    if trials[i].value == None: valuess.append(0)
sortval = np.argsort(valuess)
print(sortval)


# 0 is the best trial, 1 is the second-best and so on
chosen_num = 1

#to rewrite the besttrial file
trial_num = sortval[chosen_num]
trial = trials[trial_num]

print(f'Best: trial {sortval[0]}, currently using: trial {trial_num}')

params = trial.params
f = open(fbesttrial, 'w+')
f.write('%d\n'%(trial_num))

for key, value in trial.params.items():
  f.write("{} {}\n".format(key, value))
f.close()
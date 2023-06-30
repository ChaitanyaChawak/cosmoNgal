import os, sys
from create_Ngalaxy_data import create_galdata

##################################### INPUT ##########################################
      # [just change the values of n, num, sim, and optuna parameters]


# galaxy parameters
n = 10                   #number of galaxies to consider simultaneously (n>1)
num = 1500              #number of datapoints per simulation (combinations)

# data parameters
sim         = 'SIMBA'         # 'SIMBA' or 'IllustrisTNG' or 'Astrid_5.0e+08'
prefix = '5.0e+08_top5prop_OnlyOmegaM'

#f_off = '/mnt/ceph/users/cchawak/data/offset_%s_z=0.00.txt'%(sim)
f_off = '/mnt/ceph/users/cchawak/data/offset_%s_5.0e+08_z=0.00.txt'%(sim)


if n == 1: 
    f_prop = '/mnt/ceph/users/cchawak/data/galaxies_%s_5.0e+08_z=0.00.txt'%(sim)
else:
    f_prop = '/mnt/ceph/users/cchawak/data/%s_%dgal_%d_5.0e+08.txt'%(sim,n,num)
    if os.path.isfile(f_prop) == False: create_galdata(n,num,sim,f_off)
    

    
f_params = '/mnt/ceph/users/cchawak/data/latin_hypercube_params_%s.txt'%(sim)
f_prop_norm = None

seed        = 1
num_workers = 2
fout    = '/mnt/ceph/users/cchawak/results/Cosmo%dgal_%s_%d_%s_results.txt'%(n,sim,num,prefix)
fbesttrial = '/mnt/ceph/users/cchawak/results/besttrial/%s_cosmo%dgal_%d_%s_.txt'%(sim,n,num,prefix)

# architecture parameters
max_layers         = 5
max_neurons_layers = 1500

# features list
#|1. gas mass |2. stellar mass |3. black-hole mass |4. total mass |5. Vmax |6. velocity dispersion |7. gas metallicity |8. stars metallicity
#|9. star-formation rate |10. spin |11. peculiar velocity |12. stellar radius |13. total radius |14. Vmax radius |15. U |16. K |17. g

# top 5 features:
if sim == 'SIMBA': repeat, properties = (14, [1,4,7,11,13])
if sim == 'IllustrisTNG': repeat, properties = (17, [1,4,7,11,15])
if sim == 'Astrid_5.0e+08': repeat, properties = (14, [0,1,3,4,7])

# for all features
#if sim == 'SIMBA': repeat, properties = (14, list(range(14)))
#if sim == 'IllustrisTNG': repeat, properties = (17, list(range(17)))
#if sim == 'Astrid_5.0e+08': repeat, properties = (14, list(range(14)))

features = []
for i in range(n):
  newfeatures = [x+(i)*repeat for x in properties]
  features += newfeatures

print(f'Cosmology with {n} galaxies, {sim} suite. Total {len(features)} features considered: {features}')
    
    
input_size         = len(features)            #total number of subhalo properties
output_size        = 2                       #number of parameters to predict (posterior mean + std)

# training parameters
batch_size = 512
epochs     = 100
g          = [0]               #mean
h          = [1]             #error

# optuna parameters
study_name       = 'Cosmo%dgal_%s_%d_%s'%(n,sim,num,prefix)
n_trials         = 20        #set to None for infinite
n_startup_trials = 20        #random sample the space before using the sampler
storage          = 'sqlite:///cosmo%dgal_%s_%d_%s.db'%(n,sim,num,prefix)

######################################################################################

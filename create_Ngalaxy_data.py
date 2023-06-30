# to create new gal data
import numpy as np

def create_galdata(n, num, sim, f_off):
  onegal = np.loadtxt('/mnt/ceph/users/cchawak/data/galaxies_%s_z=0.00.txt'%(sim))
  off, length = np.loadtxt(f_off, dtype=np.int64, unpack=True)

  print(f"Creating the req dataset for {n} galaxies from the {sim} suite . . .")
  if sim == 'SIMBA': onegalfeat = list(range(14))
  if sim == 'IllustrisTNG': onegalfeat = list(range(17))
  if sim == 'Astrid_5.0e+08': onegalfeat = list(range(14))

  onegal = onegal[:, onegalfeat]

  tot = num*len(length)
  ngal = np.zeros((tot, n*len(onegalfeat)))
  galcount = 0
    
  Mstar = onegal[:, 1]
  choices = np.where(Mstar > 5.0e+08)
  choices = np.array(choices).tolist()[0]
  for i in range(0,len(length)):
    poss = list(range(off[i], off[i]+length[i]))
    choices_list = list(set(poss) & set(choices))
    for j in range(0,num):
      a = np.random.choice(choices_list, n, replace=False)
      for k in range(0,n):
        ngal[galcount, k*len(onegalfeat):(k+1)*len(onegalfeat)] = onegal[a[k], 0:len(onegalfeat)]
            
      galcount += 1

    if i%100 == 0:
      print(f'{i} done out of {len(length)}')

  #print(ngal[0])
  #print(galcount)
  np.savetxt('/mnt/ceph/users/cchawak/data/%s_%dgal_%d_5.0e+08.txt'%(sim,n,num), ngal)
  print("Dataset created successfully!")
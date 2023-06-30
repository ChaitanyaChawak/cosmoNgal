# to create new gal data
import numpy as np

n = 1
sim = 'IllustrisTNG'
f_old_off = '/mnt/ceph/users/cchawak/data/offset_%s_z=0.00.txt'%(sim)
f_new_off = '/mnt/ceph/users/cchawak/data/offset_%s_5.0e+08_z=0.00.txt'%(sim)


def create_1galdata_newcut(n, sim, f_old_off, f_new_off):
  onegal = np.loadtxt('/mnt/ceph/users/cchawak/data/galaxies_%s_z=0.00.txt'%(sim))
  off, length = np.loadtxt(f_old_off, dtype=np.int64, unpack=True)

  print(f"Creating the req dataset for {n} galaxies from the {sim} suite . . .")
  if sim == 'SIMBA': onegalfeat = list(range(14))
  if sim == 'IllustrisTNG': onegalfeat = list(range(17))
  if sim == 'Astrid_5.0e+08': onegalfeat = list(range(14))

  onegal = onegal[:, onegalfeat]

    
  Mstar = onegal[:, 1]
  choices = np.where(Mstar > 5.0e+08)
  choices = np.array(choices).tolist()[0]

  tot = len(choices)
  ngal = np.zeros((tot, n*len(onegalfeat)))
  galcount = 0
    
  f = open(f_new_off, 'a')
  f.write("{}".format(galcount))
  f.close()

  for i in range(0, len(length)):
    poss = list(range(off[i], off[i]+length[i]))
    choices_list = list(set(poss) & set(choices))
    gals_in_a_sim = 0
    for j in choices_list:
        ngal[galcount] = onegal[j]
        galcount += 1
        gals_in_a_sim += 1
    
    f = open(f_new_off, 'a')
    f.write(" {}\n{}".format(gals_in_a_sim, galcount))
    f.close()


    if i%100 == 0:
      print(f'{i} done out of {len(length)}')

  #print(ngal[0])
  #print(galcount)
  np.savetxt('/mnt/ceph/users/cchawak/data/galaxies_%s_5.0e+08_z=0.00.txt'%(sim), ngal)
  print("Dataset created successfully!")
    

create_1galdata_newcut(n, sim, f_old_off, f_new_off)
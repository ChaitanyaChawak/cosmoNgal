# results and the plots
import numpy as np
from sklearn.metrics import r2_score
from matplotlib.ticker import ScalarFormatter
import matplotlib.pyplot as plt
import all_inputs
plt.style.use("seaborn")

font = {'family': 'serif', 'weight': 'normal', 'size': 18}

# read the data
#n=2
#sim = 'Astrid_5.0e+08'
#num = 1500
n = all_inputs.n
sim = all_inputs.sim
num = all_inputs.num
prefix = all_inputs.prefix

if sim == 'Astrid_5.0e+08':
  simlabel = 'Astrid'
else:
  simlabel = sim

#fout    = '/content/drive/MyDrive/MS_project/results/Cosmo%dgal_%s_%d_results.txt'%(n,sim,num)
fout = all_inputs.fout
data = np.loadtxt(fout)

label = ['$\Omega_m$', '$\sigma_8$', '$A_{SN1}$', '$A_{AGN1}$', '$A_{SN2}$', '$A_{AGN2}$']
order = [1,4,2,3,5,6]             #order in which they appear in the final plot (left to right, top to bottom)

fig = plt.figure(figsize=(25,14))
plt.suptitle(f"{n} galaxies : {simlabel}", y=0.92, fontsize=28, fontfamily='serif')

for i in range(0,1):

  # get the unique values of Omega_m 
  Om_unique = np.unique(data[:,i])
  Om_unique = np.sort(Om_unique)

  # do a loop over all unique values of Omega_m
  #f = open(foutout, 'r')
  trueOm = []
  pred_meanOm = []
  error_meanOm = []
  for Om in Om_unique:

    # select the subhalos with that value of Omega_m
    indexes = np.where(data[:,i]==Om)[0]

    # compute mean Om and mean error
    #mean_Om  = np.mean(np.absolute(data[indexes,i+6]))
    #mean_dOm = np.mean(np.absolute(data[indexes,i+12]))
    mean_Om  = np.mean(np.absolute(data[indexes,i+1]))
    mean_dOm = np.mean(np.absolute(data[indexes,i+2]))

    # data in array format
    trueOm.append(Om)
    pred_meanOm.append(mean_Om)
    error_meanOm.append(mean_dOm)

  #adjusting A_AGN2 for Astrid
  if i==5 and sim=='Astrid_5.0e+08':
    trueOm = [item**2 for item in trueOm]
    pred_meanOm = [item**2 for item in pred_meanOm]
    error_meanOm = [2*pred_meanOm[error_meanOm.index(item)]*item for item in error_meanOm]
    
  #print(f'{label[i]} :')
  #print(f'actual values : {trueOm}')
  #print(f'pred mean values : {pred_meanOm}')
  #print(f'error values : {error_meanOm}')

  #accuracy
  acc = 0
  for k in range(0, len(data)):
    #acc += np.absolute((data[k,i] - data[k,i+6])**2)
    acc += np.absolute((data[k,i] - data[k,i+1])**2)
  acc /= len(data)
  #for k in range(0, len(trueOm)):
  #  acc += np.absolute((trueOm[k] - pred_meanOm[k])**2)
  #acc /= len(trueOm)

  acc = np.sqrt(acc)

  #chi2
  chi2 = 0
  for k in range(0, len(data)):
    #chi2 += np.absolute(((data[k,i] - data[k,i+6])**2)/data[k,i+12]**2)
    chi2 += np.absolute(((data[k,i] - data[k,i+1])**2)/data[k,i+2]**2)
  chi2 /= len(data)
  #for k in range(0, len(trueOm)):
  #  chi2 += np.absolute( ((trueOm[k] - pred_meanOm[k])**2) / (error_meanOm[k]**2))
  #chi2 /= len(trueOm)

  #r2

  #r2 = r2_score(data[:,i],data[:,i+6])
  r2 = r2_score(data[:,i],data[:,i+1])
  #r2 = r2_score(trueOm, pred_meanOm)


  #precision
  prec = 0
  for k in range(0, len(data)):
    #prec += (np.absolute(data[k,i+12])/data[k,i+6])
    prec += (np.absolute(data[k,i+2])/data[k,i+1])
  prec /= len(data)
  #for k in range(0, len(trueOm)):
  #  prec += (np.absolute(error_meanOm[k])/pred_meanOm[k])
  #prec /= len(trueOm)

  
  ax = fig.add_subplot(2,3,order[i])
  ax.set_title(label[i], x=0.5, y=0.9, fontsize=24)
  ax.scatter(trueOm, pred_meanOm)
  ax.plot(trueOm, trueOm, label=r'True value of $\Omega_m$')
  ax.errorbar(trueOm, pred_meanOm, yerr=error_meanOm, fmt='o', color='red', label=r'Predicted value of $\Omega_m$')
  ax.set_xlabel('True', fontdict=font)
  ax.set_ylabel('Prediction', fontdict=font)
  #plt.text(0.5, 0.1, f'Accuracy : {acc}\nChi^2 : {chi2}\nR^2 : {r2}', fontsize = 12, horizontalalignment='center', verticalalignment='center')
  plt.annotate(f'Accuracy : {acc:.3f}\nPrecision : {prec:.3f}\nChi^2 : {chi2:.3f}\nR^2 : {r2:.3f}', xy=(0.72, 0.02), xycoords='axes fraction', fontsize=14,
                horizontalalignment='left', verticalalignment='bottom', bbox=dict(facecolor='none', edgecolor='black', boxstyle='round'))
  ax.tick_params(axis='both', which='major', labelsize=14)


  if i==2 or i==3:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.minorticks_off()
    ax.set_xticks([0.25,0.5,1,2,4])
    ax.set_yticks([0.25,0.5,1,2,4])

  if i==4 or i==5:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.minorticks_off()
    ax.set_xticks([0.5,0.75,1,1.5,2])
    ax.set_yticks([0.5,0.75,1,1.5,2])

  if i==5 and sim=='Astrid_5.0e+08':
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.minorticks_off()
    ax.set_xticks([0.25,0.50,1,2,4])
    ax.set_yticks([0.25,0.50,1,2,4])
  


plt.savefig('/mnt/ceph/users/cchawak/results/plots/Cosmo%dgal_%s_%d_%s_results.png'%(n,sim,num,prefix), bbox_inches='tight')


#plt.savefig('/content/drive/MyDrive/MS_project/plots/Cosmo%dgal_%s_%d.png'%(n,sim,num), bbox_inches='tight')
#plt.savefig('/content/drive/MyDrive/MS_project/imgsformovie/image6_new.png', bbox_inches='tight')

##saving as pdf
#plt.savefig(f'/content/drive/MyDrive/MS_project/final_plots_pdf/Cosmo{n}gal_{simlabel}_{num}.pdf', bbox_inches='tight')

plt.show()
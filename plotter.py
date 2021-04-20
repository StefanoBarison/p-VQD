import numpy as np
import matplotlib.pyplot as plt
import json


# This is a simple script to plot the results obtained with the pVQD algorithm and compare
# them to the exact classical simulations

# In this case we plot the expectation value of \sigma_x and \sigma_z measured on the first spin
#==================================

exact  = json.load(open('data/exact_result_J0.25_B1.dat'))
data   = json.load(open('data/pVQD_J0.25_B1_800shots.dat'))


fig, ax = plt.subplots(2,1,sharex=True)

# Plot of Sx

ax[0].plot(exact['times'][:40],exact['Sx'][:40],label ="Exact",linestyle='dashed',linewidth=1.2,color='black')
ax[0].errorbar(data['times'][:40],data['Sx_0'][:40],yerr=data['err_Sx_0'][:40],label="pVQD",marker='o',linestyle='',elinewidth=1,color='C0',capsize=2,markersize=3.5)

ax[0].set(ylabel=r"$\langle\sigma_{x}\rangle_{1}$")
ax[0].set_ylim(ymax=1.1,ymin=-1.1)

# Plot of Sz

ax[1].plot(exact['times'][:40],exact['Sz'][:40],label ="Exact",linestyle='dashed',linewidth=1.2,color='black')
ax[1].errorbar(data['times'][:40],data['Sz_0'][:40],yerr=data['err_Sz_0'][:40],label="pVQD",marker='o',linestyle='',elinewidth=1,color='C0',capsize=2,markersize=3.5)
ax[1].set(ylabel=r"$\langle\sigma_{z}\rangle_{1}$",xlabel=r'$t$')
ax[1].set_ylim(ymax=1.1,ymin=-1.1)

# Legend above the plots
lines, labels = ax[0].get_legend_handles_labels()
ax[0].legend(lines , labels, loc='upper center', bbox_to_anchor=(0.5, 1.25),ncol=2, fancybox=True, shadow=False)


plt.show()
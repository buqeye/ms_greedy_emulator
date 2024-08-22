import call_tmatrix_SCh as CTm
import numpy as np
import matplotlib.pyplot as plt

name_of_plot = 'Phases_vs_Ecm_low_3pot.png'
s = 1
l = 4
ll = 4
j = 4
channel = 0
pot = 213


Ecms = np.logspace(-4, 2.2, 100)
#Ecms = np.arange(5.01, 10.0, 0.001)

mapp = 'trns'
np1 = 30
np2 = 20
number_of_points = np1+np2
p1 = 1.0  # fm^-1
p2 = 6.0  # fm^-1
p3 = 50.0  # fm^-1
c = 8.0
factor = 2/np.pi

phases, eta = CTm.Call_TMatrix(number_of_points, pot, s, l, ll, j, channel, c, Ecms, mapp, np1, np2, p1, p2, p3, factor)

#print(eta)
S = 2*s + 1
dict = {0:'S', 1:'P', 2:'D', 3:'F', 4:'G'}
L = dict[l]
import pandas as pd

# Opening the csv files generated from both python and c++. And, then plotting the data and difference in the data
file1 = pd.read_csv("./phaseShifts_3G4np.txt", sep='\s+',  header=None)

x1 = file1.iloc[:, 0]
y = file1.iloc[:, 1]

# plotting phases vs Ecm from above calculations
fig, axs = plt.subplots(1, 1, figsize=(12, 10))
#axs.plot(Ecms, phases, color='green', label='Phases from 3S1-Wave potential')
axs.plot(x1, y, color='blue', label=f'$^{S}{L}_{j}$ Dr. Drischler')
E_lab = 2*Ecms
axs.plot(E_lab, phases, color='red', label=f' $^{S}{L}_{j}$ phase shift')
axs.set_ylabel('phase shift (degrees) ')
axs.set_xlabel(r'E$_{lab}$ (MeV)')
axs.set_title('Phase Shift vs Energy')
axs.legend()
#axs.axhline(180)
#axs.axvline(0)
plt.grid()
plt.show()
fig.savefig(f'figure_{S}{L}_{j}.png')
#print(phases)


'''

# for plotting the fig side by side
fig, ax = plt.subplots(1, 1, figsize=(6, 4))


#ax.plot(x1, y, color='green', label="Dr. Drischler's" )
#ax.plot(x1, y, color='blue', label='Phase')
ax.set_xlabel(r'E_lab [MeV ] ')
ax.set_ylabel(r'Phases$_{^1S_0}$ [deg ]')
ax.legend()
ax.grid(True)


plt.show()
'''
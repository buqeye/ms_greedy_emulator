import Main_Tmatrix as MTm
import Coupled_phases as CP
import numpy as np
import matplotlib.pyplot as plt

from Config import Config

# the following function helps you with plotting epsilon vs Energy
# and delta plus and delta minus vs Energy

config = Config('Benchmark.ini')
config.print_info()


# Defining constants
h_barc = 197.327  # Mev fm
h_barc_sq = h_barc*h_barc
two_mu = 0.5*(939.56563+938.27205)/h_barc  # fm^-1


Ecms =np.logspace(-3,1.5,100)
mapp = config.mapp
np1 = config.np1
np2 = config.np2
p1 = config.p1
p2 = config.p2
p3 = config.p3
c = config.c
pot = config.pot
channel = config.channel
number_of_points = np1+np2
N = number_of_points
factor = 1 #2/np.pi
#s = 1
j = 1

phase_convention = CP.Stapp_parameterization_phase

T, k, eta_p, eta_m, delta_p, delta_m, epsi = MTm.Call_TMatrix(number_of_points, Ecms, mapp, np1, np2, p1, p2, p3, c, j, pot, channel, factor)

pointer = 6*0 + 2

m = int(0.40*len(k))
plt.plot(k[:-m], T[pointer][:-m].real)
plt.plot(k[:-m], T[pointer][:-m].imag)
plt.show()

'''
eta_p = []
eta_m = []
delta_p = []
delta_m = []
epsi = []


for i in range(len(Ecms)):
    k0 = np.sqrt(Ecms[i]*two_mu/h_barc)
    pointer = 6*i + 2
    if j != 0:
        eta_plus, eta_minus, delta_plus, delta_minus, epsilon = CP.Stapp_parameterization_phase(T2[pointer][-1], T2[pointer+1][-1], T2[pointer+2][-1], T2[pointer+3][-1], k0, factor)
        eta_p.append(eta_plus)
        eta_m.append(eta_minus)
        delta_p.append(delta_plus)
        delta_m.append(delta_minus)
        epsi.append(epsilon)
print(eta_p)

print()

print(eta_m)

'''

import pandas as pd

# Opening the csv files generated from both python and c++. And, then plotting the data and difference in the data
file1 = pd.read_csv("./phaseShifts_3S1-3D1np.txt", sep='\s+',  header=None)

x1 = file1.iloc[:, 0]
y1 = file1.iloc[:, 1]
y2 = file1.iloc[:, 2]
y3 = file1.iloc[:, 3]

# plotting phases vs Ecm from above calculations
fig1, axs = plt.subplots(1, 1, figsize=(12, 10))
#axs.plot(x1, y1, label="Dr. Drischler's phases")
E_lab = 2*Ecms
axs.plot(E_lab, delta_m, label='delta_m')
axs.set_xlabel("E_lab [MeV] ")
axs.set_ylabel("Phase shift [deg] ")
if phase_convention == CP.Stapp_parameterization_phase:
    axs.set_title(f" Stapp Parameterization phase plot, j = {j}")
else:
    axs.set_title(f" Blatt Biedenharn phase plot, j = {j}")
axs.legend()
plt.grid()
plt.show()
#fig1.savefig(f'Combined_phase_deltaminus_j_{j}')

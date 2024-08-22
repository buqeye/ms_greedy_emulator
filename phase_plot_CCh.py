#import Main_Tmatrix as MTm
import call_tmatrix_CCh as ctmCCh
import Coupled_phases as CP
import numpy as np
import matplotlib.pyplot as plt

from Config import Config

# the following function helps you with plotting epsilon vs Energy
# and delta plus and delta minus vs Energy

config = Config('Benchmark.ini')
config.print_info()



Ecms =np.logspace(-4,2,200)
#h_barc = config.h_barc
#two_mu = config.two_mu/h_barc
#j = config.j
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
factor = 2/np.pi
s = 1
j = 1


phase_convention = CP.Stapp_parameterization_phase

eta_p, eta_m, epsi_l, delta_p, delta_m = ctmCCh.call_coupled_tmatrix(N, mapp, p1, p2, p3, np1, np2, c, Ecms, phase_convention, s, j, pot, channel, factor)


import pandas as pd

# Opening the csv files generated from both python and c++. And, then plotting the data and difference in the data
file1 = pd.read_csv("./phaseShifts_3S1-3D1np.txt", sep='\s+',  header=None)

x1 = file1.iloc[:, 0]
y1 = file1.iloc[:, 1]
y2 = file1.iloc[:, 2]
y3 = file1.iloc[:, 3]

# plotting phases vs Ecm from above calculations
fig1, axs = plt.subplots(1, 1, figsize=(12, 10))
axs.plot(x1, y1, label="Dr. Drischler's phases")
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
fig1.savefig(f'Coupled_phase_deltaminus_j_{j}')


# plotting phases vs Ecm from above calculations
fig2, axs2 = plt.subplots(1, 1, figsize=(12, 10))
axs2.plot(x1, y2, label="Dr. Drischler's phases")
E_lab = 2*Ecms
axs2.plot(E_lab, delta_p, label='delta_p')
axs2.set_xlabel("E_lab [MeV]")
axs2.set_ylabel("Phase Shift [deg] ")
if phase_convention == CP.Stapp_parameterization_phase:
    axs2.set_title(f" Stapp Parameterization phase plot, j = {j}")
else:
    axs2.set_title(f" Blatt Biedenharn phase plot, j = {j}")
axs2.legend()
plt.grid()
plt.show()
fig2.savefig(f'Coupled_phase_deltaplus_j_{j}')


# plotting phases vs Ecm from above calculations
fig3, axs3 = plt.subplots(1, 1, figsize=(12, 10))
axs3.plot(x1, y3, label="Dr. Drischler's phases")
E_lab = 2*Ecms
axs3.plot(E_lab, epsi_l, label='epsilon')
axs3.set_xlabel("E_lab [MeV]")
axs3.set_ylabel("Epsilon [deg] ")
if phase_convention == CP.Stapp_parameterization_phase:
    axs3.set_title(f" Stapp Parameterization phase plot, j = {j}")
else:
    axs3.set_title(f" Blatt Biedenharn phase plot, j = {j}")
axs3.legend()
plt.grid()
plt.show()
fig3.savefig(f'Coupled_phase_epsilon_j_{j}')

print(eta_p)

print()

print(eta_m)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import chiralPot

# two_mu = 938.9182

# Defining constants
h_barc = 197.327  # Mev fm
h_barc_sq = h_barc*h_barc
two_mu = 0.5*(939.56563+938.27205)/h_barc  # fm^-1

# Opening the csv files generated from both python and c++. And, then plotting the data and difference in the data
file1 = pd.read_csv("./VNN_N2LO_fulllocal_R0_1.0_SLLJT_00001_lambda_50.00_Np_125_np.dat", sep='\s+',  header=None)
table = pd.DataFrame(file1)
inv_two_mu = 1/two_mu

l = 0
ll = 0
s = 0
j = 0
pot = 213
channel = 0

for m in range(120, 125):
    index_lower = 125*m
    index_upper = 125*(m+1)
    y1 = file1.iloc[index_lower:index_upper, 2]*inv_two_mu
    y = np.array(y1)

    x1 = np.array(file1.iloc[index_lower:index_upper, 1])
    V = np.zeros(len(y))
    diff = np.zeros(len(y))
    ki = file1.iloc[index_lower+1, 0]
    for i in range(len(y)):
        V[i] = chiralPot.V0(ki, x1[i], pot, s, l, ll, j, channel) * h_barc_sq
        diff[i] = abs(V[i]-y[i])

    # for plotting the fig side by side
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))


    #ax.plot(x1, y, color='green', label="Dr. Drischler's" )
    ax.plot(x1, diff, color='blue', label='difference')
    ax.set_xlabel(r'k [ fm$^{-1}$ ] ')
    ax.set_ylabel(r'V$_{^1S_0}$ [fm$^2$ ]')
    ax.set_title(f'slice {m}')
    ax.legend()
    ax.grid(True)
    plt.show()
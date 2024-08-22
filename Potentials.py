import numpy as np
from scipy.integrate import quad
from scipy.special import legendre

'''
All the potentials used for single channel T-matrix calculation are defined here, even for the 
phase plot and 2d plot for specific k-prime are defined here

For T-matrix and Contour plots: 
 Malfiet_Tjon(k, kp, costheta)
 SWave_V_s0i1(p, pprime, costheta)
 SWave_V_s1i0(p, pprime, costheta)
 Minnesota_Pot(k, kprime, costheta)

For V vs q plot where q = k-kprime: 
 V(q) ----> Malfiet-Tjon
 V_swave_3s1(q)
 V_swave_1s0(q)
 
 
To calculate integral over cosine-theta: 
 partial_potential2(fun, k, kp,l, roots,w )--> use this one if calculation is needed more than once
 partial_potential(fun, k, kp, l)

'''


# constants
V_A = 626.8932  # Mev fm
V_R = 1438.7228  # Mev fm
mu_A = 1.550  # fm^-1
mu_R = 3.11  # fm^-1

two_mu = 0.5*(939.56563+938.27205)


# constants
R0 = 1.54592984  # fm
R1 = 1.83039397  # fm
C01 = -0.68232207/32  # fm^2
C10 = -0.90747292/32  # fm^2
h_barc = 197.327053  # MeV fm

def V(q):
    out = V_R/(q**2 +mu_R**2) - V_A/(q**2 +mu_A**2)
    return 0.5*out/(np.pi**2)

def Malfiet_Tjon(k,kp,costheta):
    out1 = V_R/((k**2+kp**2-2*k*kp*costheta) + mu_R**2) - V_A/((k**2+kp**2-2*k*kp*costheta) + mu_A**2)
    return 0.5*out1/(np.pi**2)


# V(q) for sigma=0 and tau=1
def SWave_V_s0i1(p,pprime,costheta):
    q_sq = p**2 + pprime**2 - 2*p*pprime*costheta
    out = h_barc*C01*np.exp(-0.25*q_sq*(R1**2))
    return out

# V(q) for sigma=1 and tau=0
def SWave_V_s1i0(p,pprime,costheta):
    q_sq = p**2 + pprime**2 - 2*p*pprime*costheta
    out = h_barc*C10*np.exp(-0.25*q_sq*(R0**2))
    return out

def V_swave_3s1(k):
    out = h_barc * C10 * np.exp(-0.25 * (k**2 ** 2) * (R0 ** 2))
    return out

def V_swave_1s0(k):
    out = h_barc * C01 * np.exp(-0.25 * (k**2 ** 2) * (R1 ** 2))
    return out


def partial_potential2(fun, k, kp,l, roots,w ):
    f = lambda costheta: fun(k, kp, costheta) * 2 * np.pi * legendre(l)(costheta)
    return np.dot(f(roots), w)


def partial_potential(fun, k, kp, l):
    f = lambda costheta: fun(k, kp, costheta) * 2 * np.pi * legendre(l)(costheta)
    return quad(f, -1, 1)[0]


#  Defining Constants for Minnesota potentials
alpha = 200
beta = 222.015
V0 = 200.0/alpha  # MeV
V1 = -91.85/beta  # MeV
K0 = 1.487   # fm^-2
K1 = 0.465  # fm^-2

def Minnesota_Pot(k, kprime, costheta):
    q_sq = k*k + kprime*kprime - 2*k*kprime*costheta
    cube1 = (np.sqrt(np.pi/K0))**3
    cube2 = (np.sqrt(np.pi/K1))**3
    return (cube1*V0*np.exp(-0.25*q_sq/K0) + cube2*V1*np.exp(-0.25*q_sq/K1))/(0.5*two_mu)

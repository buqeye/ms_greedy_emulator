import numpy as np
import Map
import chiralPot

'''
    Coupled_Tmatrix(mapp, N, k0, np1, np2, p1, p2, p3, c)

    Parameters
    ----------
    mapp : string 
        decides which map to use ('trns' or 'gauleg') for Lippmann schwinger Equation

    N : integer
        total number of gauss-points to calculate the integral (N = np1 + np2) 
    np1, np2 : integer 
        total number of points to pass for trns map
        
    p1, p2, p3: float 
        three momentum points in fm^-1 needed for trns map
        
    c: float
        coefficient for tangent map, in fm^-1
    

    Returns
    -------
    floats: t00, t20, t02, t22, k
        half-shell t-matrix elements for t00, t20, t02, t22
        and k (momentum) used to calculate them
        
    '''



# Defining constants
h_barc = 197.327  # Mev fm
h_barc_sq = h_barc*h_barc
two_mu = (939.56563+938.27205)/h_barc  # fm^-1


def Coupled_Tmatrix(mapp, N, k0, np1, np2, p1, p2, p3, c, s1, s2, j1, j2, l, ll, pot, channel):
    # getting roots and weights for the Lippmann-Schwinger Integral
    kn, wn = Map.root_and_weight(mapp, p1, p2, p3, np1, np2, c)
    ki = np.array(kn)
    wi = np.array(wn)


    # getting propagator for all k_i's from i=1 to i=N
    propagator_inv = k0*k0-ki*ki
    propagator = 1/propagator_inv

    # calculating g_i's from i=1 to i=N
    g_n = two_mu*wi*propagator

    # calculating sum of g_i from i=1 to i=N:
    gnp1_val = np.sum(g_n)

    # getting g_i for i= n+1
    first_term = -gnp1_val * k0 * k0
    second_term = 0.5*k0 * two_mu * (np.log((ki[-1]+k0)/(ki[-1]-k0))-np.pi*1j)
    b = first_term + second_term

    # Calculating W_n from n= 1 to n=N
    W_n = g_n*ki*ki

    # getting W_n from n= 1 to n=N+1
    W = np.append(W_n, b)

    # getting k from n= 1 to n = N+1
    k = np.append(ki, k0)


    # preparing v00
    v00 = np.zeros((len(k), len(k)))
    one = np.identity(len(k))

    # preparing v20
    v20 = np.zeros((len(k), len(k)))

    # preparing v02
    v02 = np.zeros((len(k), len(k)))

    # preparing v22
    v22 = np.zeros((len(k), len(k)))

    # Filling up v_ll' matrix and a_ll' matrix
    for i in range(len(k)):
        for p in range(i, len(k)):
            v00[i, p] = chiralPot.V0(k[i], k[p], pot, s1, l, ll, j1, channel) * h_barc_sq
            v22[i, p] = chiralPot.V0(k[i], k[p], pot, s2, l, ll, j2, channel) * h_barc_sq
            v00[p, i] = v00[i, p]
            v22[p, i] = v22[i, p]

    a00 = -np.einsum('j, ij->ij', W, v00)
    a00 = one + a00
    a22 = -np.einsum('j, ij->ij', W, v22)
    a22 = one + a22
    a20 = -np.einsum('j, ij->ij', W, v20)
    a02 = -np.einsum('j, ij->ij', W, v02)

    # getting A matrix
    A = np.block([[a00, a02], [a20, a22]])


    v1 = v00[:, -1]
    v2 = v20[:, -1]

    v3 = v02[:, -1]
    v4 = v22[:, -1]

    # getting first column of V_matrix
    Vmatrix1 = np.append(v1, v2)

    # getting second Column of V_matrix
    Vmatrix2 = np.append(v3, v4)

    # solving for first column of  T_matrix
    t1 = np.linalg.solve(A, Vmatrix1)

    # solving for second column of T_matrix
    t2 = np.linalg.solve(A, Vmatrix2)

    t00 = t1[0:N+1]
    t20 = t1[N+1:]

    t02 = t2[0:N + 1]
    t22 = t2[N+1:]

    return t00, t20, t02, t22, k

import numpy as np
import Map
import time
import chiralPot

'''
    Coupled_Tmatrix(k0, ki, wi, one, v00, v02, v20, v22, s, j, pot, channel, factor):

    Parameters

    k0 : real 
        k_0 value corresponding to E_0 
        
    ki, wi: real 
        roots and weights for the Lippmann-Schwinger integration 

    pot : integer 
        'xyz' --> x 'order' --> 0- LO, 1- NLO, 2 - N2LO
              --> y ' cutoff' --> 0- 0.8, 1- 1.0, 2- 1.2, 3- 0.9, 4- 1.1
              --> z 'SFR cutoff' --> 2- 800 , 3- 1000, 4- 1200, 5- 1400

    s, j : integer
        They define partial wave 

    channel: integer 
        nn -> -1, np -> 0, pp -> 1

    ki, wi : real
        roots and weight for the Lippmann-Schwinger equation

    ki_sq : real
        array of ki square values

    one : real array
        N+1 by N+1 array of Identity Matrix

    v00, v02, v20, v22: real array
        N+1 by N+1 array of V_matrix with N-by-N block already filled 
        and N+1 column and N+1 row need to be filled 

    factor : real
        2/pi or 1.0 factor for the principal value integration

    Returns
    -------
    floats: T
        T-matrix elements for specified 'fun'

    This function takes N-by-N filled potential, and fills the remaining elements for 
    corresponding energy and returns T matrix for that energy
    '''

'''
    Call_TMatrix(number_of_points, pot, s, l, ll, j, channel, c, Ecms, mapp, np1, np2, p1, p2, p3, factor)
    call_coupled_tmatrix(N, mapp, p1, p2, p3, np1, np2, c, E_range, phase_convention, s, j, pot, channel, factor)
    Parameters
    ----------  
    N : integer
        total number of gauss-points to calculate the integral ( = np1 + np2) 
        
    mapp : string 
        decides which map to use ('trns' or 'gauleg') for Lippmann schwinger Equation
        
    p1, p2, p3: float 
        three momentum points in fm^-1 needed for trns map
        
    np1, np2 : integer 
        total number of points to pass for trns map
        
    c : real 
        factor for the tangent map
        
    Erange: real array
        array of energy in CM frame 
    
    Phase_convention: function
        "CP.Stapp_parameterization_phase" or "CP.Blatt_Biedenharn_phase" to get phase shifts
        
    s,  j : integer
        They define partial wave 

    pot : integer 
        'xyz' --> x 'order' --> 0- LO, 1- NLO, 2 - N2LO
              --> y ' cutoff' --> 0- 0.8, 1- 1.0, 2- 1.2, 3- 0.9, 4- 1.1
              --> z 'SFR cutoff' --> 2- 800 , 3- 1000, 4- 1200, 5- 1400

    channel: integer 
        nn -> -1, np -> 0, pp -> 1

    factor : real
        2/pi or 1.0 factor for the principal value integration

    Returns
    -------
    floats:  eta_p, eta_m, epsi_l, delta_p, delta_m
        returns arrays of phases and etas for respective energy in center of mass (Ecms) 

    '''


# Defining constants
h_barc = 197.327  # Mev fm
h_barc_sq = h_barc*h_barc
two_mu = 0.5*(939.56563+938.27205)/h_barc  # fm^-1

def Coupled_Tmatrix(k0, ki, wi, one, v00, v02, v20, v22, s, j, pot, channel, factor):
    N = len(ki)
    # getting propagator for all k_i's from i=1 to i=N
    propagator_inv = k0*k0-ki*ki
    propagator = 1/propagator_inv


    # calculating g_i's from i=1 to i=N
    g_n = two_mu*wi*propagator

    # calculating sum of g_i from i=1 to i=N
    gnp1_val = np.sum(g_n)

    # getting g_i for i= n+1
    first_term = -gnp1_val * k0 * k0
    second_term = 0.5*k0 * two_mu * (np.log((ki[-1]+k0)/(ki[-1]-k0))-np.pi*1j)
    b = first_term + second_term

    # Calculating W_n from n= 1 to n=N
    W_n = g_n*ki*ki

    # getting W_n from n= 1 to n=N+1
    W = np.append(W_n, b)*factor

    # getting k from n= 1 to n = N+1
    k = np.append(ki, k0)


    # Filling up v_ll' matrix and a_ll' matrix
    for i in range(len(k)):
        v00[i, N] = chiralPot.V0(k[i], k[N], pot, s, j - 1, j - 1, j, channel) * h_barc_sq
        v02[i, N] = -chiralPot.V0(k[i], k[N], pot, s, j - 1, j + 1, j, channel) * h_barc_sq
        v20[N, i] = v02[i, N]
        v22[i, N] = chiralPot.V0(k[i], k[N], pot, s, j+1, j+1, j, channel) * h_barc_sq

        v20[i, N] = -chiralPot.V0(k[i], k[N], pot, s, j + 1, j - 1, j, channel) * h_barc_sq
        v00[N, i] = v00[i, N]
        v02[N, i] = v20[i, N]
        v22[N, i] = v22[i, N]

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
    #Vmatrix1 = np.block([[v00, v02], [v20, v22]])

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
    #t00 = t[0:N+1, N]
    #t20 = t[N+1:, N]
    #t02 = t[0:N+1, 2*N + 1]
    #t22 = t[N+1:, 2*N + 1]


    return t00, t20, t02, t22

def call_coupled_tmatrix(N, mapp, p1, p2, p3, np1, np2, c, E_range, phase_convention, s, j, pot, channel, factor):
    # getting roots and weights for the Lippmann-Schwinger Integral
    kn, wn = Map.root_and_weight(mapp, p1, p2, p3, np1, np2, c)
    ki = np.array(kn)
    wi = np.array(wn)

    # preparing v00
    v00 = np.zeros((N+1, N+1))
    one = np.identity(N+1)

    # preparing v20
    v20 = np.zeros((N+1, N+1))

    # preparing v02
    v02 = np.zeros((N+1, N+1))

    # preparing v22
    v22 = np.zeros((N+1, N+1))

    # Filling up v_ll' matrix and a_ll' matrix
    for i in range(len(ki)):
        for p in range(i, len(ki)):
            v00[i, p] = chiralPot.V0(ki[i], ki[p], pot, s, j - 1, j - 1, j, channel) * h_barc_sq
            v02[i, p] = -chiralPot.V0(ki[i], ki[p], pot, s, j - 1, j + 1, j, channel) * h_barc_sq
            v20[p, i] = v02[i, p]
            v22[i, p] = chiralPot.V0(ki[i], ki[p], pot, s, j + 1, j + 1, j, channel) * h_barc_sq

            v20[i, p] = -chiralPot.V0(ki[i], ki[p], pot, s, j + 1, j - 1, j, channel) * h_barc_sq
            v00[p, i] = v00[i, p]
            v02[p, i] = v20[i, p]
            v22[p, i] = v22[i, p]

    epsi_l = []
    delta_p = []
    delta_m = []
    eta_p = []
    eta_m = []

    for index in range(len(E_range)):
        E = E_range[index]/h_barc
        knp1 = np.sqrt(two_mu * E)
        time1 = time.time()
        t00, t20, t02, t22 = Coupled_Tmatrix(knp1, ki, wi, one, v00, v02, v20, v22,s, j, pot, channel, factor)
        time2 = time.time()
        print(f' elaspsed time :  {time2-time1:.6f}  s')
        eta_plus, eta_minus, delta_plus, delta_minus, epsi = phase_convention(t00[-1], t20[-1], t02[-1], t22[-1], knp1, factor)
        eta_p.append(eta_plus)
        eta_m.append(eta_minus)
        epsi_l.append(abs(epsi)*180/np.pi)
        #if delta_plus <= 0:
        #    delta_plus = 180 + delta_plus

        delta_p.append(delta_plus)
        if delta_minus <= 0:
            delta_minus = 180 + delta_minus

        delta_m.append(delta_minus)
    return eta_p, eta_m, epsi_l, delta_p, delta_m
import numpy as np
import chiralPot
import Map

'''
    T_matrix(N, knp1, pot, s, l, ll, j,  channel, ki, wi, ki_sq, Identity_matrix, V_matrix, factor):

    Parameters

    N : integer
        total number of gauss-points to calculate the integral (N = np1 + np2) 
        
    knp1 : real 
        k_0 value corresponding to E_0 
        
    pot : integer 
        'xyz' --> x 'order' --> 0- LO, 1- NLO, 2 - N2LO
              --> y ' cutoff' --> 0- 0.8, 1- 1.0, 2- 1.2, 3- 0.9, 4- 1.1
              --> z 'SFR cutoff' --> 2- 800 , 3- 1000, 4- 1200, 5- 1400
        
         
    s, l, ll, j : integer
        They define partial wave 
        
    channel: integer 
        nn -> -1, np -> 0, pp -> 1
    
    ki, wi : real
        roots and weight for the Lippmann-Schwinger equation
    
    ki_sq : real
        array of ki square values

    Identity_matrix : real array
        N+1 by N+1 array of Identity Matrix
        
    V_matrix : real array
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

    Parameters
    ----------  
    number_of_points : integer
        total number of gauss-points to calculate the integral ( = np1 + np2) 
        
    pot : integer 
        'xyz' --> x 'order' --> 0- LO, 1- NLO, 2 - N2LO
              --> y ' cutoff' --> 0- 0.8, 1- 1.0, 2- 1.2, 3- 0.9, 4- 1.1
              --> z 'SFR cutoff' --> 2- 800 , 3- 1000, 4- 1200, 5- 1400
        
         
    s, l, ll, j : integer
        They define partial wave 
        
    channel: integer 
        nn -> -1, np -> 0, pp -> 1
    
    c : real 
        factor for the tangent map
        
    Ecms: real array
        array of energy in CM frame 

    mapp : string 
        decides which map to use ('trns' or 'gauleg') for Lippmann schwinger Equation


    np1, np2 : integer 
        total number of points to pass for trns map

    p1, p2, p3: float 
        three momentum points in fm^-1 needed for trns map
        
    factor : real
        2/pi or 1.0 factor for the principal value integration

    Returns
    -------
    floats: phases, etas
        returns arrays of phases and etas for respective energy in center of mass (Ecms) 

    '''


# Defining constants
h_barc = 197.327  # Mev fm
h_barc_sq = h_barc*h_barc
two_mu = 0.5*(939.56563+938.27205)/h_barc  # fm^-1

def T_matrix(N, knp1, pot, s, l, ll, j,  channel, ki, wi, ki_sq, Identity_matrix, V_matrix, factor):
    # propagator from i=1 ti i = N
    prop = 1 / (knp1 * knp1 - ki_sq)

    # g_i from i = 1 to i = N
    g_i = two_mu * wi * prop

    # W_i from i = 1 to i = N
    W_i = g_i * ki_sq

    # W_(N+1)
    Wnp1 = -np.sum(g_i) * knp1 * knp1 + 0.5 * knp1 * two_mu * (
            np.log((ki[- 1] + knp1) / (ki[- 1] - knp1)) - np.pi * 1j)

    # array of W_i and k_i from i = 1 to i = N+1
    W = np.append(W_i, Wnp1)*factor
    k = np.append(ki, knp1)

    for i in range(N):
        # filling up the last (N+1) column
        V_matrix[i, -1] = chiralPot.V0(k[i], knp1, pot, s, l, ll, j, channel)*h_barc_sq

        # filling up the last (N+1) row
        V_matrix[-1, i] = chiralPot.V0(knp1, k[i], pot, s, l, ll, j, channel)*h_barc_sq

    # filling up the last (N+1, N+1) element of the matrix
    V_matrix[N, N] = chiralPot.V0(knp1, knp1, pot, s, l, ll, j, channel)*h_barc_sq

    GV_matrix = np.einsum('j,ij->ij', W, V_matrix)


    # calculating 1-GV matrix
    A = Identity_matrix - GV_matrix

    T = np.linalg.solve(A, V_matrix)

    return T

def Call_TMatrix(number_of_points, pot, s, l, ll, j, channel, c, Ecms, mapp, np1, np2, p1, p2, p3, factor):

    number_Ecm = len(Ecms)

    # Initializing phases and etas array
    #T_halfshell = np.zeros(number_of_points)
    phases = np.zeros(number_Ecm)
    etas = np.zeros(number_Ecm)

    # roots and weight for Lippmann-Schwinger equation
    kn, wn = Map.root_and_weight(mapp, p1, p2, p3, np1, np2, c)

    ki = np.array(kn)
    wi = np.array(wn)

    # calculating the W_i from i=1 to i = N+1
    ki_sq = ki*ki

    # Identity Matrix for 1 - VG part
    I = np.identity(number_of_points + 1)

    # Initializing Potential Matrix
    V = np.zeros((number_of_points + 1, number_of_points + 1))

    # this loop helps to fill up the N-by-N block part of the potential matrix
    for i in range(number_of_points):
        for p in range(i, number_of_points, 1):
            if i == p:
                V[i, i] = chiralPot.V0(ki[i], ki[i], pot, s, l, ll, j, channel)*h_barc_sq
            else:
                V[i, p] = chiralPot.V0(ki[i], ki[p], pot, s, l, ll, j, channel)*h_barc_sq
                V[p, i] = chiralPot.V0(ki[p], ki[i], pot, s, l, ll, j, channel)*h_barc_sq

    # set up file and write to file
    #file1 = open("./Phases_single1S0.csv", "w")

    for index in range(number_Ecm):

        # choosing energy in CM frame
        Ecm = Ecms[index]/h_barc  # MeV

        # calculating pole
        k_np1 = np.sqrt(two_mu * Ecm)   # in fm^-1

        # getting T-Matrix for the above k_np1
        Tmatrix = T_matrix(number_of_points, k_np1, pot, s, l, ll, j, channel, ki, wi, ki_sq, I, V, factor)

        # getting on-shell elements of the T-Matrix ( Real and Imaginary part)
        real = Tmatrix[number_of_points, number_of_points].real
        imag = Tmatrix[number_of_points, number_of_points].imag

        # Calculating Phase Shift using On-Shell element
        y = -factor*0.5*np.pi*k_np1*two_mu*real
        x1 = -factor*0.5*np.pi*k_np1*two_mu*imag
        x = (0.5-x1)
        phase = 90*np.arctan2(y, x)/np.pi

        # Calculation for etas
        sq_term = y*y + x1*x1
        brac = sq_term-x1
        etas[index] = np.sqrt(1+4*brac)
        print(etas[index])

        #if phase <= 0:

        #    phases[index] = 180+phase
        #else:
        phases[index] = phase

        #file1.write("%.15f, %.15f\n" % (Ecms[index], phases[index]))

    # close the file1
    #file1.close()

    return phases, etas
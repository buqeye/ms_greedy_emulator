import numpy as np
from numpy import linalg
import Coupled_phases as CP
import chiralPot
import Map

# Defining constants
h_barc = 197.327  # Mev fm
h_barc_sq = h_barc*h_barc
two_mu = 0.5*(939.56563+938.27205)/h_barc  # fm^-1

def k_and_W(ki, wi, ki_sq, knp1, factor):
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

    return k, W

def ST_matrix(N, s, l, j, channel, pot, k, W, one, V_matrix):
    for i in range(N+1):
        # filling up the last (N+1) column
        V_matrix[i, N] = chiralPot.V0(k[i], k[-1],  pot, s, l, l, j, channel)*h_barc_sq

        # filling up the last (N+1) row
        V_matrix[N, i] = V_matrix[i, N]
    GV_matrix = np.einsum('j,ij->ij', W, V_matrix)

    # calculating 1-GV matrix
    A = one - GV_matrix

    T = np.linalg.solve(A, V_matrix)

    return T

def CT_matrix(N, j, channel, pot, k, W, one, vmm, vpm, vmp, vpp):
    # Filling up v_ll' matrix and a_ll' matrix
    jm1 = j - 1
    jp1 = j + 1

    for i in range(N):
        vmm[i, -1] = chiralPot.V0(k[i], k[-1],  pot, 1, jm1, jm1, j, channel)*h_barc_sq
        vpm[i, -1] = -chiralPot.V0(k[i], k[-1],  pot, 1, jp1, jm1, j, channel)*h_barc_sq
        vmp[-1, i] = vpm[i, -1]
        vpp[i, -1] = chiralPot.V0(k[i], k[-1],  pot, 1, jp1, jp1, j, channel)*h_barc_sq

        vmm[-1, i] = vmm[i, -1]
        vpm[-1, i] = -chiralPot.V0(k[-1], k[i], pot, 1, jp1, jm1, j, channel)*h_barc_sq
        vmp[i, -1] = vpm[-1, i]
        vpp[-1, i] = vpp[i, -1]

    vmm[-1, -1] = chiralPot.V0(k[-1], k[-1], pot, 1, jm1, jm1, j, channel) * h_barc_sq
    vpm[-1, -1] = -chiralPot.V0(k[-1], k[-1], pot, 1, jp1, jm1, j, channel) * h_barc_sq
    vmp[-1, -1] = vpm[-1, -1]
    vpp[-1, -1] = chiralPot.V0(k[-1], k[-1], pot, 1, jp1, jp1, j, channel) * h_barc_sq

    a00 = -np.einsum('j, ij->ij', W, vmm)
    a00 = one + a00
    a22 = -np.einsum('j, ij->ij', W, vpp)
    a22 = one + a22
    a20 = -np.einsum('j, ij->ij', W, vpm)
    a02 = -np.einsum('j, ij->ij', W, vmp)

    # getting A matrix
    A = np.block([[a00, a02], [a20, a22]])

    v1 = vmm[:, -1]
    v2 = vpm[:, -1]

    v3 = vmp[:, -1]
    v4 = vpp[:, -1]

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

    return t00, t20, t02, t22

def Call_TMatrix(number_of_points, Ecms, mapp, np1, np2, p1, p2, p3, c, j, pot, channel, factor):

    number_Ecm = len(Ecms)
    T = []

    # roots and weight for Lippmann-Schwinger equation
    kn, wn = Map.root_and_weight(mapp, p1, p2, p3, np1, np2, c)

    ki = np.array(kn)
    wi = np.array(wn)

    # calculating the W_i from i=1 to i = N+1
    ki_sq = ki*ki

    # Identity Matrix for 1 - VG part
    I = np.identity(number_of_points + 1)
    eta_p = []
    eta_m = []
    delta_p = []
    delta_m = []
    epsi = []

    if j == 0:
        l1 = j
        l2 = j + 1
        t_mm, t_pm, t_mp, t_pp = 0, 0, 0, 0

        # Initializing Potential Matrix
        V1 = np.zeros((number_of_points + 1, number_of_points + 1))
        V2 = np.zeros((number_of_points + 1, number_of_points + 1))


        # Filling up N-by-N block of v_ll'
        for i in range(len(ki)):
            for p in range(i, len(ki)):
                if i == p:
                    V1[i, p] = chiralPot.V0(ki[i], ki[p], pot, 0, l1, l1, j, channel) * h_barc_sq
                    V2[i, p] = chiralPot.V0(ki[i], ki[p], pot, 1, l2, l2, j, channel) * h_barc_sq
                else:
                    V1[i, p] = chiralPot.V0(ki[i], ki[p], pot, 0, l1, l1, j, channel)*h_barc_sq
                    V1[p, i] = V1[i, p]
                    V2[i, p] = chiralPot.V0(ki[i], ki[p], pot, 1, l2, l2, j, channel)*h_barc_sq
                    V2[p, i] = V2[i, p]


        for index in range(number_Ecm):
            # choosing energy in CM frame
            Ecm = Ecms[index]/h_barc # MeV

            # calculating pole
            k_np1 = np.sqrt(two_mu * Ecm)   # in fm^-1

            # getting k, W
            k, W = k_and_W(ki, wi, ki_sq, k_np1, factor)

            # getting T-Matrix for the above k_np1
            T1 = ST_matrix(number_of_points, 0, l1, j, channel, pot, k, W, I, V1)
            T.append(T1)
            T2 = ST_matrix(number_of_points, 1, l2, j, channel, pot, k, W, I, V2)
            T.append(T2)
            T.append(t_mm)
            T.append(t_pm)
            T.append(t_mp)
            T.append(t_pp)

    else:
        l1 = j
        jm1 = j - 1
        jp1 = j + 1

        # preparing v--
        v_mm = np.zeros((number_of_points + 1, number_of_points + 1))

        # preparing v+-
        v_pm = np.zeros((number_of_points + 1, number_of_points + 1))

        # preparing v-+
        v_mp = np.zeros((number_of_points + 1, number_of_points + 1))

        # preparing v++
        v_pp = np.zeros((number_of_points + 1, number_of_points + 1))

        # Initializing Potential Matrix
        V1 = np.zeros((number_of_points + 1, number_of_points + 1))
        V2 = np.zeros((number_of_points + 1, number_of_points + 1))

        # Filling up N-by-N block of v_ll'
        for i in range(len(ki)):
            for p in range(i, len(ki)):
                V1[i, p] = chiralPot.V0(ki[i], ki[p], pot, 0, l1, l1, j, channel)*h_barc_sq
                V1[p, i] = V1[i, p]

                V2[i, p] = chiralPot.V0(ki[i], ki[p], pot, 1, jm1, jm1, j, channel)*h_barc_sq
                V2[p, i] = V2[i, p]

                v_mm[i, p] = V2[i, p]
                v_pm[i, p] = -chiralPot.V0(ki[i], ki[p], pot, 1, jp1, jm1, j, channel)*h_barc_sq
                v_mp[p, i] = v_pm[i, p]
                v_pp[i, p] = chiralPot.V0(ki[i], ki[p], pot, 1, jp1, jp1, j, channel)*h_barc_sq

                v_mm[p, i] = v_mm[i, p]
                v_pm[p, i] = -chiralPot.V0(ki[p], ki[i], pot, 1, jp1, jm1, j, channel)*h_barc_sq
                v_mp[i, p] = v_pm[p, i]
                v_pp[p, i] = v_pp[i, p]

        for index in range(number_Ecm):
            # choosing energy in CM frame
            Ecm = Ecms[index]/h_barc  # MeV

            # calculating pole
            k_np1 = np.sqrt(two_mu * Ecm)  # in fm^-1

            # getting k, W
            k, W = k_and_W(ki, wi, ki_sq, k_np1, factor)

            # getting T-Matrix for the above k_np1
            T1 = ST_matrix(number_of_points, 0, l1, j, channel, pot, k, W, I, V1)
            T.append(T1)
            T2 = ST_matrix(number_of_points, 1, jm1, j, channel, pot, k, W, I, V2)
            T.append(T2)
            t_mm, t_pm, t_mp, t_pp = CT_matrix(number_of_points, j, channel, pot, k, W, I, v_mm, v_pm, v_mp, v_pp)
            eta_plus, eta_minus, delta_plus, delta_minus, epsilon = CP.Stapp_parameterization_phase(t_mm[-1], t_pm[-1], t_mp[-1], t_pp[-1], k_np1, factor)
            eta_p.append(eta_plus)
            eta_m.append(eta_minus)
            delta_p.append(delta_plus)
            delta_m.append(delta_minus)
            epsi.append(epsilon)
            T.append(t_mm)
            T.append(t_pm)
            T.append(t_mp)
            T.append(t_pp)
    return T, k, eta_p, eta_m, delta_p, delta_m, epsi
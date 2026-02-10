import numpy as np
from scipy.special import legendre
import Map

# Defining Constants for Minnesota potentials

m = 100
# roots and weight for partial wave integral
roots, w = Map.get_roots_and_weight(m)

K0 = 1.487   # fm^-2
K1 = 0.465  # fm^-2


# Defining constants
h_barc = 197.3269804  # Mev fm
#two_mu = 0.5*(939.56563+938.27205)
two_mu = 0.5*(938.9260+938.9260)/h_barc  # fm^-1

#two_mu_1 = 0.5*(939.56563+938.27205)
two_mu_1 = 0.5*(938.9260+938.9260)
fact = 1

def MP_1(k,kprime, costheta):
    f1 = 2*2*2*np.pi*np.pi*np.pi
    f_inv = 1/f1
    q_sq = k*k + kprime*kprime - 2*k*kprime*costheta
    cube1 = (np.sqrt(np.pi/K0))**3
    return f_inv*cube1*np.exp(-0.25*q_sq/K0)/h_barc

def MP_2(k,kprime, costheta):
    f1 = 2*2*2*np.pi*np.pi*np.pi
    f_inv = 1/f1
    q_sq = k*k + kprime*kprime - 2*k*kprime*costheta
    cube2 = (np.sqrt(np.pi/K1))**3
    return f_inv*cube2*np.exp(-0.25*q_sq/K1)/h_barc

def partial_potential3(pot_int, k, kp, l, roots, w):
    f = lambda costheta: pot_int(k, kp, costheta) * 2 * np.pi * legendre(l)(costheta)
    return np.dot(f(roots), w)



def Minnesota_Pot(k, kprime, costheta, V0, V1):
    f1 = 2*2*2*np.pi*np.pi*np.pi
    f_inv = 1/f1
    q_sq = k*k + kprime*kprime - 2*k*kprime*costheta
    cube1 = (np.sqrt(np.pi/K0))**3
    cube2 = (np.sqrt(np.pi/K1))**3
    return f_inv*((cube1*V0*np.exp(-0.25*q_sq/K0)) + cube2*V1*np.exp(-0.25*q_sq/K1))/h_barc


def partial_potential2(k, kp, l, V0, V1, roots, w):
    f = lambda costheta: Minnesota_Pot(k, kp, costheta, V0, V1) * 2 * np.pi * legendre(l)(costheta)
    return np.dot(f(roots), w)

def weight_and_root(Ecm, p1, p2, p3, np1, np2, mapp, factor):
    # choosing energy in CM frame
    Ecm = Ecm / h_barc  # MeV

    # calculating pole
    knp1 = np.sqrt(two_mu * Ecm)  # in fm^-1

    c = 35.0  # fm^-1

    # roots and weight for Lippmann-Schwinger equation
    kn, wn = Map.root_and_weight(mapp, p1, p2, p3, np1, np2, c)

    ki = np.array(kn)
    wi = np.array(wn)

    # calculating the W_i from i=1 to i = N+1
    ki_sq = ki * ki

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
    W = np.append(W_i, Wnp1) * factor
    k = np.append(ki, knp1)

    return k, W


def Call_TMatrix(number_of_points, l, V0, V1, k, W, I, V):
    N = number_of_points
    # this loop helps to fill up the N+1-by-N+1 block part of the potential matrix
    for i in range(number_of_points+1):
        for j in range(i, number_of_points+1, 1):
            if i == j:
                V[i, j] = partial_potential2(k[i], k[j], l,  V0, V1, roots, w)
            else:
                V[i, j] = partial_potential2(k[i], k[j], l,  V0, V1, roots, w)
                V[j, i] = V[i, j]

    GV_matrix = np.einsum('j,ij->ij', W, V)

    # calculating 1-GV matrix
    A = I - GV_matrix
    v = V[:, N]

    T = np.linalg.solve(A, v)

    return T, A, v

def weight_and_root2(knp1, ki, wi, factor):


    # calculating the W_i from i=1 to i = N+1
    ki_sq = ki * ki

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
    W = np.append(W_i, Wnp1) * factor
    k = np.append(ki, knp1)

    return k, W

def GV_matrix(number_of_points, l, k, W, I, V):
    pot_op0 = MP_1
    pot_op1 = MP_2
    v_first = np.copy(V)
    v_sec = np.copy(V)
    N = number_of_points
    # this loop helps to fill up the N+1-by-N+1 block part of the potential matrix
    for i in range(number_of_points+1):
        for j in range(i, number_of_points+1, 1):
            if i == j:
                v_first[i, j] = partial_potential3(pot_op0, k[i], k[j], l, roots, w)
                v_sec[i, j] = partial_potential3(pot_op1, k[i], k[j], l, roots, w)
            else:
                v_first[i, j] = partial_potential3(pot_op0, k[i], k[j], l, roots, w)
                v_sec[i, j] = partial_potential3(pot_op1, k[i], k[j], l, roots, w)

                v_first[j, i] = v_first[i, j]
                v_sec[j, i] = v_sec[i, j]

    GV_matrix_first = np.einsum('j,ij->ij', W, v_first)
    GV_matrix_sec = np.einsum('j,ij->ij', W, v_sec)
    v_1 = v_first[:, N]
    v_2 = v_sec[:, N]

    return GV_matrix_first, GV_matrix_sec, v_first, v_sec, v_1, v_2

def Call_TMatrix2(number_of_points, l, theta_vec, k, W, I, V):
    N = number_of_points
    GV_matrix_first, GV_matrix_sec, v_first, v_sec, v_1, v_2 = GV_matrix(number_of_points, l, k, W, I, V)
    # calculating 1-GV matrix
    A = I - theta_vec[0]*GV_matrix_first - theta_vec[1]*GV_matrix_sec
    v_tot = theta_vec[0]*v_1 + theta_vec[1]*v_2
    T = np.linalg.solve(A, v_tot)

    return T, A, v_tot,  v_first, v_sec, GV_matrix_first, GV_matrix_sec

import numpy as np
import trns as tr
from numpy.polynomial import Legendre as P

# function that will return k_i and w_i to calculate W_i and V
def tangent_map_roots_and_weight(N, c):
    # generate Legendre Polynomial
    PN = P.basis(N)

    # use roots() method to find roots, these are p"_i
    roots = PN.roots()

    # get k'_i
    k = []
    for i in range(0, N, 1):
        k.append(c * np.tan(0.25 * np.pi * (roots[i] + 1)))


    # calculate derivatives of P-N at the roots
    PN_prime = PN.deriv(1)(roots)

    # calculate the w_i
    w = []
    for i in range(0, N, 1):
        old_w = (2 / (1 - roots[i] ** 2)) / (PN_prime[i]) ** 2
        new = 0.25 * c * np.pi * old_w / ((np.cos(0.25 * np.pi * (roots[i] + 1))) ** 2)
        w.append(new)

    return k, w


def root_and_weight(mapp, p1, p2, p3, np1, np2, c):
    N = np1+np2
    if mapp == 'trns':
        roots = np.asfortranarray(np.empty((N,)))
        weights = np.asfortranarray(np.empty((N,)))
        tr.trns(np1, np2, p1, p2, p3, roots, weights)
    else:
        roots, weights = tangent_map_roots_and_weight(N, c)

    return roots, weights

# The following function will give the roots and weights for partial wave integration
# to get V(k, kprime) from V(k,kprime, costheta)
def get_roots_and_weight(m):
    # generate Legendre Polynomial
    PN = P.basis(m)
    # use roots() method to find roots
    roots = PN.roots()
    # calculate derivatives of P3 at the roots
    PN_prime = PN.deriv(1)(roots)
    # calculate the weights
    w = []
    for i in range(0, m, 1):
        w.append((2 / (1 - roots[i] * roots[i])) / (PN_prime[i] * PN_prime[i]))

    return roots, w

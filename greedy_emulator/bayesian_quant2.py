import multiprocessing

import numpy as np
#import Map
import h5py


import yaml
import numpy as np
with open("data/localGT+_lecs_order_2_R0_1.0_lam_1000.yaml", 'r') as stream:
    lecs = yaml.safe_load(stream)
lecs_filtered = {k: v for k, v in lecs.items() if (k != 'potId' and k!='R0' and k!='order' and k!='lambda')}
potid = {k: v for k, v in lecs.items() if k == 'potId'}
pot = potid['potId']
print('pot id is', pot)
#np.array([1.] + [theta_vec[elem] for elem in ("C1", "C2", "C3", "C4", "C5", "C6", "C7", "CS", "CNN", "CPP", "CT")])
#arr = np.array([lecs_filtered[elem] for elem in ("C1", "C2", "C3", "C4", "C5", "C6", "C7", "CS")] +[2.00894] +  [lecs_filtered[elem2] for elem2 in ("CNN", "CPP", "CT")])
arr = np.array([lecs_filtered[elem] for elem in ("C1", "C2", "C3", "C4", "C5", "C6", "C7", "CS", "CNN", "CPP", "CT")])
print('best fit val for given pot id is', arr)

# Defining constants
h_barc = 197.3269804  # Mev fm
h_barc_sq = h_barc * h_barc 
two_mu = 938.918747/h_barc  # fm^-1

J = 1
with h5py.File(f'Gezerlis_potential_data_80p1J={J}.h5', 'r') as file:
    vmm_loaded = file['vmm'][:]
    #print(vmm_loaded)
    vmp_loaded = file['vmp'][:]
    vpm_loaded = file['vpm'][:]
    vpp_loaded = file['vpp'][:]
    print("Data loaded successfully.")
    vmm_1 = vmm_loaded 
    vmp_1 = vmp_loaded 
    vpm_1 = vpm_loaded 
    vpp_1 = vpp_loaded

J = 2
with h5py.File(f'Gezerlis_potential_data_80p1J={J}.h5', 'r') as file:
    vmm_loaded = file['vmm'][:]
    vmp_loaded = file['vmp'][:]
    vpm_loaded = file['vpm'][:]
    vpp_loaded = file['vpp'][:]
    print("Data loaded successfully.")
    vmm_2 = vmm_loaded 
    vmp_2 = vmp_loaded 
    vpm_2 = vpm_loaded 
    vpp_2 = vpp_loaded

# for J =0 
with h5py.File(f'Gezerlis_potential_data_80Single_channel_data.h5', 'r') as file:
    v_00 = file['v_00'][:]
    v_0p = file['v_0p'][:]
    v_1j = file['v_1j'][:]
    v_2j = file['v_2j'][:]
    v_3j = file['v_3j'][:]
    v_4j = file['v_4j'][:]
    v_5j = file['v_5j'][:]
    v_6j = file['v_6j'][:]
    v_1jm = file['v_1jm'][:]
    v_2jm = file['v_2jm'][:]
    v_3jm = file['v_3jm'][:]
    v_4jm = file['v_4jm'][:]
    v_5jm = file['v_5jm'][:]
    v_6jm = file['v_6jm'][:]
mapp = 'trns'
np1 = 60
np2 = 20
number_of_points = np1+np2
p1 = 2.5  # fm^-1
p2 = 6.0  # fm^-1
p3 = 20.0  # fm^-1
#factor = 1 #2/np.pi


m = int(0.20*number_of_points)

c = 35.0  # fm^-1

# roots and weight for Lippmann-Schwinger equation
#kn, wn = Map.root_and_weight(mapp, p1, p2, p3, np1, np2, c)
#ki = np.array(kn)
#wi = np.array(wn)
N = number_of_points + 1
I = np.identity(N)


E_cms = 0.5* np.array([20.000,30.000, 40.000,50.000, 60.000, 80.000, 100.000,120.000, 140.000,160.000])   
exp_cross = np.array([ 484.13131, 309.13945, 220.06589, 167.77182,  134.31699, 95.54391, 74.85293, 62.54202, 54.60579, 49.16640])
Ecms = E_cms
factor = 2/np.pi


from numba import njit

@njit(fastmath=True, nopython=True)
def Smatrix_calculator_2emulated_opt_numba(lec_vec, one, factor, k0, Gmm, Gpp, Gmp, Gpm, v1, v2, v3, v4, X0, XH0, G_sing, G_trip, v_sing, v_trip,X1, XH1, X2, XH2,J):
    # Assemble A_upper matrix
    Amm = one.copy().astype(np.complex64)
    App = one.copy().astype(np.complex64)
    Amp = np.zeros_like(one, dtype=np.complex64)
    Apm = np.zeros_like(one, dtype=np.complex64)
    
    for i in range(len(lec_vec)):
        Amm += lec_vec[i] * Gmm[i]
        App += lec_vec[i] * Gpp[i]
        Amp += lec_vec[i] * Gmp[i]
        Apm += lec_vec[i] * Gpm[i]

    # Manually construct A = kron(I2, A_upper)
    N = Amm.shape[0]
    #print('shape of Amm.shape[0] is', N)
    
    A = np.zeros((2 * N, 2 * N), dtype=np.complex64)
    A[:N, :N] = Amm
    A[:N, N:] = Amp
    A[N:, :N] = Apm
    A[N:, N:] = App
    A_full = np.zeros((4 * N, 4 * N), dtype=np.complex64)
    A_full[:2*N, :2*N] = A
    A_full[2*N:, 2*N:] = A

    # Construct V
    V = np.zeros(4 * N, dtype=np.complex64)
    #print('v1[3,2] is', v1[3,2])
    print('shape of v is', v1.shape)
    assert v2.shape == (12, 81) 
    assert v3.shape == (12, 81) 
    assert v4.shape == (12, 81) 
    
    for i in range(len(lec_vec)):
        for j in range(N):
            assert type(lec_vec[i]) == np.complex64
            assert type(v1[i,j]) == np.complex64
            print('Type of element of v1 is', type(v1[i,j]))
            print('Type of element of V is', type(V[j]))
            V[j] += lec_vec[i] * v1[i, j]
            V[j + N] += lec_vec[i] * v2[i, j]
            V[j + 2 * N] += lec_vec[i]* v3[i, j]
            V[j + 3 * N] += lec_vec[i] *v4[i, j]

    # Solve main system
    M = XH0 @ A_full @ X0
    rhs = XH0 @ V
    c = np.linalg.solve(M, rhs)
    t = X0 @ c

    # Partition
    t_1 = t[0:N]
    t_2 = t[N:2*N]
    t_3 = t[2*N:3*N]
    t_4 = t[3*N:4*N]

    # Singlet channel
    A_sing = one.copy().astype(np.complex64)
    V_sing = np.zeros(N, dtype=np.complex64)
    
    for i in range(len(lec_vec)):
        A_sing += lec_vec[i] * G_sing[i]
        for j in range(N):
            V_sing[j] += lec_vec[i] * v_sing[i, j]

    c_sing = np.linalg.solve(XH1 @ A_sing @ X1, XH1 @ V_sing)
    t_sing = X1 @ c_sing

    # Triplet channel
    A_trip =one.copy().astype(np.complex64)
    V_trip = np.zeros(N, dtype=np.complex64)
    for i in range(len(lec_vec)):
        A_trip += lec_vec[i] * G_trip[i]
        for j in range(N):
            V_trip[j] += lec_vec[i] * v_trip[i, j]

    c_trip = np.linalg.solve(XH2 @ A_trip @ X2, XH2 @ V_trip)
    t_trip = X2 @ c_trip

    # Final expression
    const = -0.5 * factor * np.pi * two_mu * k0
    result = (2 * J + 1) * const * 2j * (t_sing[-1] + t_trip[-1] + t_1[-1] + t_4[-1])

    return result.real



@njit(fastmath=True, nopython=True)
def compute_single_channel_numba(one, lec_vec, G, v, X, XH):
    A = one.copy().astype(np.complex128)
    for i in range(lec_vec.shape[0]):
        A += lec_vec[i] * G[i]

    V = np.zeros(v.shape[1], dtype=np.complex128)
    print('shape of v is', v.shape)
    assert v.shape == (12,81)
    for i in range(lec_vec.shape[0]):
        for j in range(v.shape[1]):
            V[j] =  2j #v[i, j] # lec_vec[i] *

    new_matrix = XH @ A 
    new_matrix = new_matrix @ X
    new_vector = XH @ V
    c = np.linalg.solve(new_matrix, new_vector)
    t = X @ c

    return t



@njit(fastmath=True, nopython=True)
def Smatrix_calculator_J0emulated_opt(lec_vec, one, factor, k0, G_sing, G_trip, v_sing, v_trip, X1, XH1, X2, XH2):
    #lec_vec = np.insert(arr, 0, 1.0)  # Insert 1.0 at the front
    # Solve singlet and triplet channels
    t_sing = compute_single_channel_numba(one, lec_vec, G_sing, v_sing, X1, XH1)
    t_trip = compute_single_channel_numba(one, lec_vec, G_trip, v_trip, X2, XH2)
    const = -0.5 * factor * np.pi * two_mu * k0
    return (const * 2j*(t_sing[-1]+t_trip[-1])).real
    


def load_list(f, name):
    grp = f[name]
    keys = sorted(grp.keys(), key=lambda k: int(k.split('_')[-1]))
    return [grp[k][()] for k in keys]

import h5py
import numpy as np

def load_list(f, name):
    grp = f[name]
    keys = sorted(grp.keys(), key=lambda k: int(k.split('_')[-1]))
    return [grp[k][()] for k in keys]

with h5py.File("data_storage.h5", "r") as f:
    # Emulated Matrices
    X00 = load_list(f, "X00")
    X00_T = load_list(f, "X00_T")
    X01 = load_list(f, "X01")
    X01_T = load_list(f, "X01_T")
    Big_T1 = load_list(f, "Big_T1")
    Big_T_conj1 = load_list(f, "Big_T_conj1")
    X11 = load_list(f, "X11")
    X11_T = load_list(f, "X11_T")
    X10 = load_list(f, "X10")
    X10_T = load_list(f, "X10_T")
    Big_T2 = load_list(f, "Big_T2")
    Big_T_conj2 = load_list(f, "Big_T_conj2")
    X22 = load_list(f, "X22")
    X22_T = load_list(f, "X22_T")
    X21 = load_list(f, "X21")
    X21_T = load_list(f, "X21_T")

    # Single Channel Potentials
    G11 = load_list(f, "G11")
    v11 = load_list(f, "v11")
    G10 = load_list(f, "G10")
    v10 = load_list(f, "v10")
    G22 = load_list(f, "G22")
    v22 = load_list(f, "v22")
    G21 = load_list(f, "G21")
    v21 = load_list(f, "v21")
    G00 = load_list(f, "G00")
    v00 = load_list(f, "v00")
    G01 = load_list(f, "G01")
    v01 = load_list(f, "v01")

    # Coupled Channel Potentials (Set 1)
    Gmm_1 = load_list(f, "Gmm_1")
    Gpm_1 = load_list(f, "Gpm_1")
    Gmp_1 = load_list(f, "Gmp_1")
    Gpp_1 = load_list(f, "Gpp_1")
    v1_1 = load_list(f, "v1_1")
    v2_1 = load_list(f, "v2_1")
    v3_1 = load_list(f, "v3_1")
    v4_1 = load_list(f, "v4_1")

    # Coupled Channel Potentials (Set 2)
    Gmm_2 = load_list(f, "Gmm_2")
    Gpm_2 = load_list(f, "Gpm_2")
    Gmp_2 = load_list(f, "Gmp_2")
    Gpp_2 = load_list(f, "Gpp_2")
    v1_2 = load_list(f, "v1_2")
    v2_2 = load_list(f, "v2_2")
    v3_2 = load_list(f, "v3_2")
    v4_2 = load_list(f, "v4_2")

for name in ['v1_1', 'v2_1', 'v3_1', 'v4_1', 'v1_2', 'v2_2', 'v3_2', 'v4_2', 'v11', 'v10', 'v22', 'v21', 'v00', 'v01']:
    locals()[name] = [arr.astype(np.complex128) for arr in locals()[name]]
    locals()[name] = np.stack(locals()[name], axis=0) 
    
print('shape of v1_1[0] is', v1_1[0].shape)
print('shape of v1_1is', v1_1.shape)

#print("v1_1[i].shape =", v1_1[0].shape)
print("v1_1[0] type =", type(v1_1[0]))
print("v1_1.dtype =",  v1_1.dtype)

#print("v1_1[0][0].shape =", v1_1[0][0].shape)
#print('shape of v1_1 is', v1_1.shape)

K_ii =[]
Fact = []
for i, Ecm_i in enumerate(Ecms):
    Ecm = Ecm_i / h_barc
    k_ii = np.sqrt(two_mu * Ecm)
    K_ii.append(k_ii)
    fact= np.pi * 10 / (2 * k_ii * k_ii)
    Fact.append(fact)

new_arr = np.delete(arr, [8, 9])
print(new_arr)
@njit
def build_theta_vec(theta):
    out = np.empty(theta.shape[0] + 3, dtype=theta.dtype)
    out[0] = 1.0
    out[1:8+1] = theta[0:8]
    out[8+1] = 0.0
    out[8+2] = 0.0
    out[8+3:] = theta[8:]
    
    #print('lec_vec shape is ', out.shape)
    return out.astype(np.complex128)
i =1
theta_vec = build_theta_vec(new_arr) #np.insert(arr, 0, 1.0)
print(type(theta_vec))
#cross_a = Smatrix_calculator_2emulated_opt_numba(theta_vec, I, factor, K_ii[i], Gmm_1[i], Gpp_1[i], Gmp_1[i], Gpm_1[i], v1_1[i], v2_1[i], v3_1[i], 
#                                         v4_1[i], Big_T1[i], Big_T_conj1[i], G11[i], G10[i], v11[i], v10[i], X11[i], X11_T[i], X10[i], X10_T[i], 1)
#cross_b = Smatrix_calculator_J0emulated_opt(theta_vec, I, factor, K_ii[i], G00[i], G01[i], v00[i], v01[i], X00[i], X00_T[i], X01[i], X01_T[i])

print('test')

#@njit(fastmath=True, nopython=True)
def log_likelihood(theta, Ecms, y_exp,  k_ii, fact, X00, X00_T, X01, X01_T, Big_T1, Big_T_conj1, X11, X11_T, X10, X10_T, Big_T2, Big_T_conj2, X22, X22_T, X21, X21_T, G11, v11, G10, v10, G22, v22,G21, v21,
G00, v00, G01, v01, Gmm_1, Gpm_1, Gmp_1, Gpp_1, v1_1, v2_1, v3_1, v4_1, 
Gmm_2, Gpm_2, Gmp_2, Gpp_2, v1_2, v2_2, v3_2, v4_2, eta=0.1):
    theta_vec = build_theta_vec(theta)#np.insert(theta, [0, 8, 8], [ 1.0, 0.0, 0.0])
    print("lec_vec.shape =", theta_vec.shape)
    #theta_vec = np.empty(arr.shape[0] + 1, dtype=arr.dtype)
    #theta_vec[0] = 1.0
    #theta_vec[1:] = arr

    y_theo = np.zeros(len(y_exp))
    resid = np.zeros(len(y_exp))
    n_pts = len(y_exp)
    #cross_0 = np.zeros(n_pts)
    #cross_1 = np.zeros(n_pts)
    #cross_2 = np.zeros(n_pts)
    for i, Ecm_i in enumerate(Ecms):
        '''
        cross_1 = Smatrix_calculator_2emulated_opt_numba(theta_vec, I, factor, k_ii[i],Gmm_1[i],Gpp_1[i],Gmp_1[i],Gpm_1[i],v1_1[i],v2_1[i],v3_1[i], v4_1[i], Big_T1[i], Big_T_conj1[i], G11[i], G10[i], v11[i], v10[i], X11[i], X11_T[i], X10[i], X10_T[i], 1)
        '''
        cross_0 = Smatrix_calculator_J0emulated_opt(theta_vec, I, factor, k_ii[i], G00[i], G01[i], v00[i][:,:], v01[i][:,:], X00[i], X00_T[i], X01[i], X01_T[i])
        '''
        cross_2 = Smatrix_calculator_2emulated_opt_numba(theta_vec, I, factor, k_ii[i], Gmm_2[i], Gpp_2[i], Gmp_2[i], Gpm_2[i], v1_2[i], v2_2[i], v3_2[i], v4_2[i], Big_T2[i], Big_T_conj2[i], G22[i], G21[i], v22[i], v21[i], X22[i], X22_T[i], X21[i], X21_T[i], 2)
        '''
        
        #y_theo[i] = (cross_0+cross_1+ cross_2) * fact[i]
        y_theo[i] = cross_0* fact[i]
        resid[i] = y_exp[i] - y_theo[i]

    #y_theo = np.sum(cross_0, cross_1 + cross_2) * (np.pi * 10) / fact

    sigma = eta * y_theo
    #resid = y_exp - y_theo
   
    return  -0.5 * np.sum(resid**2 / sigma + np.log(2 * np.pi * sigma * sigma))
#@njit(fastmath=True, nopython=True)
def log_prior(theta, mean=arr):
    theta = np.asarray(theta)
    mean = np.asarray(mean)
    sigma = 0.5 * np.abs(mean)  # Standard deviation is 50% of the mean (absolute value)
    
    # Avoid division by zero
    sigma[sigma == 0] = 1e-16

    lp = -0.5 * np.sum(((theta - mean) / sigma) ** 2 + 2 * np.log(sigma) + np.log(2 * np.pi))
    return lp

#print(log_prior(sample_scaled[0], arr))
#@njit(fastmath=True, nopython=True)
def log_posterior(theta, mean, E_cms, y_exp, k_ii, X00, X00_T, X01, X01_T, Big_T1, Big_T_conj1, X11, X11_T, X10, X10_T, Big_T2, Big_T_conj2, X22, X22_T, X21, X21_T, G11, v11, G10, v10, G22, v22,G21, v21,
G00, v00, G01, v01, Gmm_1, Gpm_1, Gmp_1, Gpp_1, v1_1, v2_1, v3_1, v4_1, 
Gmm_2, Gpm_2, Gmp_2, Gpp_2, v1_2, v2_2, v3_2, v4_2, eta=0.1):
    lp = log_prior(theta, mean)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, E_cms, y_exp, k_ii, fact,X00, X00_T, X01, X01_T, Big_T1, Big_T_conj1, X11, X11_T, X10, X10_T, Big_T2, Big_T_conj2, X22, X22_T, X21, X21_T, G11, v11, G10, v10, G22, v22,G21, v21,
G00, v00, G01, v01, Gmm_1, Gpm_1, Gmp_1, Gpp_1, v1_1, v2_1, v3_1, v4_1, 
Gmm_2, Gpm_2, Gmp_2, Gpp_2, v1_2, v2_2, v3_2, v4_2, eta=0.1)


# note eta = 0.1 for our case 



import emcee
print(arr)


pos = new_arr + 1e-4 * np.random.randn(18, 9)
nwalkers, ndim = pos.shape
print(nwalkers, ndim)


sampler = emcee.EnsembleSampler(
    nwalkers, ndim, log_posterior, args=(new_arr, Ecms, exp_cross, K_ii, Fact, X00, X00_T, X01, X01_T, Big_T1, Big_T_conj1, X11, X11_T, X10, X10_T, Big_T2, Big_T_conj2, X22, X22_T, X21, X21_T, G11, v11, G10, v10, G22, v22,G21, v21,
G00, v00, G01, v01, Gmm_1, Gpm_1, Gmp_1, Gpp_1, v1_1, v2_1, v3_1, v4_1, 
Gmm_2, Gpm_2, Gmp_2, Gpp_2, v1_2, v2_2, v3_2, v4_2))
sampler.run_mcmc(pos, 5000, progress=True);





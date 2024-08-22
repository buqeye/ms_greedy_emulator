import numpy as np
from scipy.linalg import solve_banded
from scipy.sparse import spdiags


def diag_ord_form_to_mat(ab, ab_l_and_u, toarray=False):
    ab_B, N = ab.shape
    assert ab_B == sum(ab_l_and_u) + 1
    spmat = spdiags(ab, diags=np.arange(ab_l_and_u[1], -ab_l_and_u[0]-1, -1))
    return spmat.toarray() if toarray else spmat.tocsc()


def numerov_noopt(xn, g, y0=0, yp0=0, s=None, params=None):
    # preliminaries
    if s is None:
        s = lambda x, args: 0.*x
    g_arr = g(xn, params)  # g and s could be sampled simultaneously
    s_arr = s(xn, params)
    N = len(xn) - 1
    h = np.diff(xn)[0]

    def K(gn, xi=1):
        return 1. + xi * h**2 / 12 * gn
    
    # build matrix in diagonal ordered form
    ## first row
    ab = np.empty((4, N))
    ab[0, :] = 0 
    ab[0, 1] = 1 - K(g_arr[2], 1./2)

    ## second row
    K1 = K(g_arr, 1)
    ab[1, 0] = K(g_arr[1], 3.)
    ab[1, 1:] = K1[2:] 

    ## third row
    ab[2, :] = -2*K(g_arr, -5.)[1:]
    ab[2, -1] = 0 # not necessary but useful

    ## fourth row
    ab[3, :] = K1[1:]
    ab[3, -2:] = 0  # not necessary but useful
    
    # build rhs vector s
    rhs = np.empty(N)
    rhs[1:] = [s_arr[n] + 10*s_arr[n-1] + s_arr[n-2] for n in range(2, N+1)]
    rhs *= h**2 /12
    rhs[0] = y0* (1. - 7/2 * h**2 / 12 * g_arr[0]) + h * yp0
    rhs[0] += h**2/24 * (7*s_arr[0] + 6*s_arr[1] - s_arr[2])
    rhs[1] -= y0 * K1[0]

    # solve system
    from scipy.linalg import solve_banded
    sol = solve_banded(l_and_u=(2, 1), ab=ab, b=rhs)

    return ab, rhs, np.concatenate([[y0], sol])


def numerov2(xn, g, y0=0, yp0=0, s=None, solve=True, unittest=False, params=None):
    """
    implements the (N-1)x(N-1) Numerov matrix
    """
    # preliminaries
    if s is None:
        s = lambda x, args: 0.*x
    g_arr = g(xn, params)  # g and s could be sampled simultaneously
    s_arr = s(xn, params)
    N = len(xn) - 1
    h = np.diff(xn)[0]

    def K(gn, xi=1):
        return 1. + xi * h**2 / 12 * gn
    
    K1 = K(g_arr, 1)
    
    # build rhs vector s   
    rhs = h**2 /12 * np.array([s_arr[n] + 10*s_arr[n-1] + s_arr[n-2] for n in range(2, N+1)])
    b1 = y0* K(g_arr[0], -7/2) + h * yp0 + h**2/24 * (7*s_arr[0] + 6*s_arr[1] - s_arr[2])
    b2 = -y0 * K1[0] + rhs[0]
    det = K(g_arr[1], 3.) * K(g_arr[2], 1.) + 2*K(g_arr[1], -5.) * (1-K(g_arr[2], 1/2))
    y1 = (K1[2]*b1 - (1. - K(g_arr[2], 1/2))*b2) / det
    rhs[0] = b2 + 2*K(g_arr[1], -5.) * y1
    rhs[1] -= y1 * K1[1] 
 
    # build matrix in diagonal ordered form
    ab = np.empty((3, N-1))
    ## first row
    ab[0, :] = K1[2:] 

    ## second row
    ab[1, :] = -2*K(g_arr, -5.)[2:]
    ab[1, -1] = 0 # not necessary but useful

    ## third row
    ab[2, :] = K1[2:]
    ab[2, -2:] = 0  # not necessary but useful
    
    # solve system   
    ab_sparse = diag_ord_form_to_mat(ab, ab_l_and_u=(2,0))

    if solve:
        # from scipy.sparse.linalg import spsolve
        # sol_sparse = spsolve(ab_sparse, rhs)

        # if unittest:
        from scipy.linalg import solve_banded
        sol = solve_banded(l_and_u=(2, 0), ab=ab, b=rhs)
        # assert np.allclose(sol, sol_sparse)

        return ab_sparse, rhs, np.concatenate(([y0, y1], sol))
    else:
        return ab_sparse, rhs, [y0, y1]

def numerov(xn, g, y0=0, yp0=0, s=None, solve=True, unittest=False, params=None):
    """
    implements the (N-2)x(N-2) Numerov matrix
    """
    # preliminaries
    if s is None:
        s = lambda x, args: 0.*x
    g_arr = g(xn, params)  # g and s could be sampled simultaneously
    s_arr = s(xn, params)
    N = len(xn) - 1
    h = np.diff(xn)[0]

    def K(gn, xi=1):
        return 1. + xi * h**2 / 12 * gn
    
    K1 = K(g_arr, 1)
    
    # build rhs vector s
    b = np.empty(2)   
    b[0] = y0* K(g_arr[0], -7/2) + h * yp0 + h**2/24 * (7*s_arr[0] + 6*s_arr[1] - s_arr[2])
    b[1] = -y0 * K1[0] + h**2 /12 *( s_arr[2] + 10*s_arr[1] + s_arr[0])
    mat = np.array([[K(g_arr[1], 3), 1-K(g_arr[2], 0.5)], 
                   [-2*K(g_arr[1], -5), K1[2]]])
    y1_y2 = np.linalg.solve(mat, b)

    rhs = h**2 /12 * np.array([s_arr[n] + 10*s_arr[n-1] + s_arr[n-2] for n in range(3, N+1)])
    rhs[0] += -K1[1] * y1_y2[0] + 2*K(g_arr[2], -5.) * y1_y2[1]
    rhs[1] -= y1_y2[1] * K1[2] 
 
    # build matrix in diagonal ordered form
    ab = np.empty((3, N-2))
    ## first row
    ab[0, :] = K1[3:] 

    ## second row
    ab[1, :] = -2*K(g_arr, -5.)[3:]
    ab[1, -1] = 0 # not necessary but useful

    ## third row
    ab[2, :] = K1[3:]
    ab[2, -2:] = 0  # not necessary but useful
    
    # solve system   
    ab_sparse = diag_ord_form_to_mat(ab, ab_l_and_u=(2,0))

    if solve:
        # from scipy.sparse.linalg import spsolve
        # sol_sparse = spsolve(ab_sparse, rhs)

        # if unittest:
        from scipy.linalg import solve_banded
        sol = solve_banded(l_and_u=(2, 0), ab=ab, b=rhs)
        # assert np.allclose(sol, sol_sparse)

        return ab_sparse, rhs, np.concatenate(([y0], y1_y2, sol))
    else:
        return ab_sparse, rhs, np.concatenate(([y0], y1_y2))


def numerov_euler(xn, g, y0=0, yp0=0, s=None, unittest=False, params=None):
    # preliminaries
    if s is None:
        s = lambda x, params: 0.*x
    g_arr = g(xn, params)  # g and s could be sampled simultaneously
    s_arr = s(xn, params)
    N = len(xn) - 1
    h = np.diff(xn)[0]

    def K(gn, xi=1):
        return 1. + xi * h**2 / 12 * gn
    
    # build matrix in diagonal ordered form
    ab = np.empty((3, N))

    ## second row
    K1 = K(g_arr, 1)
    ab[0, 0] = 1
    ab[0, 1:] = K1[2:] 

    ## third row
    ab[1, :] = -2*K(g_arr, -5.)[1:]
    ab[1, -1] = 0 # not necessary but useful

    ## fourth row
    ab[2, :] = K1[1:]
    ab[2, -2:] = 0  # not necessary but useful
    
    # build rhs vector s
    rhs = np.empty(N)
    rhs[1:] = [s_arr[n] + 10*s_arr[n-1] + s_arr[n-2] for n in range(2, N+1)]
    rhs *= h**2 /12
    rhs[0] = y0 + h * yp0
    rhs[1] -= y0 * K1[0]

    # solve system
    from scipy.linalg import solve_banded
    sol = solve_banded(l_and_u=(2, 0), ab=ab, b=rhs)

    # solve sparse system (just for checking)  
    if unittest:
        ab_u = 0
        ab_l = 2
        ab_B = ab_u + ab_l + 1
        ab_sparse = np.empty((N,N))
        for i in range(N):
            for j in range(N):
                row_index = ab_u + i - j
                ab_sparse[i,j] = ab[ab_u + i - j, j] if row_index >= 0 and row_index < ab_B else 0

        sol_sparse = np.linalg.solve(ab_sparse, rhs)
        assert np.allclose(sol, sol_sparse)
    
    return ab, rhs, np.concatenate([[y0], sol])


def numerov_iter(xn, g, y0=0, yp0=0, s=None, params=None):
    if s is None:
        s = lambda x, args: 0.*x
    h = np.diff(xn)[0]

    def K(x, xi=1):
        return 1. + xi * h**2 / 12 * g(x, params)

    N = len(xn) - 1
    yn = np.empty(N+1)

    yn[0] = y0
    yn[1] = yn[0] + h * yp0  # simple Euler step

    for n in range(1, N):
        yn[n+1] = 2 * yn[n] * K(xn[n], -5) - yn[n-1] * K(xn[n-1], 1)
        yn[n+1] += h**2/12 * (s(xn[n+1], params) + 10*s(xn[n], params) + s(xn[n-1], params))
        yn[n+1] /= K(xn[n+1], 1)
    return yn


class AffineNumerovSolver:
    def __init__(self, xn, g, s=None, g_s=None, y0=0., yp0=0., params=None) -> None:
        # build Numerov matrix
        ## preliminaries
        self.xn = xn
        self.N = len(xn)-1
        self.h = np.diff(xn)[0]  # assuming an equidistant grid
        self.step_fac = self.h**2 / 12
        self.params = params
        self.g = g
        self.s = s
        self.g_s = g_s
        if g_s is None:
            self.gn = self.g(xn, self.params)
            self.sn = np.zeros_like(self.gn) if s is None else s(xn, self.params)
        else:
            self.gn, self.sn = self.g_s(xn, params)
            self.g = self.s = None
        self.y0 = y0
        self.yp0 = yp0
        
        # construct banded (!) matrix
        self.A_l_and_u= 2, 0
        self.A_bandwidth = sum(self.A_l_and_u)+1
        self.A_const = np.outer(np.array([1, -2, 1]), np.ones(self.N-2))
        self.A_theta = np.einsum("i,jk->ijk", 
                                 np.array([1., 10., 1.]) * self.step_fac, self.gn[3:,:], optimize="greedy")
        # optional: we could set here the ununsed elements to zero

        # build linear system to compute (y1,y2) from initial values
        self.Aiv_const = np.array([[1., 0.], [-2., 1.]])
        self.Aiv_theta = np.einsum("ij,jk->ijk", np.array([[6., -1.], [20., 2.]]), self.gn[1:3, :], optimize="greedy") * self.step_fac/2. 
        self.biv_const = np.array([y0 + self.h*yp0, -y0])
        self.biv_theta = -y0 * np.outer(np.array([7., 2.]), self.gn[0, :]) + np.array([[7., 6., -1.], [2., 20., 2.]]) @ self.sn[:3, :]
        self.biv_theta *= self.step_fac/2. 

        # build Numerov inhomogeneous term `s`
        # `self.S_const` cannot be prestored because it depends on (y1, y2), which depend on `theta`
        mat = spdiags(np.outer([1., 10., 1.], np.ones(self.N)), 
                      diags=(0,1,2), m=(self.N-2), n=self.N)
        self.S_theta = self.step_fac * mat @ self.sn[1:,:] 

    @property
    def A_const_theta_dense(self):
        A_const = diag_ord_form_to_mat(self.A_const, ab_l_and_u=self.A_l_and_u, toarray=True)
        mat = spdiags(np.outer([1., 10., 1.], np.ones(self.N)), 
                      diags=(0,-1,-2), m=(self.N-2), n=self.N-2).toarray()
        A_theta = self.step_fac * np.einsum("ij,jk->ijk", mat, self.gn[3:,:], optimize=True)
        return A_const, A_theta

    def get_S_const(self, theta):
        y1_y2 = self.get_y1_y2(theta)
        S_const = np.array([[-1, 2.], [0, -1]]) @ y1_y2
        S_const -= np.einsum("ij,j,jk,k->i", np.array([[1., 10.], [0., 1.]]), 
                             y1_y2, self.gn[1:3, :], theta, optimize="greedy") * self.step_fac
        return S_const, y1_y2

    def get_y1_y2(self, theta):
        A = self.Aiv_const + self.Aiv_theta @ theta 
        b = self.biv_const + self.biv_theta @ theta
        return np.linalg.solve(A.astype(np.double), b)

    def get_linear_system(self, theta):
        A = self.A_const + self.A_theta @ theta
        s = self.S_theta @ theta
        S_const, y1_y2 = self.get_S_const(theta)
        s[:2] += S_const
        return A, s, y1_y2

    def solve(self, thetas):
        thetas = np.asarray(thetas)
        ret = []
        for theta in thetas:
            A_banded, s, y1_y2 = self.get_linear_system(theta)
            sol = solve_banded(l_and_u=self.A_l_and_u, ab=A_banded, b=s)
            ret.append(np.concatenate([[self.y0], y1_y2, sol]))
        return A_banded, s, y1_y2, np.array(ret)
    
    def residuals(self, xtilde, theta, squared=True, calc_error_bounds=False):
        A_banded, s, y1_y2 = self.get_linear_system(theta)
        A = diag_ord_form_to_mat(A_banded, ab_l_and_u=self.A_l_and_u, toarray=True)
        residual = A @ xtilde - s 
        error = np.linalg.norm(residual)
        lower_bound = None 
        upper_bound = None
        if calc_error_bounds:
            svals = np.linalg.svd(A, compute_uv=False)
            lower_bound = error / svals[0]  # sval_lm
            upper_bound = error / svals[-1]  # sval_sm
            assert lower_bound <= upper_bound, "lower bound has to be <= upper bound"
        return error**2 if squared else error, lower_bound, upper_bound

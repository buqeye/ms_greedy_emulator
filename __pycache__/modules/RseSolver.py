from scipy.integrate import solve_ivp
from scipy.special import spherical_jn, spherical_yn, eval_legendre
import matplotlib.pyplot as plt
import numpy as np
from uMatrix import getUmatrix
from constants import *


class LMatrix:
    def __init__(self, lbl, value=None):
        self.value = value
        self.uMat = getUmatrix(lbl)

    @property
    def lbl(self):
        return self.uMat.lbl

    def linearFracTrafo(self, u):
        tmp = u.T @ self._toFrac(self.value)
        return tmp[1] / tmp[0]  # {denominator, numerator}

    def _toFrac(self, value):
        return np.array([1, value])  # {denominator, numerator}

    def valueAs(self, LprimeLbl=None):
        if (LprimeLbl is None) or (self.lbl == LprimeLbl):
            return self.value
        else:
            up = getUmatrix(LprimeLbl)
            return self.linearFracTrafo(self.uMat.u @ up.uInv)
            # in general {denominator, numerator} = det(u') (u'^(-1))^T u^T * {1, L}
            # but notice that det(u') is a global factor and thus doesn't matter;
            # also notice that u^T * {1, L} \propto {1, K},

    @property
    def K(self):
        # just to illustrate the special case for the K matrix
        if self.lbl == "K":
            return self.value
        else:
            return self.linearFracTrafo(self.uMat.u)

    @property
    def S(self):
        # just to illustrate the special case for the S matrix
        if self.lbl == "S":
            return self.value
        else:
            return self.linearFracTrafo(self.uMat.u @ np.array([[1, 1], [-1j, 1j]]))

    @property
    def phaseShift(self):
        return np.arctan(self.K) * degrees  # K = tan delta
        # np.arctan() does not support complex types, so we can't use
        # the fraction representation of the K matrix here in general

        # return np.log(self.S)/2j * degrees  # S = exp[2i delta]
        # note that using np.angle() would be incorrect here, because it
        # returns only the real part of the (in general complex) phase shift

    def fl(self, p):  # p in fm**-1
        return (self.S - 1.) / (2j * p)  # fm

    def sigmaL(self, l, p):
        cs = (4.*np.pi) * (2*l + 1) * np.abs(self.fl(p)) ** 2  # fm**2
        return cs * 10.  # mb [100 fm**2 = 1 b]

    def dsigmaL(self, l, p, atheta, deg=True):  # p in fm**-1
        """
        Computes a given partial wave's contribution to the differential section
        See Eqs. (11.2) to (11.4) in Taylor’s book.
        """
        theta = atheta/degrees if deg else atheta
        diffCs = (2*l + 1) * self.fl(p) * eval_legendre(l, np.cos(theta))  # fm
        return diffCs * np.sqrt(10.)  # sqrt(mb)

    @staticmethod
    def getWaveFuncU(up, u):
        tmp = np.array([[+u.u[1, 1], +up.u[1, 1]],
                        [-u.u[1, 0], -up.u[1, 0]]])
        mat = u.u @ (tmp @ np.diag((up.det, u.det)))
        isSingular = np.isclose(np.abs(np.linalg.det(mat)), 0)
        return mat, isSingular

    def waveFunctionFactor(self, LprimeLbl):
        if self.lbl == LprimeLbl:
            return 1.
        up = getUmatrix(LprimeLbl)
        mat, isScalar = self.getWaveFuncU(up, self.uMat)
        if isScalar:
            return mat[0, 1] / mat[0, 0]
            # mat [1, 0] is always zero, so this is the correct limit value
        else:
            return self.linearFracTrafo(mat)

    def printWaveFuncDependencies(self, Lrequested=None):
        Larr = ("K", "S", "T", "Kinv", "Sinv", "Tinv") if (Lrequested is None) else Lrequested
        print(Larr)
        for mat0 in Larr:
            u = getUmatrix(mat0)
            # print(self.waveFunctionFactor(mat0), end="\t")
            for mat1 in Larr:
                up = getUmatrix(mat1)
                _, isScalar = self.getWaveFuncU(up, u)
                print(isScalar, end="\t")
            print("")


def _rse(r, u, params):
    l = params["scattExp"].l
    mu = params["scattExp"].mu
    E = params["scattExp"].en
    potential = params["potential"]
    lecs = params["lecs"]
    return [u[1], (l * (l + 1) / r ** 2. + (2. * mu) * (potential.eval(r, lecs) - E)) * u[0]]


class ScatteringSolution:
    def __init__(self, scattExp, potential, grid, u, uprime, vr, anc, Llbl, 
                 linear_system, matching):
        self.scattExp = scattExp
        self.potential = potential
        self.grid = grid
        self.u = u  # wave function sampled on 'grid'
        self.uprime = uprime  # derivative of wave function sampled on 'grid'
        self.vr = vr  # V(r) sampled on 'grid'
        self.anc = None
        self.Lmatrix = None
        self.linear_system = linear_system
        if matching:
            self.match(anc, param=Llbl)  # sets the scattering matrix 'self.Lmatrix' and 'self.anc'

    @property
    def chi(self):
        """returns the free solution with chi'(r=0) = 1"""
        ret= self.u - self.grid.points * spherical_jn(self.scattExp.l, 
                                                        self.scattExp.p*self.grid.points)
        ret[0] = 0
        # TODO phi solution needs to be scaled by `scale` from `match()`;
        # perhaps remove this property?
        return ret

    @property
    def rmatch(self):
        return self.grid.points[-1]

    @property
    def r0(self):
        return self.grid.points[0]

    def waveFunctionAs(self, LprimeLbl=None, prime=False):
        tmp = self.uprime if prime else self.u
        if (LprimeLbl is None) or (self.Lmatrix.lbl == LprimeLbl):
            return tmp
        else:
            pre = 1./self.Lmatrix.waveFunctionFactor(LprimeLbl)
            return pre * tmp

    @property
    def fl(self):
        return self.Lmatrix.fl(self.scattExp.p)

    @property
    def sigmaL(self):
        return self.Lmatrix.sigmaL(self.scattExp.potential.channel.L, self.scattExp.p)

    def dsigmaL(self, atheta, deg=True):
        return self.Lmatrix.dsigmaL(self.scattExp.potential.channel.L,
                                    self.scattExp.p, atheta, deg=deg)

    @property
    def phaseShift(self):
        return self.Lmatrix.phaseShift

    @property
    def cond(self):
        assert self.linear_system is not None
        from numpy.linalg import cond
        return cond(self.linear_system[0].toarray()) 

    @property
    def singular_values(self):
        assert self.linear_system is not None
        from numpy.linalg import svd 
        return svd(self.linear_system[0].toarray(), full_matrices=False, compute_uv=False)
        # it would be better to use a routine for sparse matrices like the following, which does
        # not converge for some reason
        # from scipy.sparse.linalg import svds
        # s1= svds(self.linear_system[0], k=1, tol=1e-2, 
        #          maxiter=100, solver="lobpcg", 
        #          return_singular_vectors=False, which="SM")


    def _getPhiPhiPrime(self, r, uMat):
        rho = r * self.scattExp.p  # here's no "np.pi/2 * l" because we use the j_l and eta_l below
        l = self.scattExp.l
        phi = rho * np.dot(uMat.u, np.array([spherical_jn(l, rho), -spherical_yn(l, rho)]))  # (3.1.13)
        phiPrime = phi/rho + rho * np.dot(uMat.u, np.array([spherical_jn(l, rho, derivative=True),
                                                           -spherical_yn(l, rho, derivative=True)]))  # (3.1.13)
        phiPrime *= self.scattExp.p  # from the derivative
        return phi, phiPrime

    def match(self, anc, *, param="T"):
        self.Lmatrix = LMatrix(param)
        phi, phiPrime = self._getPhiPhiPrime(self.rmatch, self.Lmatrix.uMat)
        logDeriv = self.u[-1]/self.uprime[-1]

        L = -(phi[0]-logDeriv*phiPrime[0])/(phi[1]-logDeriv*phiPrime[1])
        self.Lmatrix.value = L

        scale = anc*(phi[0]+L*phi[1])/self.u[-1]
        self.u *= scale
        self.uprime *= scale
        self.anc = anc

    def plot(self):
        plt.plot(self.grid.points, self.u)
        #plt.show()


def _solve(lecs, params):
    # set params for the RSE function
    rseParams = {**params, "lecs": lecs}

    # solve the radial SE
    if params["method"] == "Numerov":
        y0 = 0
        yp0 = 1  # solve the homogeneous SE
        from Numerov import numerov

        def g(r, rseParams):
            l = rseParams["scattExp"].l
            mu = rseParams["scattExp"].mu
            E = rseParams["scattExp"].en
            p = rseParams["scattExp"].p
            potential = rseParams["potential"]
            lecs = rseParams["lecs"]
            return -l * (l + 1) / r ** 2. - (2. * mu) * (potential.eval(r, lecs) - E)
            
        # def s(r):
        #     from scipy.special import spherical_jn  # , spherical_yn
        #     return potential.eval(r, lecs) * spherical_jn(l, p*r) * r * (2. * mu) 
        #     # note that we need to divide by because d/dr j(kr=0) = k j'(kr),
        #     # so the derivative at the origin is not 1 (in the case of S wave scattering)
        
        ab, rhs, u = numerov(xn=params["grid"].points, y0=y0, yp0=yp0, s=None, g=g, params=rseParams)
        uprime = np.gradient(u, params["grid"].points, edge_order=2)
        linear_system = (ab, rhs)
    else:
        sol = solve_ivp(_rse, (params["grid"].start, params["grid"].end), [complex(0), complex(1)],
                        method=params["method"], args=([rseParams]), t_eval=params["grid"].points,
                        rtol=1e-12, atol=1e-12)
        u=sol.y[0]
        uprime=sol.y[1]
        linear_system = None

    vr = params["potential"].eval(params["grid"].points, lecs)

    # apply boundary conditions by matching to the analytic asymptotics
    scattSol = ScatteringSolution(scattExp=params["scattExp"], potential=params["potential"], vr=vr,
                                  grid=params["grid"], u=u, uprime=uprime,
                                  anc=1./params["scattExp"].p, Llbl=params["asympParam"],
                                  matching=params["matching"], linear_system=linear_system)

    return scattSol


def solve(scattExp, grid, lecList, *, matching=True, method='RK45', asympParam="T"):
    """
    Solves the single channel 2-body scattering problem by using a Runge-Kutta integrator
    from the scipy.integrate package, and then applying scattering boundary conditions.
    """

    if np.abs(grid.start) > 1e-6:
        raise ValueError(f"Grid doesn't start at zero; grid.points[0]={grid.points[0]}")

    # solve RSE for all lec sets in 'lecList'
    params = {"grid": grid, "scattExp": scattExp, "potential": scattExp.potential,
              "asympParam": asympParam, "method": method, "matching": matching}

    # parallelized
    # pool = mp.Pool(mp.cpu_count())
    # scattSols = pool.starmap_async(_solve, [(lecs, params) for lecs in lecList]).get()
    # pool.close()

    # serial (parallelized externally)
    scattSols = [_solve(lecs, params) for lecs in lecList]

    """ printing out the phase shift and plotting the wave functions is useful for debugging
    for sol in scattSols:
        sol.plot()
        print("delta", sol.phaseShift)
    plt.show()
    """
    return scattSols

    
def g(r, params):
    l = params["scattExp"].l
    mu = params["scattExp"].mu
    E = params["scattExp"].en
    potential = params["potential"]
    lecs = params["lecs"]
    centrifugal = -l * (l + 1) / r ** 2. if l > 0 else 0.
    return centrifugal - (2. * mu) * (potential.eval(r, lecs) - E)

def s(r, params):
    from scipy.special import spherical_jn  # , spherical_yn
    l = params["scattExp"].l
    mu = params["scattExp"].mu
    p = params["scattExp"].p
    potential = params["potential"]
    lecs = params["lecs"]
    return potential.eval(r, lecs) * spherical_jn(l, p*r) * r * (2. * mu) 
    # note that we need to divide by because d/dr j(kr=0) = k j'(kr),
    # so the derivative at the origin is not 1 (in the case of S wave scattering)

def g_s(r, params):
    l = params["scattExp"].l
    mu = params["scattExp"].mu
    E = params["scattExp"].en
    p = params["scattExp"].p
    potential = params["potential"]
    lecs = params["lecs"]
    pot = potential.eval(r, lecs)
    centrifugal = -l * (l + 1) / r ** 2. if l > 0 else 0.
    g_arr = centrifugal - (2. * mu) * (pot - E)
    s_arr = pot * spherical_jn(l, p*r) * r * (2. * mu) 
    # note that we need to divide by because d/dr j(kr=0) = k j'(kr),
    # so the derivative at the origin is not 1 (in the case of S wave scattering)
    return g_arr, s_arr

def g_s_affine(r, params):
    l = params["scattExp"].l
    mu = params["scattExp"].mu
    E = params["scattExp"].en
    p = params["scattExp"].p
    potential = params["potential"]
    V_arr = potential.evalAffine(r)
    g_arr = - (2. * mu) * V_arr
    centrifugal = -l * (l + 1) / r ** 2. if l > 0 else 0.
    g_arr[:, 0] += centrifugal + (2. * mu) * E
    s_arr = np.einsum("i,ij->ij", spherical_jn(l, p*r) * r * (2. * mu), V_arr)
    # note that we need to divide by because d/dr j(kr=0) = k j'(kr),
    # so the derivative at the origin is not 1 (in the case of S wave scattering)
    return g_arr, s_arr
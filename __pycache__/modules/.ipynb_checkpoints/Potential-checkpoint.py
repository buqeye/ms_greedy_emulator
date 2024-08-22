import numpy as np

import chiralPot
from constants import *


class Potential:
    def __init__(self, channel, **kwargs):
        self.channel = channel
        self.name = kwargs["label"]
        self.kwargs = kwargs["kwargs"]
        if self.name == 'chiral':
            self.potentialFunction = chiral
        elif self.name == 'woodssaxon':
                self.potentialFunction = woodssaxon
        elif self.name == 'optical':
                self.potentialFunction = optical
        elif self.name == 'minnesota':
                self.potentialFunction = minnesota
        else:
            raise ValueError(f"Potential '{self.name}' unknown.")
        _, testingLecs = self.sampleLecs
        self.parameter_names = list(testingLecs[0].keys())

    def eval(self, r, lecs):
        if isinstance(r, np.ndarray):
            # return np.array([ self.potentialFunction(rval, **lecs) / hbarc for rval in r])
            return np.array(list(map(lambda rval: self.potentialFunction(rval, self.channel, **lecs, **self.kwargs), r)),
                            dtype=np.double) / hbarc
        elif isinstance(r, float):
            return self.potentialFunction(r, self.channel, **lecs, **self.kwargs) / hbarc
        else:
            raise ValueError("Check input parameters of 'Potential.eval()'.")

    def evalAffine(self, r):
        if self.name != "minnesota":
            raise NotImplementedError
        Vcomp = [np.zeros_like(r)]
        r2 = r ** 2
        K_arr = [1.487, 0.465]
        for K in K_arr:
            Vcomp.append(np.exp(-K * r2))
        return np.array(Vcomp).T  / hbarc

    @property
    def sampleLecs(self):
        return Potential.getSampleLecs(self.name)

    def getLecsSample(self, free_lecs, req_lecs=None, n=100, range_factor=0.1, seed=123, mode="random", as_dict=True):
        d = len(free_lecs)
        _, testingLecs = self.sampleLecs
        base_lecs = testingLecs[0]
        if free_lecs is None:
            free_lecs = list(base_lecs.keys())
        base = np.array([base_lecs[key] for key in free_lecs])
        offset = range_factor * np.abs(base)
        l_bounds = base - offset
        u_bounds = base + offset
        if mode == "random":
            from scipy.stats import qmc
            sampler = qmc.LatinHypercube(d=d, seed=seed)
            samples = qmc.scale(sampler.random(n), l_bounds, u_bounds)
        elif mode == "linear":
            lin_spaces = (np.linspace(l_bounds[i], u_bounds[i], n) for i in range(len(free_lecs)))
            samples = np.array(np.meshgrid(*lin_spaces, indexing='ij')).T.reshape(-1,len(free_lecs))
        else:
            raise ValueError(f"mode {mode} unknown to `getLecsSample()`")
        
        ret = [{**base_lecs, **{key: samples[j, ikey] for ikey, key in enumerate(free_lecs)}} for j in range(len(samples))]
        if as_dict:
            return ret
        else:  # TODO: this is hacked.. improve!
            return np.column_stack((np.ones(n), np.array([[elem[lec] for lec in req_lecs] for elem in ret])))

    @staticmethod
    def getSampleLecs(potLbl, A=1):
        """
        Parameters
        ----------
        potential

        Returns
        -------
        LECs for some training and testing
        """

        trainingLecs = []
        testingLecs = []

        if potLbl == 'chiral':
            trainingLecs.append({"CS": 5., "CT": 0.2, "C1": -0.14084, "C2": 0.04243,
            "C3": -0.12338, "C4": 0.11018, "C5": -2.11254,
            "C6": 0.15898, "C7": -0.26994, "CNN": 0.04344, "CPP": 0.062963})

            trainingLecs.append({"CS": 6., "CT": 0.2, "C1": -0.14084, "C2": 0.04243,
            "C3": -0.12338, "C4": 0.11018, "C5": -2.11254,
            "C6": 0.15898, "C7": -0.26994, "CNN": 0.04344, "CPP": 0.062963})

            trainingLecs.append({"CS": 5., "CT": 0.3, "C1": -0.14084, "C2": 0.04243,
            "C3": -0.12338, "C4": 0.11018, "C5": -2.11254,
            "C6": 0.15898, "C7": -0.26994, "CNN": 0.04344, "CPP": 0.062963})

            # for i in range(1000):
            trainingLecs.append({"CS": 6., "CT": 0.3, "C1": -0.14084, "C2": 0.04243,
            "C3": -0.12338, "C4": 0.11018, "C5": -2.11254,
            "C6": 0.15898, "C7": -0.26994, "CNN": 0.04344, "CPP": 0.062963})

            testingLecs.append({"CS": 5.43850, "CT": 0.27672, "C1": -0.14084, "C2": 0.04243,
            "C3": -0.12338, "C4": 0.11018, "C5": -2.11254,
            "C6": 0.15898, "C7": -0.26994, "CNN": 0.04344, "CPP": 0.062963})

            testingLecs.append({"CS": 5.53850, "CT": 0.37672, "C1": -0.14084, "C2": 0.04243,
            "C3": -0.12338, "C4": 0.11018, "C5": -2.11254,
            "C6": 0.15898, "C7": -0.26994, "CNN": 0.04344, "CPP": 0.062963})

        elif potLbl == 'woodssaxon':
            # trainingLecs.append({"depth": 40, "radius": 3, "diffuseness": 0.5})
            # trainingLecs.append({"depth": 60, "radius": 3, "diffuseness": 0.5})
            # trainingLecs.append({"depth": 40, "radius": 4, "diffuseness": 0.5})
            # trainingLecs.append({"depth": 60, "radius": 4, "diffuseness": 0.5})
            # # trainingLecs.append({"depth": 55, "radius": 3, "diffuseness": 0.5})
            #
            # testingLecs.append({"depth": 50, "radius": 3, "diffuseness": 0.5})

            tmp = [{'diffuseness': 0.303, 'depth': 66.69, 'radius': 4.018},
             {'diffuseness': 0.549, 'depth': 58.072, 'radius': 3.611},
             {'diffuseness': 0.667, 'depth': 48.636, 'radius': 3.995},
             {'diffuseness': 0.4, 'depth': 66.28, 'radius': 3.981},
             {'diffuseness': 0.35, 'depth': 60.526, 'radius': 3.928},
             {'diffuseness': 0.372, 'depth': 34.963, 'radius': 3.542},
             {'diffuseness': 0.413, 'depth': 35.917, 'radius': 3.584},
             {'diffuseness': 0.604, 'depth': 43.708, 'radius': 3.795}]
            for elem in tmp:
                trainingLecs.append(elem)

            tmp = [{'diffuseness': 0.628, 'depth': 39.358, 'radius': 3.253}]
            for elem in tmp:
                testingLecs.append(elem)

        elif potLbl == 'optical':
            tmp = 1.2 * np.cbrt(A)
            trainingLecs.append({"V": 45, "R": tmp, "a": 0.65, "Vw": 5, "Rw": tmp, "aw": 0.65})
            trainingLecs.append({"V": 55, "R": tmp, "a": 0.65, "Vw": 5, "Rw": tmp, "aw": 0.65})
            trainingLecs.append({"V": 45, "R": tmp, "a": 0.65, "Vw": 15, "Rw": tmp, "aw": 0.65})
            trainingLecs.append({"V": 55, "R": tmp, "a": 0.65, "Vw": 15, "Rw": tmp, "aw": 0.65})

            testingLecs.append({"V": 50, "R": tmp, "a": 0.65, "Vw": 10, "Rw": tmp, "aw": 0.65})

        elif potLbl == "minnesota":
            #trainingLecs.append({"V0": 0, "V1": -291.85, "K0": 1.487, "K1": 0.465})
            #trainingLecs.append({"V0": 100, "V1": 8.15, "K0": 1.487, "K1": 0.465})
            #trainingLecs.append({"V0": 300, "V1": -191.85, "K0": 1.487, "K1": 0.465})
            #trainingLecs.append({"V0": 300, "V1": 8.15, "K0": 1.487, "K1": 0.465})

            #testingLecs.append({"V0": 200, "K0": -91.85, "V1": 1.487, "K1": 0.465})

            #trainingLecs.append({"V0": 300, "V1": 8.15, "K0": 1.487, "K1": 0.465})

            trainingLecs.append({"V0": 0,   "V1": -291.85, "K0": 1.487, "K1": 0.465})
            trainingLecs.append({"V0": 100, "V1": 8.15, "K0": 1.487, "K1": 0.465})
            trainingLecs.append({"V0": 300, "V1": -191.85, "K0": 1.487, "K1": 0.465})
            trainingLecs.append({"V0": 300, "V1": 8.15, "K0": 1.487, "K1": 0.465})

            testingLecs.append({"V0": 200, "V1": -91.85, "K0": 1.487, "K1": 0.465})


        else:
            raise ValueError(f"Potential '{potLbl}' unknown.")

        return trainingLecs, testingLecs


def woodssaxon(x, chan, **kwargs):
    return -kwargs["depth"] / (1. + np.exp((x - kwargs["radius"]) / kwargs["diffuseness"]))


def woodssurface(x, chan, **kwargs):
    expon = np.exp(-(x - kwargs["radius"]) / kwargs["diffuseness"])

    return (-kwargs["depth"] * 4. * expon) / (1. + expon) ** 2.


def minnesota(x, chan, **kwargs):
    x2 = x ** 2
    exp0 = -kwargs["K0"] * x2
    exp1 = -kwargs["K1"] * x2

    # prevent overflow due to the exp(x**2) terms
    #if (np.abs(np.array([exp0, exp1])) > 8.).any():
    #    return 0.
    #else:
    return kwargs["V0"] * np.exp(exp0) + kwargs["V1"] * np.exp(exp1)


def chiral(x, chan, **kwargs):
    channel = chiralPot.Channel(S=chan.S, L=chan.L, LL=chan.LL, J=chan.J, channel=chan.channel)
    potId = kwargs["potId"]  # 213  # [order][cutoff][sfr cutoff]
    lecs = chiralPot.Lecs(kwargs["CS"], kwargs["CT"], kwargs["C1"], kwargs["C2"], kwargs["C3"], kwargs["C4"],
                          kwargs["C5"], kwargs["C6"], kwargs["C7"], kwargs["CNN"], kwargs["CPP"])
    return chiralPot.Vrlocal(x, potId, channel, lecs)

def chiral_affine(x, chan, **kwargs):
    channel = chiralPot.Channel(S=chan.S, L=chan.L, LL=chan.LL, J=chan.J, channel=chan.channel)
    potId = kwargs["potId"]  # 213  # [order][cutoff][sfr cutoff]
    ret = np.zeros(12, dtype=np.double)
    chiralPot.Vrlocal_affine(x, potId, channel, ret)
    return ret

def chiral_lec_trafo_matrix():
    """
    basis (CS, CT, C1, C2, ..., C7, CNN, CPP)
    """
    mat = np.zeros((11,11))
    mat[0, :2] = [1,1]
    mat[1, :2] = [1,-3]
    mat[2, 2:6] = [1,-3, 1, -3]
    mat[3, 7:9] = [1, -3]
    mat[4, 2:6] = [1,1, -3, -3]
    mat[5, 2:6] = 4*[1]
    mat[6, 2:6] = [1,-3, -3, 9]
    mat[7, 6] =  1/2
    mat[8, 7:9] = [1, 1]
    mat[9, 9] = 1
    mat[10, 10] = 1
    return mat

def optical(x, chan, **kwargs):
    kwargsV = {"depth": kwargs["V"], "radius": kwargs["R"], "diffuseness": kwargs["a"]}
    kwargsW = {"depth": kwargs["Vw"], "radius": kwargs["Rw"], "diffuseness": kwargs["aw"]}
    return complex(woodssaxon(x, chan, **kwargsV),
                   woodssaxon(x, chan, **kwargsW))

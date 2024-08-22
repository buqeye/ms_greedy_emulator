import numpy as np
from constants import *


class ScatteringExp:
    def __init__(self, E_MeV, potential, *, Z=0, A=1, Aproj=1):
        self.en_MeV = E_MeV
        self.potential = potential
        self.Z = Z
        self.A = A
        self.Aproj = Aproj

        self.en = self.en_MeV/hbarc
        self.mu = self._getMu()
        self.p = self._getP()

    @property
    def l(self):
        return self.potential.channel.L

    def _getMu(self):
        M1 = self.Aproj * mNeutron
        M2 = self.A * mNeutron
        return (M1 * M2) / (M1 + M2)

    def _getP(self):
        return np.sqrt(2.*self.mu*self.en)

    def __str__(self):
        return f"ScatteringExp A={self.Aproj}-->(A={self.A}, Z={self.Z}) at {self.en_MeV} MeV"


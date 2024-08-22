import numpy as np

# Defining constants
h_barc = 197.327  # Mev fm
h_barc_sq = h_barc*h_barc
two_mu = 0.5*(939.56563+938.27205)/h_barc  # fm^-1

def Blatt_Biedenharn_phase(tmm, tpm, tmp, tpp, k0, factor):
    tau_00 = -factor*0.5*np.pi*two_mu*k0*tmm
    tau_20 = -factor*0.5*np.pi*two_mu*k0*tpm
    tau_02 = -factor*0.5*np.pi*two_mu*k0*tmp
    tau_22 = -factor*0.5*np.pi*two_mu*k0*tpp

    S_00 = 1+2*tau_00*1j  # S--
    S_20 = 2*tau_20*1j  # S+-
    S_02 = 2*tau_02*1j  # S-+
    S_22 = 1+2*tau_22*1j  # S++

    y = -2*S_02
    x = S_22-S_00
    z = y/x

    epsi = -0.5*0.5*np.log((1+z*1j)/(1-z*1j))*1j
    #epsi = 0.5*np.arctan(y/x)

    S_plus = 0.5*(S_22+S_00 + ((S_22-S_00)/np.cos(2*abs(epsi))))

    S_minus = 0.5*(S_22+S_00 - ((S_22-S_00)/np.cos(2*abs(epsi))))

    eta_plus = np.abs(S_plus)

    eta_minus = np.abs(S_minus)

    delta_plus = 90*np.arctan2(S_plus.imag, S_plus.real)/np.pi

    delta_minus = 90*np.arctan2(S_minus.imag, S_minus.real)/np.pi

    return eta_plus, eta_minus, delta_plus, delta_minus, abs(epsi)


def Stapp_parameterization_phase(tmm, tpm, tmp, tpp, k0, factor):
    tau_00 = -factor*0.5*np.pi*two_mu*k0*tmm
    tau_20 = -factor*0.5*np.pi*two_mu*k0*tpm
    tau_02 = -factor*0.5*np.pi*two_mu*k0*tmp
    tau_22 = -factor*0.5*np.pi*two_mu*k0*tpp

    S_00 = 1+2*tau_00*1j # S--
    S_20 = 2*tau_20*1j # S+-
    S_02 = 2*tau_02*1j # S-+
    S_22 = 1+2*tau_22*1j # S++

    y = -0.5*1j*(S_20+S_02)
    x = np.sqrt(S_22*S_00)
    z = y/x

    epsi = -0.5*0.5*np.log((1+z*1j)/(1-z*1j))*1j

    S_pp = S_22/np.cos(2*abs(epsi))

    S_mm = S_00/np.cos(2*abs(epsi))

    eta_plus = np.abs(S_pp)

    eta_minus = np.abs(S_mm)

    delta_plus = 90*np.arctan2(S_pp.imag, S_pp.real)/np.pi

    delta_minus = 90*np.arctan2(S_mm.imag, S_mm.real)/np.pi

    return eta_plus, eta_minus, delta_plus, delta_minus, abs(epsi)

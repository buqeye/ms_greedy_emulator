import numpy as np
import multiple_E_couple_phase as MCP
import Coupled_phases as CP
import matplotlib.pyplot as plt


# Defining constants
h_barc = 197.3269804  # Mev fm
two_mu = 938.918747/h_barc  # fm^-1

# the following function helps you with plotting epsilon vs Energy
# and delta plus and delta minus vs Energy
def phase_plot(epsi, deltap, deltam, E_range, phase_convention):
    plt.plot(E_range, epsi, label='epsi')
    plt.xlabel("E_cm (MeV)")
    plt.ylabel("degree ")
    plt.legend()
    plt.grid()
    plt.show()

    plt.plot(E_range, deltap, label='delta_p')
    plt.plot(E_range, deltam, label='delta_m')
    #plt.axhline(180, color='blue')
    #plt.axvline(0, color='blue')
    plt.xlabel("E_cm (MeV)")
    plt.ylabel("degree ")
    if phase_convention == CP.Stapp_parameterization_phase :
        plt.title(f" Stapp Parameterization phase plot")
    else:
        plt.title(f" Blatt Biedenharn phase plot")
    plt.legend()
    plt.grid()
    plt.show()

# The following function helps you with writing the datas in the file
def write_data(mapp, np1, np2, N, p1, p2, p3, phase_convention1, phase_convention2,
               E_range, eta_p1, eta_m1, epsi_l1, delta_p1, delta_m1, eta_p2, eta_m2, epsi_l2, delta_p2, delta_m2):

    # set up file and write to file
    file1 = open("./phases_file.txt", "w") # creates and opens the file in write mode

    file1.write(f"map : {mapp} \n")
    file1.write(f"np1 : {np1:.1f}, np2 : {np2:.1f}, N : {N}\n")
    file1.write(f"p1 : {p1:.1f}, p2 : {p2:.1f}, p3 : {p3:.1f}\n")

    file1.write(f"\n")
    file1.write(f"Phase Convention : {phase_convention1} \n")
    file1.write("Energy (MeV)  eta_plus eta_minus epsilon (deg) delta_plus (deg) delta_minus (deg)  \n")

    for i in range(len(E_range)):
        etap = eta_p1[i]
        etam = eta_m1[i]
        ener = E_range[i]
        deltap = delta_p1[i]
        deltam = delta_m1[i]
        epsi = epsi_l1[i]

        file1.write(f"{ener:.1f}, {etap:.6f},  {etam:.6f}, {epsi:.6f},  {deltap:.6f}, {deltam:.6f}\n")


    file1.write(f"\n")
    file1.write(f"\n")
    file1.write(f"Phase Convention : {phase_convention2} \n")
    file1.write("Energy (MeV)  eta_plus eta_minus epsilon (deg) delta_plus (deg) delta_minus (deg)  \n")


    for i in range(len(E_range)):
        etap = eta_p2[i]
        etam = eta_m2[i]
        ener = E_range[i]
        deltap = delta_p2[i]
        deltam = delta_m2[i]
        epsi = epsi_l2[i]

        file1.write(f"{ener:.1f}, {etap:.6f},  {etam:.6f}, {epsi:.6f},  {deltap:.6f}, {deltam:.6f}\n")

    # close file
    file1.close()



mapp = 'trns'
E_range = np.logspace(-6, 2, 200) # Mev
np1 = 20
np2 = 12
N = np1 + np2
p1 = 2.5  # fm^-1
p2 = 6.0  # fm^-1
p3 = 30.0  # fm^-1
c = 5.0  # fm^-1
phase_convention1 = CP.Stapp_parameterization_phase
phase_convention2 = CP.Blatt_Biedenharn_phase

#eta_p1, eta_m1, epsi_l1, delta_p1, delta_m1 = MCP.call_coupled_tmatrix(N, mapp, p1, p2, p3, np1, np2, c, E_range,
#                                                                       phase_convention1)
eta_p2, eta_m2, epsi_l2, delta_p2, delta_m2 = MCP.call_coupled_tmatrix(N, mapp, p1, p2, p3, np1, np2, c, E_range,
                                                                       phase_convention2)

#phase_plot(epsi_l1, delta_p1, delta_m1, E_range, phase_convention1)
phase_plot(epsi_l2, delta_p2, delta_m2, E_range, phase_convention2)

#write_data(mapp, np1, np2, N, p1, p2, p3, phase_convention1, phase_convention2,
#           E_range, eta_p1, eta_m1, epsi_l1, delta_p1, delta_m1, eta_p2, eta_m2, epsi_l2, delta_p2, delta_m2)

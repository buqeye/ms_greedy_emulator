import chiralPot
import numpy as np
import matplotlib.pyplot as plt

h_barc = 197.3269804  # Mev fm
h_barc_sq = h_barc*h_barc

channels = [0]
minval = 0
maxval = 9
minval1 = 0
maxval1 = 9
k = np.arange(minval, maxval, 0.1)
k1 = np.arange(minval1, maxval1, 0.1)
vmm = np.zeros((len(k), len(k)))
vmp = np.zeros((len(k1), len(k1)))
vpm = np.zeros((len(k1), len(k1)))
vpp = np.zeros((len(k), len(k)))
c = 0
s = 1
for l in range(1, 2):
    m = l - 1
    p = l + 1
    for channel in channels:
        c += 1
        for i in range(len(k)):
            for j in range(len(k)):
                vmm[i, j] = chiralPot.V0(k[i], k[j], 213, s, m, m, l, channel)*h_barc_sq
                vpp[i, j] = chiralPot.V0(k[i], k[j], 213, s, p, p, l, channel)*h_barc_sq

                # vmm[j,i] =
                # vpm[j,i] = vmp[i,j]
                # vmp[j,i] = vpm[i,j]
                # vpp[j,i] = vpp[i,j]

        for i1 in range(len(k1)):
            for j1 in range(len(k1)):
                vmp[i1, j1] = chiralPot.V0(k1[i1], k1[j1], 213, s, m, p, l, channel)*h_barc_sq
                vpm[i1, j1] = chiralPot.V0(k1[i1], k1[j1], 213, s, p, m, l, channel)*h_barc_sq

        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        fig.suptitle(f' S = {s}, J = {l}, channel = {channel}', fontsize=16)

        # decorate plot
        axs[0, 0].set_ylabel("k' (fm$^{-1}$)", fontsize=12)
        axs[0, 0].set_title('v_mm (fm$^2$)')

        # set up a figure twice as wide as it is tall

        c1 = axs[0, 0].imshow(vmm, extent=[minval, maxval, minval, maxval], cmap="inferno", origin='lower')
        fig.colorbar(c1, ax=axs[0, 0], shrink=1, aspect=20)


        # decorate plot
        axs[0, 1].set_ylabel("k' (fm$^{-1}$)", fontsize=12)
        axs[0, 1].set_title(' v_mp (fm$^2$) ')

        # set up a figure twice as wide as it is tall

        c2 = axs[0, 1].imshow(vmp, extent=[minval1, maxval1, minval1, maxval1], cmap="inferno", origin='lower')
        fig.colorbar(c2, ax=axs[0, 1], shrink=1, aspect=20)


        # decorate plot
        axs[1, 0].set_xlabel('k (fm$^{-1}$)', fontsize=12)
        axs[1, 0].set_ylabel("k' (fm$^{-1}$)", fontsize=12)
        axs[1, 0].set_title('v_pm (fm$^2$)')

        # set up a figure twice as wide as it is tall

        c3 = axs[1, 0].imshow(vpm, extent=[minval1, maxval1, minval1, maxval1], cmap="inferno", origin='lower')
        fig.colorbar(c3, ax=axs[1, 0], shrink=1, aspect=20)



        # decorate plot
        axs[1, 1].set_xlabel('k (fm$^{-1}$)', fontsize=12)
        axs[1, 1].set_ylabel("k' (fm$^{-1}$)", fontsize=12)
        axs[1, 1].set_title('v_pp (fm$^2$)')

        # set up a figure twice as wide as it is tall

        c4 = axs[1, 1].imshow(vpp, extent=[minval, maxval, minval, maxval], cmap="inferno", origin='lower')
        fig.colorbar(c4, ax=axs[1, 1], shrink=1, aspect=20)

        plt.savefig(f'figure_{c}.png')
        plt.show()


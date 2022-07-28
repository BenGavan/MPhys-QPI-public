import matplotlib.pyplot as plt
from files import *

# TODO: Decouple data generation and plotting
# TODO: https://eigen.tuxfamily.org/index.php?title=Main_Page

'''
Notes:
 - All matrices and vectors should conform to the standard convension where <psi| = (...)
   - when introducing a new matrix, if not automatically in the format, change as soon as defined.
'''


pauli = np.array((
    ((0, 1), (1, 0)),
    ((0, -1j), (1j, 0)),
    ((1, 0), (0, -1))
))

'''
Calculates the bare Green's function G_0(k, w) = \sigma_n \frac{}{omega + i\eta - \epsilon}
 - psi = eigenstate (2*2 matrix)
'''


def generate_axes(x_n, y_n, x_length, y_length):
    '''

    :param x_n: # pixes along x axis
    :param y_n: # pixes along y axis
    :param x_length:
    :param y_length:
    :return:
    '''
    xs = np.linspace(-x_length / 2, x_length / 2, x_n)
    ys = np.linspace(-y_length / 2, y_length / 2, y_n)

    return xs, ys


def generate_real_axis(xn, yn, x_length, y_length):
    xs = np.linspace(-x_length / 2, x_length / 2, xn)
    ys = np.linspace(-y_length / 2, y_length / 2, yn)

    return xs, ys


def plot_bands():
    from hamiltonian_36_band import extract_eigenstates_from_hamiltonian
    from hamiltonian_36_band import calculate_36_band_H_k_from_supercell
    from hamiltonian_36_band import get_supercell_36band_hamiltonian
    def calculate_eigenvalues_for_ks(ks):
        xs = []

        dk_vec = ks[-1] - ks[0]
        dk = np.sqrt(dk_vec[0] ** 2 + dk_vec[1] ** 2) / len(ks)

        eigenvalues_k = []

        for i in range(len(ks)):
            xs.append(dk * i)
            h_k = calculate_36_band_H_k_from_supercell(H_36band_arr_R, rs_supercell, ks[i][0], ks[i][1])
            _, eigenvalues = extract_eigenstates_from_hamiltonian(h_k)
            eigenvalues_k.append(eigenvalues)

        return xs, eigenvalues_k

    H_36band_arr_R, rs_supercell = get_supercell_36band_hamiltonian()

    print(len(H_36band_arr_R))

    n = 2000

    M = Params.M_sc
    K = Params.K_sc

    # Gamma -> M
    ks_Gam_M = np.array([Params.G + i * M for i in np.linspace(0, 1, n)])
    xs_Gam_M, evals_Gam_M = calculate_eigenvalues_for_ks(ks_Gam_M)

    # M -> K
    ks_M_K = np.array([M + (K - M) * i for i in np.linspace(0, 1, n)])
    xs_M_K, evals_M_K= calculate_eigenvalues_for_ks(ks_M_K)
    xs_M_K = xs_M_K + xs_Gam_M[-1]

    # K -> Gamma
    ks_K_G = np.array([K - K * i for i in np.linspace(0, 1, n)])
    xs_K_G, evals_K_G = calculate_eigenvalues_for_ks(ks_K_G)
    xs_K_G = xs_K_G + xs_M_K[-1]

    # Fermi-energy
    E_f = Params.E_f  # eV
    dE_f = Params.delta_E_f  # 0 meV
    E_f = E_f + dE_f

    min_e = min(evals_K_G[0]) - E_f
    max_e = max(evals_K_G[0]) - E_f

    plt.figure(figsize=(3*2.5, 6))

    for i in range(36):
        print('plotting \"band\" {}'.format(i))
        xs = np.array([])
        eigenvalues = np.array([])

        xs = np.append(xs, xs_Gam_M)
        for es in evals_Gam_M:
            eigenvalues = np.append(eigenvalues, es[i] - E_f)

        xs = np.append(xs, xs_M_K)
        for es in evals_M_K:
            eigenvalues = np.append(eigenvalues, es[i] - E_f)

        xs = np.append(xs, xs_K_G)
        for es in evals_K_G:
            eigenvalues = np.append(eigenvalues, es[i] - E_f)

        # plt.plot(xs, eigenvalues, marker='.', markersize=.5, linewidth=0, color='black')
        plt.plot(xs, eigenvalues, linewidth=.5, color='black')
        # print(eigenvalues)

        min_e = min(np.min(eigenvalues), min_e)
        max_e = max(np.max(eigenvalues), max_e)

    e_range_min = min_e - 0.05
    e_range_max = max_e + 0.05

    # plt.plot([0, xs_K_G[-1]], [E_f, E_f], color='gray', label='$E_f =${} eV'.format(E_f), linewidth=1)
    # plt.plot([0, xs_K_G[-1]], [0, 0], color='gray', label='$E_f =${} eV'.format(E_f), linewidth=1)
    x = [0, xs_Gam_M[-1], xs_M_K[-1], xs_K_G[-1]]
    print('x = ', x)
    # labels = ['$\overline{\Gamma}$', '$\overline{M}$', '$\overline{K}$', '$\overline{\Gamma}$']
    labels = ['$\overline{\Gamma}$', '$\overline{M}$', '$\overline{K}$', '$\overline{\Gamma}$']
    plt.xticks(x, labels, rotation='horizontal')
    plt.xlim(x[0], x[-1])
    plt.ylim(e_range_min, e_range_max)

    # add horizontal bars
    plt.plot([x[1], x[1]], [e_range_min, e_range_max], color='#000000', linewidth=0.5)
    plt.plot([x[2], x[2]], [e_range_min, e_range_max], color='#000000', linewidth=0.5)

    plt.ylabel('$E-E_f$ (eV)')
    plt.tight_layout()
    # filepath = FilePaths.bands_plots + 'bands_{}.png'.format(i)
    filepath = FilePaths.bands_plots + 'band_plot_3_{}.pdf'.format(get_version_number())
    print('filepath: \n', filepath)
    plt.savefig(filepath, bbox_inches='tight')
    plt.savefig(filepath + '.png', bbox_inches='tight', dpi=300)
    plt.close()
        # plt.show()


def plot_contours_seperate():
    from hamiltonian_36_band import get_supercell_36band_hamiltonian
    from hamiltonian_36_band import calculate_36_band_H_k_from_supercell
    from hamiltonian_36_band import extract_eigenstates_from_hamiltonian
    H_R_supercell_array, supercell_Rs = get_supercell_36band_hamiltonian()

    # Fermi-energy
    E_f = Params.E_f  # eV
    dE_f = Params.delta_E_f

    E_f = E_f + dE_f

    kxn = 200
    kyn = 200

    kx_length = 1
    ky_length = 1

    kxs, kys = generate_axes(kxn, kyn, kx_length, ky_length)

    eigenvalues_k = np.zeros((kxn, kyn), dtype=type([1.]))

    for i in range(kxn):
        for j in range(kyn):
            kx = kxs[i]
            ky = kys[j]

            h_k = calculate_36_band_H_k_from_supercell(H_R_supercell_array, supercell_Rs, kx, ky)
            _, eigenvalues = extract_eigenstates_from_hamiltonian(h_k)
            eigenvalues_k[i, j] = eigenvalues

        if i % 20 == 0:
            print('finished row: {}'.format(i))

    for i in range(36):
        fig, ax = plt.subplots()

        zs = np.zeros((kxn, kyn), dtype=float)

        for j in range(kxn):
            for k in range(kyn):
                zs[j, k] = eigenvalues_k[j, k][i]

        ax.contour(kxs, kys, zs, levels=[E_f])

        # Plot BZ (new)
        G = Params.G_sc
        K = Params.K_sc
        M = Params.M_sc

        cs = [K, (K[0], -K[1]), -2 * K + 2 * M, (-K[0], -K[1]), (-K[0], K[1]), 2 * K - 2 * M, K]
        xs_brillouin = np.array([])
        ys_brillouin = np.array([])

        for coord in cs:
            xs_brillouin = np.append(xs_brillouin, coord[0])
            ys_brillouin = np.append(ys_brillouin, coord[1])

        ax.plot(xs_brillouin, ys_brillouin, color='gray', linestyle='dashed')  # , zorder=1

        ax.plot([G[0], M[0], K[0]], [G[1], M[1], K[1]])

        # can move out of 'for i in range()' to plot all bands on one
        filepath = FilePaths.contour_plots + 'kxn={}_eigh_contour_band={}.png'.format(kxn, i)
        fig.gca().set_aspect('equal', adjustable='box')
        fig.tight_layout()
        fig.savefig(filepath, rasterized=True, bbox_inches='tight')
        # fig.show()
        plt.close(fig)
        print('Points plot:\n{}'.format(filepath))


def plot_contours_on_one():
    from hamiltonian_36_band import get_supercell_36band_hamiltonian
    from hamiltonian_36_band import calculate_36_band_H_k_from_supercell
    from hamiltonian_36_band import extract_eigenstates_from_hamiltonian

    H_R_supercell_array, supercell_Rs = get_supercell_36band_hamiltonian()

    # Fermi-energy
    E_f = Params.E_f  # eV
    dE_f = Params.delta_E_f

    E_f = E_f + dE_f

    kxn = 100
    kyn = 100

    a_0 = 3.44
    kx_length = 3 * np.pi / a_0
    ky_length = 3 * np.pi / a_0

    kxs, kys = generate_axes(kxn, kyn, kx_length, ky_length)

    eigenvalues_k = np.zeros((kxn, kyn), dtype=type([1.]))

    for i in range(kxn):
        for j in range(kyn):
            kx = kxs[i]
            ky = kys[j]

            h_k = calculate_36_band_H_k_from_supercell(H_R_supercell_array, supercell_Rs, kx, ky)
            _, eigenvalues = extract_eigenstates_from_hamiltonian(h_k)
            eigenvalues_k[i, j] = eigenvalues - E_f

        if i % 20 == 0:
            print('finished row: {}'.format(i))

    fig, ax = plt.subplots()

    for i in range(36):
        zs = np.zeros((kxn, kyn), dtype=float)

        for j in range(kxn):
            for k in range(kyn):
                zs[j, k] = eigenvalues_k[j, k][i].real

        ax.contour(kxs, kys, zs.transpose(), levels=[0], linewidths=[0.5], linestyles=['solid'], colors=['black'])

    # Plot BZ (new)
    G = Params.G
    K = Params.K_sc
    M = Params.M_sc
    cs = [K, (K[0], -K[1]), -2 * K + 2 * M, (-K[0], -K[1]), (-K[0], K[1]), 2 * K - 2 * M, K]
    xs_brillouin = np.array([])
    ys_brillouin = np.array([])

    for coord in cs:
        xs_brillouin = np.append(xs_brillouin, coord[0])
        ys_brillouin = np.append(ys_brillouin, coord[1])
    #
    ax.plot(xs_brillouin, ys_brillouin, color='red', linestyle='solid', linewidth=1)  # , zorder=1

    G = Params.G
    K = Params.K
    M = Params.M

    cs = [K, (K[0], -K[1]), -2 * K + 2 * M, (-K[0], -K[1]), (-K[0], K[1]), 2 * K - 2 * M, K]
    xs_brillouin = np.array([])
    ys_brillouin = np.array([])

    for coord in cs:
        xs_brillouin = np.append(xs_brillouin, coord[0])
        ys_brillouin = np.append(ys_brillouin, coord[1])

    ax.plot(xs_brillouin, ys_brillouin, color='blue', linestyle='solid', linewidth=1)  # , zorder=1

    # ax.plot([G[0], M[0], K[0], G[0]], [G[1], M[1], K[1], G[1]])

    ax.set_xlabel('$k_x \:(\AA)$')
    ax.set_ylabel('$k_y \:(\AA)$')

    # can move out of 'for i in range()' to plot all bands on one
    filepath = FilePaths.contour_plots + 'kxn={}_contour_{}.pdf'.format(kxn, get_version_number())
    fig.gca().set_aspect('equal', adjustable='box')
    fig.tight_layout()
    fig.savefig(filepath, bbox_inches='tight')
    fig.savefig(filepath + '.png', bbox_inches='tight', dpi=300)
    # fig.show()
    plt.close(fig)
    print('Points plot:\n{}'.format(filepath))


def main():
    plot_bands()
    # plot_contours_seperate()
    plot_contours_on_one()
    # plot_bands()



def get_version_number():
    return '_1.80.2_eig_dEf={}'.format(Params.delta_E_f)


if __name__ == '__main__':
    main()

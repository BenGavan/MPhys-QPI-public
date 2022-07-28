import matplotlib.pyplot as plt
import numpy as np

from files import *
from hamiltonian_4band import *

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
    from hamiltonian_4band import extract_eigenstates_from_hamiltonian
    def calculate_eigenvalues_for_ks(ks):
        xs = []

        dk_vec = ks[-1] - ks[0]
        dk = np.sqrt(dk_vec[0] ** 2 + dk_vec[1] ** 2) / len(ks)

        eigenvalues_k = []

        for i in range(len(ks)):
            xs.append(dk * i)
            h_k = calculate_4_band_H_k(H_arr_R, rs, weights, ks[i][0], ks[i][1])
            _, eigenvalues = extract_eigenstates_from_hamiltonian(h_k)
            eigenvalues_k.append(eigenvalues)

        return xs, eigenvalues_k

    H_arr_R, rs, weights = get_4_band_hamiltonian(FilePaths.hamiltonian_NbSe2_4band_data)

    n = 2000

    M = Params.M
    K = Params.K

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

    for i in range(4):
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

        min_e = min(np.min(eigenvalues), min_e)
        max_e = max(np.max(eigenvalues), max_e)

    e_range_min = min_e - 0.05
    e_range_max = max_e + 0.05

    # plt.plot([0, xs_K_G[-1]], [E_f, E_f], color='gray', label='$E_f =${} eV'.format(E_f), linewidth=1)
    # plt.plot([0, xs_K_G[-1]], [0, 0], color='gray', label='$E_f =${} eV'.format(E_f), linewidth=1)
    x = [0, xs_Gam_M[-1], xs_M_K[-1], xs_K_G[-1]]
    print('x = ', x)
    # labels = ['$\overline{\Gamma}$', '$\overline{M}$', '$\overline{K}$', '$\overline{\Gamma}$']
    labels = ['$\Gamma$', '$M$', '$K$', '$\Gamma$']
    plt.xticks(x, labels, rotation='horizontal')
    plt.xlim(x[0], x[-1])
    plt.ylim(e_range_min, e_range_max)

    # add horizontal bars
    plt.plot([x[1], x[1]], [e_range_min, e_range_max], color='#000000', linewidth=0.5)
    plt.plot([x[2], x[2]], [e_range_min, e_range_max], color='#000000', linewidth=0.5)

    plt.ylabel('$E-E_f$ (eV)')
    plt.tight_layout()
    # filepath = FilePaths.bands_plots + 'bands_{}.png'.format(i)
    filepath = FilePaths.bands_plots + 'band_plot_{}.pdf'.format(get_version_number())
    print('filepath: \n', filepath)
    plt.savefig(filepath, bbox_inches='tight')
    plt.savefig(filepath + '.png', bbox_inches='tight', dpi=300)
    plt.close()
        # plt.show()


def plot_bands_with_weighting():
    from hamiltonian_4band import extract_eigenstates_from_hamiltonian
    def calculate_eigenvalues_for_ks(ks):
        xs = []

        dk_vec = ks[-1] - ks[0]
        dk = np.sqrt(dk_vec[0] ** 2 + dk_vec[1] ** 2) / len(ks)

        eigenvalues_k = []
        basis_weights = []

        for i in range(len(ks)):
            xs.append(dk * i)
            h_k = calculate_4_band_H_k(H_arr_R, rs, weights, ks[i][0], ks[i][1])
            eigenvectors, eigenvalues = extract_eigenstates_from_hamiltonian(h_k)
            print(eigenvectors)
            print('----')
            print(eigenvectors[0])
            print('----')
            print(eigenvectors[1])
            exit()
            eigenvalues_k.append(eigenvalues)
            # basis weights
            w = [0] * len(eigenvalues)
            # print(w)
            # print(eigenvectors)
            # print(eigenvectors[0][0, 0])
            for j in range(len(eigenvalues)):
                # inplane
                # w[j] = (eigenvectors[j][0, 0] * np.conjugate(eigenvectors[j][0, 0])).real
                # w[j] += (eigenvectors[j][2, 0] * np.conjugate(eigenvectors[j][2, 0])).real
                # outplane
                w[j] = (eigenvectors[j][1, 0] * np.conjugate(eigenvectors[j][1, 0])).real
                w[j] += (eigenvectors[j][3, 0] * np.conjugate(eigenvectors[j][3, 0])).real
                # spin-up
                # w[j] = (eigenvectors[j][0, 0] * np.conjugate(eigenvectors[j][0, 0])).real
                # w[j] += (eigenvectors[j][1, 0] * np.conjugate(eigenvectors[j][1, 0])).real
                # spin-down
                # w[j] = (eigenvectors[j][2, 0] * np.conjugate(eigenvectors[j][2, 0])).real
                # w[j] += (eigenvectors[j][3, 0] * np.conjugate(eigenvectors[j][3, 0])).real
            basis_weights.append(w)

        return xs, eigenvalues_k, basis_weights

    H_arr_R, rs, weights = get_4_band_hamiltonian(FilePaths.hamiltonian_NbSe2_4band_data)

    n = 2000

    M = Params.M
    K = Params.K
    print(M)
    print(K)

    # rotate
    M_mat = np.matrix(M)
    K_mat = np.matrix(K)

    theta = np.pi / 3
    # theta = 0
    R_z = np.matrix(((np.cos(theta), -np.sin(theta)),
                     (np.sin(theta), np.cos(theta))))

    M_mat = R_z * M_mat.transpose()
    K_mat = R_z * K_mat.transpose()

    M = np.array([M_mat[0, 0], M_mat[1, 0]])
    K = np.array([K_mat[0, 0], K_mat[1, 0]])
    # print(M)
    # print(K)
    # exit()

    # Gamma -> M
    ks_Gam_M = np.array([Params.G + i * M for i in np.linspace(0, 1, n)])
    xs_Gam_M, evals_Gam_M, basis_weights_Gam_M = calculate_eigenvalues_for_ks(ks_Gam_M)

    # M -> K
    ks_M_K = np.array([M + (K - M) * i for i in np.linspace(0, 1, n)])
    xs_M_K, evals_M_K, basis_weights_M_K = calculate_eigenvalues_for_ks(ks_M_K)
    xs_M_K = xs_M_K + xs_Gam_M[-1]

    # K -> Gamma
    ks_K_G = np.array([K - K * i for i in np.linspace(0, 1, n)])
    xs_K_G, evals_K_G, basis_weights_K_G = calculate_eigenvalues_for_ks(ks_K_G)
    xs_K_G = xs_K_G + xs_M_K[-1]

    # Fermi-energy
    E_f = Params.E_f  # eV
    dE_f = Params.delta_E_f  # 0 meV
    E_f = E_f + dE_f

    min_e = min(evals_K_G[0]) - E_f
    max_e = max(evals_K_G[0]) - E_f

    # plt.figure(figsize=(3*2.5, 6))

    fig, ax = plt.subplots(figsize=(3*2.5, 6))
    from matplotlib.collections import LineCollection

    for i in range(4):
        xs = np.array([])
        eigenvalues = np.array([])
        ws = np.array([])

        xs = np.append(xs, xs_Gam_M)
        for es in evals_Gam_M:
            eigenvalues = np.append(eigenvalues, es[i] - E_f)
        for w in basis_weights_Gam_M:
            ws = np.append(ws, w[i])

        xs = np.append(xs, xs_M_K)
        for es in evals_M_K:
            eigenvalues = np.append(eigenvalues, es[i] - E_f)
        for w in basis_weights_M_K:
            ws = np.append(ws, w[i])

        xs = np.append(xs, xs_K_G)
        for es in evals_K_G:
            eigenvalues = np.append(eigenvalues, es[i] - E_f)
        for w in basis_weights_K_G:
            ws = np.append(ws, w[i])

        # plt.plot(xs, eigenvalues, marker='.', markersize=.5, linewidth=0, color='black')
        points = np.array([xs, eigenvalues]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = LineCollection(segments, cmap='winter')
        # Set the values used for colormapping
        lc.set_array(ws)
        lc.set_linewidth(2)
        line = ax.add_collection(lc)
        if i == 0:
            fig.colorbar(line, ax=ax)

        min_e = min(np.min(eigenvalues), min_e)
        max_e = max(np.max(eigenvalues), max_e)

    e_range_min = min_e - 0.05
    e_range_max = max_e + 0.05

    # plt.plot([0, xs_K_G[-1]], [E_f, E_f], color='gray', label='$E_f =${} eV'.format(E_f), linewidth=1)
    # plt.plot([0, xs_K_G[-1]], [0, 0], color='gray', label='$E_f =${} eV'.format(E_f), linewidth=1)
    x = [0, xs_Gam_M[-1], xs_M_K[-1], xs_K_G[-1]]
    print('x = ', x)
    # labels = ['$\overline{\Gamma}$', '$\overline{M}$', '$\overline{K}$', '$\overline{\Gamma}$']
    labels = ['$\Gamma$', '$M$', '$K$', '$\Gamma$']
    ax.set_xticks(x, labels, rotation='horizontal')
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(e_range_min, e_range_max)

    # add horizontal bars
    ax.plot([x[1], x[1]], [e_range_min, e_range_max], color='#000000', linewidth=0.5)
    ax.plot([x[2], x[2]], [e_range_min, e_range_max], color='#000000', linewidth=0.5)

    ax.set_ylabel('$E-E_f$ (eV)')
    fig.tight_layout()
    # filepath = FilePaths.bands_plots + 'bands_{}.png'.format(i)
    filepath = FilePaths.bands_plots + 'band_plot_weighted_outpane_{}_theta={}.pdf'.format(get_version_number(), theta)
    print('filepath: \n', filepath)
    fig.savefig(filepath, bbox_inches='tight')
    fig.savefig(filepath + '.png', bbox_inches='tight', dpi=300)
    plt.close()
        # plt.show()


def plot_bands_with_spin_texture():
    from hamiltonian_4band import extract_eigenstates_from_hamiltonian
    def calculate_eigenvalues_for_ks(ks):
        xs = []

        dk_vec = ks[-1] - ks[0]
        dk = np.sqrt(dk_vec[0] ** 2 + dk_vec[1] ** 2) / len(ks)

        eigenvalues_k = []
        basis_weights = []

        for i in range(len(ks)):
            xs.append(dk * i)
            h_k = calculate_4_band_H_k(H_arr_R, rs, weights, ks[i][0], ks[i][1])
            eigenvectors, eigenvalues = extract_eigenstates_from_hamiltonian(h_k)
            eigenvalues_k.append(eigenvalues)

            w = []

            for j in range(len(eigenvalues)):
                # calculate spin operator expectation value

                p_z = np.zeros((4, 4), dtype=complex)
                p_z[0, 0] = 1
                p_z[1, 1] = 1
                p_z[2, 2] = -1
                p_z[3, 3] = -1

                w.append((eigenvectors[j].getH() * p_z * eigenvectors[j])[0, 0].real)
            basis_weights.append(w)

        return xs, eigenvalues_k, basis_weights

    H_arr_R, rs, weights = get_4_band_hamiltonian(FilePaths.hamiltonian_NbSe2_4band_data)

    n = 2000

    M = Params.M
    K = Params.K
    print(M)
    print(K)

    # rotate
    M_mat = np.matrix(M)
    K_mat = np.matrix(K)

    theta = -np.pi / 3
    # theta = 0
    R_z = np.matrix(((np.cos(theta), -np.sin(theta)),
                     (np.sin(theta), np.cos(theta))))

    M_mat = R_z * M_mat.transpose()
    K_mat = R_z * K_mat.transpose()

    M = np.array([M_mat[0, 0], M_mat[1, 0]])
    K = np.array([K_mat[0, 0], K_mat[1, 0]])
    # print(M)
    # print(K)
    # exit()

    # Gamma -> M
    ks_Gam_M = np.array([Params.G + i * M for i in np.linspace(0, 1, n)])
    xs_Gam_M, evals_Gam_M, basis_weights_Gam_M = calculate_eigenvalues_for_ks(ks_Gam_M)

    # M -> K
    ks_M_K = np.array([M + (K - M) * i for i in np.linspace(0, 1, n)])
    xs_M_K, evals_M_K, basis_weights_M_K = calculate_eigenvalues_for_ks(ks_M_K)
    xs_M_K = xs_M_K + xs_Gam_M[-1]

    # K -> Gamma
    ks_K_G = np.array([K - K * i for i in np.linspace(0, 1, n)])
    xs_K_G, evals_K_G, basis_weights_K_G = calculate_eigenvalues_for_ks(ks_K_G)
    xs_K_G = xs_K_G + xs_M_K[-1]

    # Fermi-energy
    E_f = Params.E_f  # eV
    dE_f = Params.delta_E_f  # 0 meV
    E_f = E_f + dE_f

    min_e = min(evals_K_G[0]) - E_f
    max_e = max(evals_K_G[0]) - E_f

    # plt.figure(figsize=(3*2.5, 6))

    fig, ax = plt.subplots(figsize=(3*2.5, 6))
    from matplotlib.collections import LineCollection

    for i in range(4):
        xs = np.array([])
        eigenvalues = np.array([])
        ws = np.array([])

        xs = np.append(xs, xs_Gam_M)
        for es in evals_Gam_M:
            eigenvalues = np.append(eigenvalues, es[i] - E_f)
        for w in basis_weights_Gam_M:
            ws = np.append(ws, w[i])

        xs = np.append(xs, xs_M_K)
        for es in evals_M_K:
            eigenvalues = np.append(eigenvalues, es[i] - E_f)
        for w in basis_weights_M_K:
            ws = np.append(ws, w[i])

        xs = np.append(xs, xs_K_G)
        for es in evals_K_G:
            eigenvalues = np.append(eigenvalues, es[i] - E_f)
        for w in basis_weights_K_G:
            ws = np.append(ws, w[i])

        # plt.plot(xs, eigenvalues, marker='.', markersize=.5, linewidth=0, color='black')
        points = np.array([xs, eigenvalues]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = LineCollection(segments, cmap='winter')
        # Set the values used for colormapping
        print(ws)
        lc.set_array(ws)
        lc.set_linewidth(2)
        line = ax.add_collection(lc)
        if i == 0:
            fig.colorbar(line, ax=ax)

        min_e = min(np.min(eigenvalues), min_e)
        max_e = max(np.max(eigenvalues), max_e)

    e_range_min = min_e - 0.05
    e_range_max = max_e + 0.05

    # plt.plot([0, xs_K_G[-1]], [E_f, E_f], color='gray', label='$E_f =${} eV'.format(E_f), linewidth=1)
    # plt.plot([0, xs_K_G[-1]], [0, 0], color='gray', label='$E_f =${} eV'.format(E_f), linewidth=1)
    x = [0, xs_Gam_M[-1], xs_M_K[-1], xs_K_G[-1]]
    print('x = ', x)
    # labels = ['$\overline{\Gamma}$', '$\overline{M}$', '$\overline{K}$', '$\overline{\Gamma}$']
    labels = ['$\Gamma$', '$M$', '$K$', '$\Gamma$']
    ax.set_xticks(x, labels, rotation='horizontal')
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(e_range_min, e_range_max)

    # add horizontal bars
    ax.plot([x[1], x[1]], [e_range_min, e_range_max], color='#000000', linewidth=0.5)
    ax.plot([x[2], x[2]], [e_range_min, e_range_max], color='#000000', linewidth=0.5)

    ax.set_ylabel('$E-E_f$ (eV)')
    fig.tight_layout()
    # filepath = FilePaths.bands_plots + 'bands_{}.png'.format(i)
    filepath = FilePaths.bands_plots + 'band_plot_weighted_spin_texture_{}_theta={}.pdf'.format(get_version_number(), theta)
    print('filepath: \n', filepath)
    fig.savefig(filepath, bbox_inches='tight')
    fig.savefig(filepath + '.png', bbox_inches='tight', dpi=300)
    plt.close()



def plot_contours_seperate():
    H_arr_R, rs, weights = get_4_band_hamiltonian(FilePaths.hamiltonian_NbSe2_4band_data)

    # Fermi-energy
    E_f = Params.E_f  # eV
    dE_f = 0 * pow(10, -3)

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

            h_k = calculate_4_band_H_k(H_arr_R, rs, weights, kx, ky)
            _, eigenvalues_k[i, j] = extract_eigenstates_from_hamiltonian(h_k)

        if i % 20 == 0:
            print('finished row: {}'.format(i))

    for i in range(4):
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
    H_arr_R, rs, weights = get_4_band_hamiltonian(FilePaths.hamiltonian_NbSe2_4band_data)

    # Fermi-energy
    E_f = Params.E_f + 2.55  # eV
    dE_f = Params.delta_E_f

    E_f = E_f + dE_f

    kxn = 200
    kyn = 200

    a_0 = 3.44
    kx_length = 10 * np.pi / a_0
    ky_length = 10 * np.pi / a_0
    # kx_length = 4/3
    # ky_length = 4/3

    kxs, kys = generate_axes(kxn, kyn, kx_length, ky_length)

    eigenvalues_k = np.zeros((kxn, kyn), dtype=type([1.]))

    for i in range(kxn):
        for j in range(kyn):
            kx = kxs[i]
            ky = kys[j]

            h_k = calculate_4_band_H_k(H_arr_R, rs, weights, kx, ky)
            _, evs = extract_eigenstates_from_hamiltonian(h_k)
            eigenvalues_k[i, j] = evs - E_f

        if i % 20 == 0:
            print('finished row: {}'.format(i))

    fig, ax = plt.subplots()
    from plotting import plot_surface

    # zss = []
    #
    for i in range(4):
        zs = np.zeros((kxn, kyn), dtype=float)

        for j in range(kxn):
            for k in range(kyn):
                zs[j, k] = eigenvalues_k[j, k][i]

        ax.contour(kxs, kys, zs.transpose(), levels=[0], linewidths=[0.5], linestyles=['solid'], colors=['black'])
        # zss.append(zs)
    # plot_surface(kxs, kys, zss[:-2])
    # plot_surface(kxs, kys, zss[2:])
    #     , colors=['black']

    # Plot BZ (new)
    G = Params.G
    K = Params.K / 3
    M = Params.M / 3
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
    filepath = FilePaths.contour_plots + 'kxn={}_contour_Ef={}_{}.pdf'.format(kxn, E_f, get_version_number())
    fig.gca().set_aspect('equal', adjustable='box')
    fig.tight_layout()
    fig.savefig(filepath, bbox_inches='tight')
    fig.savefig(filepath + '.png', bbox_inches='tight', dpi=300)
    # fig.show()
    plt.close(fig)
    print('Points plot:\n{}'.format(filepath))


def surface_plot_test():
    xn, yn = 100, 100
    x_length, y_length = 5, 5
    xs, ys = generate_axes(xn, yn, x_length, y_length)

    zs = np.zeros((xn, yn), dtype=float)
    for i in range(xn):
        for j in range(yn):
            zs[i, j] = i+j
            if i == 50 and j == 50:
                zs[i, j] = 200

    from plotting import plot_surface
    plot_surface(xs, ys, [zs])


def plot_greens():
    print('plot Greens')
    from greens_4_band import generate_4_band_bare_greens_eigenstate_weighting

    kxn, kyn = 300, 300
    kx_length, ky_length = 4 * np.pi / Params.a_0, 4 * np.pi / Params.a_0
    # kx_length, ky_length = 0.05, 0.1
    # kx_centre, ky_centre = -0.37, -0.33
    kx_centre, ky_centre = 0, 0
    kxs, kys = generate_axes(kxn, kyn, kx_length, ky_length)
    # kxs = np.linspace(kx_centre - kx_length / 2, kx_centre + kx_length / 2, kxn)
    # kys = np.linspace(ky_centre - ky_length / 2, ky_centre + ky_length / 2, kyn)

    G_0_k = generate_4_band_bare_greens_eigenstate_weighting(kxs, kys, phi_k=np.ones((kxn, kyn), dtype=float))

    G_0_k_trace = np.zeros((kxn, kyn), dtype=complex)
    for i in range(kxn):
        for j in range(kyn):
            G_0_k_trace[i, j] = G_0_k[i, j].trace()

    from plotting import plot_heatmap

    run_number = '1.3.0'

    filepath = '../plots/greens/greens_trace_{}_kxn={}_kx_length={:.2f}_kx_centre={}_ky_centre={}__eig-phi={}_eigenvalue-phi={}.png'\
        .format(run_number, kxn, kx_length, kx_centre, ky_centre, get_eigen_phi_str(), get_eigenvalue_phi_str())
    plot_heatmap(kxs, kys, np.abs(G_0_k_trace.transpose()), filename=filepath, x_label='$k_x (\AA^{-1})$', y_label='$k_y (\AA^{-1})$', show_bz=False)
    plot_heatmap(kxs, kys, np.abs(G_0_k_trace.transpose()), filename=filepath+'bz.png', x_label='$k_x (\AA^{-1})$',
                 y_label='$k_y (\AA^{-1})$', show_bz=True)


def plot_greens_den():
    print('plot Greens den')
    from greens_4_band import generate_4_band_bare_greens_den

    kxn, kyn = 300, 300
    kx_length, ky_length = 4 * np.pi / Params.a_0, 4 * np.pi / Params.a_0

    kxs, kys = generate_axes(kxn, kyn, kx_length, ky_length)

    kxn, kyn = 100, 100
    # kx_length, ky_length = 1.5 * np.pi / Params.a_0, 1.5 * np.pi / Params.a_0
    kx_length, ky_length = 0.05, 0.1
    kx_centre, ky_centre = -0.37, -0.33
    # kxs, kys = generate_axes(kxn, kyn, kx_length, ky_length)
    kxs = np.linspace(kx_centre - kx_length / 2, kx_centre + kx_length / 2, kxn)
    kys = np.linspace(ky_centre - ky_length / 2, ky_centre + ky_length / 2, kyn)

    G_0_k_den = generate_4_band_bare_greens_den(kxs, kys)

    from plotting import plot_heatmap

    filepath = '../plots/greens/greens_1-over-den_abs_kxn={}_{}.png'.format(kxn, get_eigenvalue_phi_str())
    plot_heatmap(kxs, kys, np.abs(G_0_k_den.transpose()), filename=filepath, x_label='$k_x (\AA^{-1})$',
                 y_label='$k_y (\AA^{-1})$', show_bz=False)
    plot_heatmap(kxs, kys, np.abs(G_0_k_den.transpose()), filename=filepath+'bz.png', x_label='$k_x (\AA^{-1})$',
                 y_label='$k_y (\AA^{-1})$', show_bz=True)


def plot_greens_num():
    print('plot Greens')
    from greens_4_band import generate_4_band_bare_greens_num

    kxn, kyn = 300, 300
    kx_length, ky_length = 4 * np.pi / Params.a_0, 4 * np.pi / Params.a_0

    kxs, kys = generate_axes(kxn, kyn, kx_length, ky_length)

    G_0_k = generate_4_band_bare_greens_num(kxs, kys)

    G_0_k_trace = np.zeros((kxn, kyn), dtype=complex)
    for i in range(kxn):
        for j in range(kyn):
            G_0_k_trace[i, j] = G_0_k[i, j].trace()

    from plotting import plot_heatmap

    print('min = {}'.format(np.min(np.abs(G_0_k_trace))))
    print('max = {}'.format(np.max(np.abs(G_0_k_trace))))

    filepath = '../plots/greens/greens_num_eig-phi={}_kxn={}.png'.format(get_eigen_phi_str(), kxn)
    plot_heatmap(kxs, kys, np.abs(G_0_k_trace), filename=filepath, x_label='$k_x (\AA^{-1})$', y_label='$k_y (\AA^{-1})$', show_bz=True)


def plot_greens_num_element():
    print('plot Greens')
    from greens_4_band import generate_4_band_bare_greens_num

    kxn, kyn = 300, 300
    kx_length, ky_length = 4 * np.pi / Params.a_0, 4 * np.pi / Params.a_0

    kxs, kys = generate_axes(kxn, kyn, kx_length, ky_length)

    kxn, kyn = 100, 100
    # kx_length, ky_length = 1.5 * np.pi / Params.a_0, 1.5 * np.pi / Params.a_0
    kx_length, ky_length = 0.05, 0.1
    kx_centre, ky_centre = -0.37, -0.33
    # kxs, kys = generate_axes(kxn, kyn, kx_length, ky_length)
    kxs = np.linspace(kx_centre - kx_length / 2, kx_centre + kx_length / 2, kxn)
    kys = np.linspace(ky_centre - ky_length / 2, ky_centre + ky_length / 2, kyn)

    G_0_k = generate_4_band_bare_greens_num(kxs, kys)

    G_0_k_00 = np.zeros((kxn, kyn), dtype=complex)
    G_0_k_11 = np.zeros((kxn, kyn), dtype=complex)
    G_0_k_22 = np.zeros((kxn, kyn), dtype=complex)
    G_0_k_33 = np.zeros((kxn, kyn), dtype=complex)
    for i in range(kxn):
        for j in range(kyn):
            G_0_k_00[i, j] = G_0_k[i, j][0, 0]
            G_0_k_11[i, j] = G_0_k[i, j][1, 1]
            G_0_k_22[i, j] = G_0_k[i, j][2, 2]
            G_0_k_33[i, j] = G_0_k[i, j][3, 3]

    from plotting import plot_heatmap

    print('min_00 = {}'.format(np.min(np.abs(G_0_k_00))))
    print('max_00 = {}'.format(np.max(np.abs(G_0_k_00))))

    print('min_11 = {}'.format(np.min(np.abs(G_0_k_11))))
    print('max_11 = {}'.format(np.max(np.abs(G_0_k_11))))

    print('min_22 = {}'.format(np.min(np.abs(G_0_k_22))))
    print('max_22 = {}'.format(np.max(np.abs(G_0_k_22))))

    print('min_33 = {}'.format(np.min(np.abs(G_0_k_33))))
    print('max_33 = {}'.format(np.max(np.abs(G_0_k_33))))

    filepath = '../plots/greens/greens_num_00_eig-phi={}_kxn={}.png'.format(get_eigen_phi_str(), kxn)
    plot_heatmap(kxs, kys, np.abs(G_0_k_00), filename=filepath, x_label='$k_x (\AA^{-1})$',
                 y_label='$k_y (\AA^{-1})$', show_bz=False)
    plot_heatmap(kxs, kys, np.abs(G_0_k_00), filename=filepath+'bz.png', x_label='$k_x (\AA^{-1})$',
                 y_label='$k_y (\AA^{-1})$', show_bz=True)

    filepath = '../plots/greens/greens_num_11_eig-phi={}_kxn={}.png'.format(get_eigen_phi_str(), kxn)
    plot_heatmap(kxs, kys, np.abs(G_0_k_11), filename=filepath, x_label='$k_x (\AA^{-1})$',
                 y_label='$k_y (\AA^{-1})$', show_bz=False)
    plot_heatmap(kxs, kys, np.abs(G_0_k_11), filename=filepath+'bz.png', x_label='$k_x (\AA^{-1})$',
                 y_label='$k_y (\AA^{-1})$', show_bz=True)

    filepath = '../plots/greens/greens_num_22_eig-phi={}_kxn={}.png'.format(get_eigen_phi_str(), kxn)
    plot_heatmap(kxs, kys, np.abs(G_0_k_22), filename=filepath, x_label='$k_x (\AA^{-1})$',
                 y_label='$k_y (\AA^{-1})$', show_bz=False)
    plot_heatmap(kxs, kys, np.abs(G_0_k_22), filename=filepath+'bz.png', x_label='$k_x (\AA^{-1})$',
                 y_label='$k_y (\AA^{-1})$', show_bz=True)

    filepath = '../plots/greens/greens_num_33_eig-phi={}_kxn={}.png'.format(get_eigen_phi_str(), kxn)
    plot_heatmap(kxs, kys, np.abs(G_0_k_33), filename=filepath, x_label='$k_x (\AA^{-1})$',
                 y_label='$k_y (\AA^{-1})$', show_bz=False)
    plot_heatmap(kxs, kys, np.abs(G_0_k_33), filename=filepath+'bz.png', x_label='$k_x (\AA^{-1})$',
                 y_label='$k_y (\AA^{-1})$', show_bz=True)


def investigate_hamiltonian_at_ks(kxs, kys):
    print(kxs, kys)
    from hamiltonian_4band import calculate_4_band_H_k
    from hamiltonian_4band import get_4_band_hamiltonian

    H_arr_R, rs, weights = get_4_band_hamiltonian(FilePaths.hamiltonian_NbSe2_4band_data)
    H_k_s = []
    for kx, ky in zip(kxs, kys):
        h = calculate_4_band_H_k(H_arr_R, rs, weights, kx, ky)
        H_k_s.append(h)
        print('({}, {})'.format(kx, ky))
        print(h)
        print('----')


def plot_basis_weightings():
    from hamiltonian_4band import calculate_4_band_H_k
    from hamiltonian_4band import get_4_band_hamiltonian
    from hamiltonian_4band import extract_eigenstates_from_hamiltonian
    from plotting import plot_heatmap

    kxn, kyn = 100, 100
    kx_length, ky_length = 4 * np.pi / Params.a_0, 4 * np.pi / Params.a_0
    kxs, kys = generate_axes(kxn, kyn, kx_length, ky_length)

    H_arr_R, rs, weights = get_4_band_hamiltonian(FilePaths.hamiltonian_NbSe2_4band_data)

    weighting_0 = np.zeros((kxn, kyn), dtype=float)
    weighting_1 = np.zeros((kxn, kyn), dtype=float)
    weighting_2 = np.zeros((kxn, kyn), dtype=float)
    weighting_3 = np.zeros((kxn, kyn), dtype=float)

    weighting_inplane = np.zeros((kxn, kyn), dtype=float)
    weighting_outplane = np.zeros((kxn, kyn), dtype=float)
    weighting_up = np.zeros((kxn, kyn), dtype=float)
    weighting_down = np.zeros((kxn, kyn), dtype=float)
    eigval_i = 0
    for i in range(kxn):
        for j in range(kyn):
            h = calculate_4_band_H_k(H_arr_R, rs, weights, kxs[i], kys[j])

            eigenvectors, eigenvalues = extract_eigenstates_from_hamiltonian(h)
            # print(eigenvectors)
            # for k in range(len(eigenvalues)):
            #     total = 0
                # for l in range(len(eigenvectors[0])):
                #     total += eigenvectors[j][i] * np.conjugate(eigenvectors[j][i])
                    # print(k, l, np.dot(np.conjugate(eigenvectors[k].transpose()), eigenvectors[l]))
                    # print(k, l, eigenvectors[k].getH() * eigenvectors[l])
            # exit()
            # for k in range(len(eigenvalues)):
            #     weighting_inplane[i, j] += (np.conjugate(eigenvectors[k][0]) * eigenvectors[k][0])[0, 0].real
            #     weighting_inplane[i, j] += (np.conjugate(eigenvectors[k][2]) * eigenvectors[k][2])[0, 0].real
            #     weighting_outplane[i, j] += (np.conjugate(eigenvectors[k][1]) * eigenvectors[k][1])[0, 0].real
            #     weighting_outplane[i, j] += (np.conjugate(eigenvectors[k][3]) * eigenvectors[k][3])[0, 0].real
            #
            #     weighting_up[i, j] += (np.conjugate(eigenvectors[k][0]) * eigenvectors[k][0])[0, 0].real
            #     weighting_up[i, j] += (np.conjugate(eigenvectors[k][1]) * eigenvectors[k][1])[0, 0].real
            #
            #     weighting_down[i, j] += (np.conjugate(eigenvectors[k][2]) * eigenvectors[k][2])[0, 0].real
            #     weighting_down[i, j] += (np.conjugate(eigenvectors[k][3]) * eigenvectors[k][3])[0, 0].real
            #
            #     weighting_0[i, j] += (np.conjugate(eigenvectors[k][0]) * eigenvectors[k][0])[0, 0].real
            #     weighting_1[i, j] += (np.conjugate(eigenvectors[k][1]) * eigenvectors[k][1])[0, 0].real
            #     weighting_2[i, j] += (np.conjugate(eigenvectors[k][2]) * eigenvectors[k][2])[0, 0].real
            #     weighting_3[i, j] += (np.conjugate(eigenvectors[k][3]) * eigenvectors[k][3])[0, 0].real
            weighting_0[i, j] += (np.conjugate(eigenvectors[eigval_i][0]) * eigenvectors[eigval_i][0])[0, 0].real
            weighting_1[i, j] += (np.conjugate(eigenvectors[eigval_i][1]) * eigenvectors[eigval_i][1])[0, 0].real
            weighting_2[i, j] += (np.conjugate(eigenvectors[eigval_i][2]) * eigenvectors[eigval_i][2])[0, 0].real
            weighting_3[i, j] += (np.conjugate(eigenvectors[eigval_i][3]) * eigenvectors[eigval_i][3])[0, 0].real

            weighting_inplane[i, j] += (np.conjugate(eigenvectors[eigval_i][0]) * eigenvectors[eigval_i][0])[0, 0].real
            weighting_inplane[i, j] += (np.conjugate(eigenvectors[eigval_i][2]) * eigenvectors[eigval_i][2])[0, 0].real
            weighting_outplane[i, j] += (np.conjugate(eigenvectors[eigval_i][1]) * eigenvectors[eigval_i][1])[0, 0].real
            weighting_outplane[i, j] += (np.conjugate(eigenvectors[eigval_i][3]) * eigenvectors[eigval_i][3])[0, 0].real

            weighting_up[i, j] += (np.conjugate(eigenvectors[eigval_i][0]) * eigenvectors[eigval_i][0])[0, 0].real
            weighting_up[i, j] += (np.conjugate(eigenvectors[eigval_i][1]) * eigenvectors[eigval_i][1])[0, 0].real
            weighting_down[i, j] += (np.conjugate(eigenvectors[eigval_i][2]) * eigenvectors[eigval_i][2])[0, 0].real
            weighting_down[i, j] += (np.conjugate(eigenvectors[eigval_i][3]) * eigenvectors[eigval_i][3])[0, 0].real

    print(np.min(weighting_0), np.max(weighting_0))
    print(np.min(weighting_1), np.max(weighting_1))
    print(np.min(weighting_2), np.max(weighting_2))
    print(np.min(weighting_3), np.max(weighting_3))
    print(np.min(weighting_inplane), np.max(weighting_inplane))
    print(np.min(weighting_outplane), np.max(weighting_outplane))
    print(np.min(weighting_up), np.max(weighting_up))
    print(np.min(weighting_down), np.max(weighting_down))

    filepath = '../plots/ham/weighting/state={}_eigenvalue={}_.png'
    plot_heatmap(kxs, kys, weighting_0.transpose(), filename=filepath.format(0, eigval_i), x_label='$k_x (\AA^{-1})$', y_label='$k_y (\AA^{-1})$', show_bz=True)
    plot_heatmap(kxs, kys, weighting_1.transpose(), filename=filepath.format(1, eigval_i), x_label='$k_x (\AA^{-1})$',
                 y_label='$k_y (\AA^{-1})$', show_bz=True)
    plot_heatmap(kxs, kys, weighting_2.transpose(), filename=filepath.format(2, eigval_i), x_label='$k_x (\AA^{-1})$',
                 y_label='$k_y (\AA^{-1})$', show_bz=True)
    plot_heatmap(kxs, kys, weighting_3.transpose(), filename=filepath.format(3, eigval_i), x_label='$k_x (\AA^{-1})$',
                 y_label='$k_y (\AA^{-1})$', show_bz=True)
    filepath_in = '../plots/ham/weighting/inplane_eigval={}_.png'.format(eigval_i)
    plot_heatmap(kxs, kys, weighting_inplane.transpose(), filename=filepath_in, x_label='$k_x (\AA^{-1})$',
                 y_label='$k_y (\AA^{-1})$', show_bz=True)
    filepath_out = '../plots/ham/weighting/outplane_eigval={}_.png'.format(eigval_i)
    plot_heatmap(kxs, kys, weighting_outplane.transpose(), filename=filepath_out, x_label='$k_x (\AA^{-1})$',
                 y_label='$k_y (\AA^{-1})$', show_bz=True)
    filepath_up = '../plots/ham/weighting/up_eigval={}.png'.format(eigval_i)
    plot_heatmap(kxs, kys, weighting_up.transpose(), filename=filepath_up,
                 x_label='$k_x (\AA^{-1})$',
                 y_label='$k_y (\AA^{-1})$', show_bz=True)
    filepath_down = '../plots/ham/weighting/down_eigval={}.png'.format(eigval_i)
    plot_heatmap(kxs, kys, weighting_down.transpose(), filename=filepath_down,
                 x_label='$k_x (\AA^{-1})$',
                 y_label='$k_y (\AA^{-1})$', show_bz=True)

    from plotting import plot_combined_images
    plot_combined_images([filepath_in, filepath_out], '../plots/ham/weighting/in-out-plane-combined_eigval={}.png'.format(eigval_i))
    plot_combined_images([filepath_up, filepath_down], '../plots/ham/weighting/up-out-down-combined_eigval={}.png'.format(eigval_i))


def plot_spin_surface():
    from hamiltonian_4band import calculate_4_band_H_k
    from hamiltonian_4band import get_4_band_hamiltonian
    from hamiltonian_4band import extract_eigenstates_from_hamiltonian
    from plotting import plot_heatmap

    kxn, kyn = 100, 100
    kx_length, ky_length = 4 * np.pi / Params.a_0, 4 * np.pi / Params.a_0
    kxs, kys = generate_axes(kxn, kyn, kx_length, ky_length)

    H_arr_R, rs, weights = get_4_band_hamiltonian(FilePaths.hamiltonian_NbSe2_4band_data)

    weighting_0 = np.zeros((kxn, kyn), dtype=float)
    weighting_1 = np.zeros((kxn, kyn), dtype=float)
    weighting_2 = np.zeros((kxn, kyn), dtype=float)
    weighting_3 = np.zeros((kxn, kyn), dtype=float)

    p_z = np.zeros((4, 4), dtype=complex)
    p_z[0, 0] = 1
    p_z[1, 1] = 1
    p_z[2, 2] = -1
    p_z[3, 3] = -1

    for i in range(kxn):
        for j in range(kyn):
            h = calculate_4_band_H_k(H_arr_R, rs, weights, kxs[i], kys[j])

            eigenvectors, eigenvalues = extract_eigenstates_from_hamiltonian(h)

            weighting_0[i, j] += (eigenvectors[0].getH() * p_z * eigenvectors[0])[0, 0].real
            weighting_1[i, j] += (eigenvectors[1].getH() * p_z * eigenvectors[1])[0, 0].real
            weighting_2[i, j] += (eigenvectors[2].getH() * p_z * eigenvectors[2])[0, 0].real
            weighting_3[i, j] += (eigenvectors[3].getH() * p_z * eigenvectors[3])[0, 0].real

    print(np.min(weighting_0), np.max(weighting_0))
    print(np.min(weighting_1), np.max(weighting_1))
    print(np.min(weighting_2), np.max(weighting_2))
    print(np.min(weighting_3), np.max(weighting_3))

    filepath_0 = '../plots/ham/weighting/spin-surface-eigenvalue={}_.png'.format(0)
    plot_heatmap(kxs, kys, weighting_0.transpose(), filename=filepath_0, x_label='$k_x (\AA^{-1})$', y_label='$k_y (\AA^{-1})$', show_bz=True)
    filepath_1 = '../plots/ham/weighting/spin-surface-eigenvalue={}_.png'.format(1)
    plot_heatmap(kxs, kys, weighting_1.transpose(), filename=filepath_1, x_label='$k_x (\AA^{-1})$',
                 y_label='$k_y (\AA^{-1})$', show_bz=True)
    filepath_2 = '../plots/ham/weighting/spin-surface-eigenvalue={}_.png'.format(2)
    plot_heatmap(kxs, kys, weighting_2.transpose(), filename=filepath_2, x_label='$k_x (\AA^{-1})$',
                 y_label='$k_y (\AA^{-1})$', show_bz=True)
    filepath_3 = '../plots/ham/weighting/spin-surface-eigenvalue={}_.png'.format(3)
    plot_heatmap(kxs, kys, weighting_3.transpose(), filename=filepath_3, x_label='$k_x (\AA^{-1})$',
                 y_label='$k_y (\AA^{-1})$', show_bz=True)

    from plotting import plot_hstack_images, plot_vstack_images
    filepath_01_hstack = '../plots/temp/spin-surface-hstack_01.png'
    filepath_23_hstack = '../plots/temp/spin-surface-hstack_23.png'
    plot_hstack_images([filepath_0, filepath_1], filepath_01_hstack)
    plot_hstack_images([filepath_2, filepath_3], filepath_23_hstack)
    plot_vstack_images([filepath_01_hstack, filepath_23_hstack], '../plots/ham/weighting/spin-surface-stack_0134.png')


def plot_fermi_surface_spin_texture():
    from matplotlib.collections import LineCollection
    H_arr_R, rs, weights = get_4_band_hamiltonian(FilePaths.hamiltonian_NbSe2_4band_data)

    # Fermi-energy
    E_f = Params.E_f  # eV
    dE_f = Params.delta_E_f

    E_f = E_f + dE_f

    kxn = 200
    kyn = 200

    a_0 = 3.44
    kx_length = 4 * np.pi / a_0
    ky_length = 4 * np.pi / a_0
    # kx_length = 4/3
    # ky_length = 4/3

    kxs, kys = generate_axes(kxn, kyn, kx_length, ky_length)

    eigenvalues_k = np.zeros((kxn, kyn), dtype=type([1.]))

    for i in range(kxn):
        for j in range(kyn):
            kx = kxs[i]
            ky = kys[j]

            h_k = calculate_4_band_H_k(H_arr_R, rs, weights, kx, ky)
            _, evs = extract_eigenstates_from_hamiltonian(h_k)
            eigenvalues_k[i, j] = evs - E_f

        if i % 20 == 0:
            print('finished row: {}'.format(i))

    fig, ax = plt.subplots()
    from plotting import plot_surface

    p_z = np.zeros((4, 4), dtype=complex)
    p_z[0, 0] = 1
    p_z[1, 1] = 1
    p_z[2, 2] = -1
    p_z[3, 3] = -1

    # zss = []
    #
    for i in range(4):
        if i == 2:
            break
        zs = np.zeros((kxn, kyn), dtype=float)

        for j in range(kxn):
            for k in range(kyn):
                zs[j, k] = eigenvalues_k[j, k][i]

        plt.figure()
        CS = plt.contour(kxs, kys, zs.transpose(), levels=[0], linewidths=[0.5], linestyles=['solid'], colors=['black'])
        # plt.show()
        for m in range(len(CS.allsegs[0])):
            dat0 = CS.allsegs[0][m]
            ws = []
            for kx, ky in zip(dat0[:, 0], dat0[:, 1]):
                h_k = calculate_4_band_H_k(H_arr_R, rs, weights, kx, ky)
                evects, evs = extract_eigenstates_from_hamiltonian(h_k)
                w = (evects[i].getH() * p_z * evects[i])[0, 0].real
                ws.append(w)

            points = np.array([dat0[:, 0], dat0[:, 1]]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            lc = LineCollection(segments, cmap='winter', norm=plt.Normalize(vmin=-1, vmax=1))
            # Set the values used for colormapping
            # print(ws)
            lc.set_array(ws)
            lc.set_linewidth(0.5)
            line = ax.add_collection(lc)
            # ax.plot(dat0[:, 0], dat0[:, 1])

            if i == 0 and m == 0:
                fig.colorbar(line, ax=ax)


        print(len(CS.allsegs[0][0]))
        # exit()

        # ax.plot(dat0[:, 0], dat0[:, 1])

        # (eigenvectors[0].getH() * p_z * eigenvectors[0])[0, 0].real


        # zss.append(zs)
    # plot_surface(kxs, kys, zss[:-2])
    # plot_surface(kxs, kys, zss[2:])
    #     , colors=['black']

    # Plot BZ (new)
    G = Params.G
    K = Params.K / 3
    M = Params.M / 3
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
    filepath = FilePaths.contour_plots + 'kxn={}_contour_spin-texture_Ef={}_lthick=0.5_{}.pdf'.format(kxn, E_f, get_version_number())
    fig.gca().set_aspect('equal', adjustable='box')
    fig.tight_layout()
    fig.savefig(filepath, bbox_inches='tight')
    fig.savefig(filepath + '.png', bbox_inches='tight', dpi=300)
    # fig.show()
    plt.close(fig)
    print('Points plot:\n{}'.format(filepath))


def plot_H_k_elements():
    from hamiltonian_4band import calculate_4_band_H_k
    from hamiltonian_4band import get_4_band_hamiltonian
    from hamiltonian_4band import extract_eigenstates_from_hamiltonian
    from plotting import plot_heatmap
    from plotting import plot_combined_H_k_els

    kxn, kyn = 100, 100
    kx_length, ky_length = 8 * np.pi / Params.a_0, 8 * np.pi / Params.a_0
    # kx_length, ky_length = 0.05, 0.1
    # kx_centre, ky_centre = -0.37, -0.33
    kxs, kys = generate_axes(kxn, kyn, kx_length, ky_length)
    # kxs = np.linspace(kx_centre - kx_length / 2, kx_centre + kx_length / 2, kxn)
    # kys = np.linspace(ky_centre - ky_length / 2, ky_centre + ky_length / 2, kyn)
    kx_centre, ky_centre = 0, 0

    H_arr_R, rs, weights = get_4_band_hamiltonian(FilePaths.hamiltonian_NbSe2_4band_data)

    for m in range(4):
        for n in range(4):
            h_k_mn = np.zeros((kxn, kyn), dtype=complex)
            for i in range(kxn):
                for j in range(kyn):
                    h = calculate_4_band_H_k(H_arr_R, rs, weights, kxs[i], kys[j])
                    h_k_mn[i, j] = h[m, n]

                    eigenvectors, eigenvalues = extract_eigenstates_from_hamiltonian(h)

                    for k in range(len(eigenvalues)):
                        # total = 0
                        for l in range(len(eigenvectors[0])):
                            # total += eigenvectors[j][i] * np.conjugate(eigenvectors[j][i])
                            print(k, l, np.dot(np.conjugate(eigenvectors[k].transpose()), eigenvectors[l]))
                            print(k, l, eigenvectors[k].getH() * eigenvectors[l])
                        # print(total, np.abs(total))


            # filename = '../plots/ham/h_k_{}{}_k-centre={},{}_k-length={},{}_kxn={}'.format(m, n, kx_centre, ky_centre, kx_length, ky_length, kxn)
            filename = '../plots/ham/H_k/h_k_{}{}'.format(m, n)
            plot_heatmap(kxs, kys, h_k_mn.transpose().imag, filename=filename+'_imag.png', x_label='$k_x (\AA^{-1})$', y_label='$k_y (\AA^{-1})$', show_bz=True)
            plot_heatmap(kxs, kys, h_k_mn.transpose().real, filename=filename + '_real.png', x_label='$k_x (\AA^{-1})$',
                         y_label='$k_y (\AA^{-1})$', show_bz=True)
            plot_heatmap(kxs, kys, np.abs(h_k_mn.transpose()), filename=filename + '_abs.png', x_label='$k_x (\AA^{-1})$',
                         y_label='$k_y (\AA^{-1})$', show_bz=True)

            plot_combined_H_k_els(m, n)


def plot_H_r_elements():
    from hamiltonian_4band import get_4_band_hamiltonian
    from plotting import plot_scatter_heatmap
    from plotting import plot_combined_H_r_els

    H_arr_R, rs, weights = get_4_band_hamiltonian(FilePaths.hamiltonian_NbSe2_4band_data)
    xs = []
    ys = []
    a1 = Params.a_1
    a2 = Params.a_2
    for r in rs:
        xs.append((r[0] * a1 + r[1] * a2)[0])
        ys.append((r[0] * a1 + r[1] * a2)[1])

    for m in range(H_arr_R[0].shape[0]):
        for n in range(H_arr_R[0].shape[1]):
            h_mn_s_real = []
            h_mn_s_imag = []
            h_mn_s_abs = []
            for h in H_arr_R:
                h_mn_s_real.append(h[m, n].real)
                h_mn_s_imag.append(h[m, n].imag)
                h_mn_s_abs.append(np.abs(h[m, n]))

            filename = '../plots/ham/H_r/h_r_{}{}'.format(m, n)
            plot_scatter_heatmap(xs, ys, h_mn_s_imag, filename=filename + '_imag.png', x_label='x $(\AA)$', y_label='x $(\AA)$', show_nb=True)
            plot_scatter_heatmap(xs, ys, h_mn_s_real, filename=filename + '_real.png', x_label='x $(\AA)$',
                                 y_label='x $(\AA)$', show_nb=True)
            plot_scatter_heatmap(xs, ys, h_mn_s_abs, filename=filename + '_abs.png', x_label='x $(\AA)$',
                                 y_label='x $(\AA)$', show_nb=True)
            plot_combined_H_r_els(m, n)


def init():
    Params.eigen_phi = [0, 0, 1, 0]
    Params.eigenvalue_phi = [1, 1, 1, 1]
    RunParams.USE_CACHE = True


def main():
    init()
    # plot_greens()
    # investigate_hamiltonian_at_ks([-0.39, -0.38, -0.37, -0.36, -0.35], [-0.34, -0.34, -0.34,  -0.34, -0.34])
    # plot_H_k_elements()
    # plot_H_r_elements()
    # plot_basis_weightings()
    # plot_greens_num_element()
    # plot_greens_den()
    # plot_bands()
    # plot_bands_with_weighting()
    # plot_bands_with_spin_texture()
    # plot_spin_surface()
    plot_fermi_surface_spin_texture()
    # surface_plot_test()
    # plot_contours_on_one()
    # plot_bands()


def get_version_number():
    return '_1.80.3_eig_dEf={}'.format(Params.delta_E_f)


if __name__ == '__main__':
    main()

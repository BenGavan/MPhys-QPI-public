import time
# from numba import jit
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy import linalg
from params import *


def extract_values_from_line(line):
    els = line.split(' ')

    vs = np.array([])

    for e in els:
        if e == '' or e == '\n':  # check for '\n' might not be needed (but added just in case)
            continue

        vs = np.append(vs, float(e))

    return vs


def extract_eigenstates_from_hamiltonian(hamiltonian):
    '''
    Extracts the eigenvectors and eigenvalues from a given hamiltonian (H not diagonalized yet)
    Returns vectors in column form
    '''
    eigenstates = linalg.eig(hamiltonian)

    eigenvalues = eigenstates[0]
    eigenvects = eigenstates[1]

    eigenvectors = [None] * len(eigenvects)

    for i in range(len(eigenvects)):
        numpy_mat = np.matrix(eigenvects[:, i])
        if (numpy_mat.shape[0] == 1) and (numpy_mat.shape[1] == 2):
            numpy_mat = numpy_mat.transpose()

        eigenvectors[i] = numpy_mat

    # sort
    # sort eigenvectors and eigenvalues based on eigenvalues
    n = len(eigenvalues)
    for i in range(n):
        already_sorted = True

        for j in range(n - i - 1):
            if eigenvalues[j] > eigenvalues[j + 1]:  # swap
                eigenvalues[j], eigenvalues[j + 1] = eigenvalues[j + 1], eigenvalues[j]
                eigenvectors[j], eigenvectors[j + 1] = eigenvectors[j + 1], eigenvectors[j]
                already_sorted = False
        if already_sorted:
            break

    return eigenvectors, eigenvalues


def generate_axes(kx_n, ky_n, kx_length, ky_length, kx_centre=0., ky_centre=0.):
    '''

    :param kx_n: # pixes along x axis
    :param ky_n: # pixes along y axis
    :param kx_length:
    :param ky_length:
    :param ky_centre:
    :param kx_centre:
    :return:
    '''
    xs = np.linspace((-kx_length / 2) + kx_centre, (kx_length / 2) + kx_centre, kx_n)
    ys = np.linspace((-ky_length / 2) + ky_centre, (ky_length / 2) + ky_centre, ky_n)

    return xs, ys


def calculate_H_k(ham_arr_R, rs, weights, a_1, a_2, k):
    h_k = np.zeros((2, 2), dtype=complex)

    for r_i in range(len(rs)):
        r = rs[r_i]
        R = r[0] * a_1 + r[1] * a_2

        phase = -1j * np.dot(k, R)

        h_k += ham_arr_R[r_i] * np.exp(phase) / weights[r_i]

    return h_k


def calculate_eigenvalues_for_ks(ham_arr_R, rs, weights, a_1, a_2, ks):
    xs = []
    H_ks = []

    dk_vec = ks[-1] - ks[0]
    dk = np.sqrt(dk_vec[0] ** 2 + dk_vec[1] ** 2) / len(ks)

    for i in range(len(ks)):
        xs.append(dk * i)
        H_ks.append(calculate_H_k(ham_arr_R, rs, weights, a_1, a_2, k=ks[i]))

    eigenvals_lower = np.array([])
    eigenvals_higher = np.array([])

    for i in range(len(H_ks)):
        _, eigenvalues = extract_eigenstates_from_hamiltonian(H_ks[i])
        eigenvals_higher = np.append(eigenvals_higher, max(eigenvalues))
        eigenvals_lower = np.append(eigenvals_lower, min(eigenvalues))

    return xs, eigenvals_lower, eigenvals_higher


def get_hamiltonian(filepath):
    f = open(filepath)

    number_bands = -1
    number_hopping_vectors = -1  # = number of weights

    weights = np.array([])
    H_arr_R = np.array([], dtype=np.matrix)
    hopping_vectors = np.array([])

    line_num = -1
    H_line = -1
    for line in f:
        line_num += 1

        if line_num == 0:
            continue

        vs = extract_values_from_line(line)

        if line_num == 1 and len(vs) == 1:
            number_bands = vs[0]
            continue

        if line_num == 2 and len(vs) == 1:
            number_hopping_vectors = int(vs[0])

            H_arr_R = np.array([np.zeros((2, 2), dtype=complex) for _ in range(number_hopping_vectors)])
            hopping_vectors = np.array([np.zeros(2, dtype=float) for _ in range(number_hopping_vectors)])
            continue

        if number_hopping_vectors == -1:
            continue  # still need to read number_hopping_vectors on line 2

        if len(weights) != number_hopping_vectors:
            weights = np.append(weights, vs)
            continue

        H_line += 1
        H_arr_index = int(H_line/4)

        if H_line % 4 == 0:  # only need to assign hopping vector at the start of each possition (repeasted 4 times correspinding to the 4 elements of the hamilitonian at a given point)
            hopping_vectors[H_arr_index] = vs[:2]
        H_arr_R[H_arr_index][int(vs[3])-1, int(vs[4])-1] = vs[5] + 1j * vs[6]

    f.close()

    rs = hopping_vectors

    return H_arr_R, rs, weights


def plot_bands():
    H_arr_R, rs, weights = get_hamiltonian('../data/Hamiltonian.dat')

    # real lattice vectors
    a_0 = 3.44  # \AA (Angstrom)
    a_1 = a_0 * np.array([np.sqrt(3) / 2, -1 / 2])
    a_2 = a_0 * np.array([0, 1])

    # reciprocal lattice vectors
    b_1 = (2 * np.pi / a_0) * np.array([2 * np.sqrt(3) / 3, 0])
    b_2 = (2 * np.pi / a_0) * np.array([np.sqrt(3) / 3, 1])

    G = np.array([0, 0])
    M = 0.5 * b_1
    K = np.array([0.5 * b_1[0], 0.5 * b_1[0] * np.tan(30 * np.pi / 180)])

    n = 1000

    # Gamma -> M
    ks_Gam_M = np.array([G + i * M for i in np.linspace(0, 1, n)])
    xs_G_M, evals_lower_G_M, evals_upper_G_M = calculate_eigenvalues_for_ks(H_arr_R, rs, weights, a_1, a_2, ks_Gam_M)

    # M -> K
    ks_M_K = np.array([M + (K - M) * i for i in np.linspace(0, 1, n)])
    xs_M_K, evals_lower_M_K, evals_upper_M_K = calculate_eigenvalues_for_ks(H_arr_R, rs, weights, a_1, a_2, ks_M_K)
    xs_M_K = xs_M_K + xs_G_M[-1]

    # K -> Gamma
    ks_K_G = np.array([K - K * i for i in np.linspace(0, 1, n)])
    xs_K_G, evals_lower_K_G, evals_upper_K_G = calculate_eigenvalues_for_ks(H_arr_R, rs, weights, a_1, a_2, ks_K_G)
    xs_K_G = xs_K_G + xs_M_K[-1]

    xs = np.array([])
    eigenvalues_lower = np.array([])
    eigenvalues_upper = np.array([])

    xs = np.append(xs, xs_G_M)
    eigenvalues_lower = np.append(eigenvalues_lower, evals_lower_G_M)
    eigenvalues_upper = np.append(eigenvalues_upper, evals_upper_G_M)

    xs = np.append(xs, xs_M_K)
    eigenvalues_lower = np.append(eigenvalues_lower, evals_lower_M_K)
    eigenvalues_upper = np.append(eigenvalues_upper, evals_upper_M_K)

    xs = np.append(xs, xs_K_G)
    eigenvalues_lower = np.append(eigenvalues_lower, evals_lower_K_G)
    eigenvalues_upper = np.append(eigenvalues_upper, evals_upper_K_G)

    # Fermi-energy
    E_f = -3.9285  # eV
    dE_f = -50 * pow(10, -3)  # 110 meV
    E_f = E_f + dE_f

    eigenvalues_lower = eigenvalues_lower + 3.9785
    eigenvalues_upper = eigenvalues_upper + 3.9785

    min_e = min(min(eigenvalues_lower), min(eigenvalues_upper))
    max_e = max(max(eigenvalues_lower), max(eigenvalues_upper))

    e_range_min = min_e - 0.05
    e_range_max = max_e + 0.05

    plt.figure(figsize=(9, 6))

    # ff7f0e = upper, 1f77b4=lower
    plt.plot(xs, eigenvalues_lower, label='lower', color='#1f77b4')
    plt.plot(xs, eigenvalues_upper, label='upper', color='#ff7f0e')

    # plt.plot([min(xs), max(xs)], [E_f + 3.9785, E_f + 3.9785], label='$E_f =${} eV'.format(dE_f), color='gray')
    x = [0, xs_G_M[-1], xs_M_K[-1], xs_K_G[-1]]
    print('x = ', x)
    x_tick_labels = ['$\Gamma$', 'M', 'K', '$\Gamma$']
    plt.xticks(x, x_tick_labels, rotation='horizontal')
    plt.ylabel('$E-E_f$ (eV)')
    plt.xlim(x[0], x[-1])
    plt.ylim(e_range_min, e_range_max)

    # add horizontal bars
    plt.plot([x[1], x[1]], [e_range_min, e_range_max], color='#000000', linewidth=0.5)
    plt.plot([x[2], x[2]], [e_range_min, e_range_max], color='#000000', linewidth=0.5)

    # plt.legend(loc='lower right')
    filepath = '../plots/NbSe2_band_with_fermi_3_n={}_dEf={}_E_f={}_new.pdf'.format(n, dE_f, E_f)
    print('filepath: \n', filepath)
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()
    # plt.show()


def plot_bz(ax):
    K = Params.K
    M = Params.M

    cs = [K, (K[0], -K[1]), -2 * K + 2 * M, (-K[0], -K[1]), (-K[0], K[1]), 2 * K - 2 * M, K]
    xs_brillouin = np.array([])
    ys_brillouin = np.array([])

    for coord in cs:
        xs_brillouin = np.append(xs_brillouin, coord[0])
        ys_brillouin = np.append(ys_brillouin, coord[1])

    ax.plot(xs_brillouin, ys_brillouin, color='gray', linestyle='dashed', linewidth=0.5)  # , zorder=1

    xs_brillouin = np.array([])
    ys_brillouin = np.array([])

    for coord in cs:
        xs_brillouin = np.append(xs_brillouin, coord[0] * 2 / 3)
        ys_brillouin = np.append(ys_brillouin, coord[1] * 2 / 3)

    ax.plot(xs_brillouin, ys_brillouin, color='gray', linestyle='dashed', linewidth=0.5)  # , zorder=1

    xs_brillouin = np.array([])
    ys_brillouin = np.array([])

    for coord in cs:
        xs_brillouin = np.append(xs_brillouin, coord[0] * 1 / 3)
        ys_brillouin = np.append(ys_brillouin, coord[1] * 1 / 3)

    ax.plot(xs_brillouin, ys_brillouin, color='gray', linestyle='dashed', linewidth=0.5)  # , zorder=1


def plot_fermi_surface():
    H_arr_R, rs, weights = get_hamiltonian('../data/Hamiltonian.dat')

    # real lattice vectors
    a_0 = 3.44  # \AA (Angstrom)
    a_1 = a_0 * np.array([np.sqrt(3) / 2, -1 / 2])
    a_2 = a_0 * np.array([0, 1])

    # reciprocal lattice vectors
    b_1 = (2 * np.pi / a_0) * np.array([2 * np.sqrt(3) / 3, 0])
    b_2 = (2 * np.pi / a_0) * np.array([np.sqrt(3) / 3, 1])

    # Fermi-energy
    E_f = -3.9285  # eV
    dE_f = -50 * pow(10, -3)

    E_f = E_f + dE_f

    # make k-space (to be used in constructing H(k))
    kx_n = 100
    ky_n = 100
    kxs, kys = generate_axes(kx_n, ky_n, 4.2, 4.2)

    # make mesh to hold hamiltonians(k) (mesh of 2*2 matrices)
    Hs_k = np.matrix(np.zeros((kx_n, ky_n), dtype=np.matrix))

    es_lower = np.matrix(np.zeros((kx_n, ky_n), dtype=float))
    es_upper = np.matrix(np.zeros((kx_n, ky_n), dtype=float))

    K = Params.K
    M = Params.M

    cs = [K, (K[0], -K[1]), -2 * K + 2 * M, (-K[0], -K[1]), (-K[0], K[1]), 2 * K - 2 * M, K]

    for i in range(kx_n):
        for j in range(ky_n):

            # is_k_pocket = False
            # for p in cs:
            #     kx = kxs[i] - p[0]
            #     ky = kys[j] - p[1]
            #
            #     k = np.sqrt(kx**2 + ky**2)
            #     phi_k[i, j] += unnormalized_gaussian(kx, ky, Params.alpha_gauss)
                # if k < Params.k_pocket_radius:
                #     is_k_pocket = True

            # if not is_k_pocket:
            #     es_lower[i, j] = -1000
            #     es_upper[i, j] = -1000
            #     continue

            # if np.sqrt(kxs[i] ** 2 + kys[j] ** 2) > 0.65:
            #     es_lower[i, j] = -1000
            #     es_upper[i, j] = -1000
            #     continue
            Hs_k[i, j] = calculate_H_k(H_arr_R, rs, weights, a_1=a_1, a_2=a_2, k=np.array([kxs[i], kys[j]]))

            _, e_values = extract_eigenstates_from_hamiltonian(Hs_k[i, j])

            e_values = e_values.real

            es_lower[i, j] = min(e_values)
            es_upper[i, j] = max(e_values)

            # es_lower[i, j] = -1000

        if i % 10 == 0:
            print('finished ', i)

    # plot fermi-surface
    fig, ax = plt.subplots()
    c_upper = ax.contour(kxs, kys, es_upper.transpose(), levels=[E_f], colors=['#ff7f0e'], linestyles='solid')
    c_lower = ax.contour(kxs, kys, es_lower.transpose(), levels=[E_f], colors=['#1f77b4'], linestyles='solid')

    plot_heatmap(kxs, kys, es_upper.transpose(), 'surface.png', "x", 'y')
    ax.set_xlabel('$k_x$ $(\AA^{-1})$')
    ax.set_ylabel('$k_y$ $(\AA^{-1})$')

    plot_bz(ax)

    c_upper.collections[0].set_label('$E_f = ${} eV (upper)'.format(E_f))

    c_lower.collections[0].set_label('$E_f = ${} eV (lower)'.format(E_f))

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=2)
    # ax.legend(loc='upper center')
    fig.tight_layout()
    fig.gca().set_aspect('equal', adjustable='box')
    filepath = '../plots/contour_with-bz_kxn={}_dEf={}_all.pdf'.format(kx_n, dE_f)
    print(filepath)
    fig.savefig(filepath)
    fig.show()
    plt.close(fig)


def plot_heatmap(xs_top, ys_top, zs, filename, x_label, y_label, show_bz=False, show_nb=False, show_se=False):
    fig, ax = plt.subplots()

    c = ax.pcolormesh(xs_top, ys_top, zs, cmap='hot')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if show_bz:
        K = Params.K
        M = Params.M

        cs = [Params.K, (K[0], -K[1]), -2 * K + 2 * M, (-K[0], -K[1]), (-K[0], K[1]), 2 * K - 2 * M, Params.K]
        xs_brillouin = np.array([])
        ys_brillouin = np.array([])

        for coord in cs:
            xs_brillouin = np.append(xs_brillouin, coord[0])
            ys_brillouin = np.append(ys_brillouin, coord[1])

        ax.plot(xs_brillouin, ys_brillouin, color='gray', linestyle='dashed')  # , zorder=1

        # xs_brillouin = np.array([])
        # ys_brillouin = np.array([])
        #
        # for coord in cs:
        #     xs_brillouin = np.append(xs_brillouin, coord[0]/3)
        #     ys_brillouin = np.append(ys_brillouin, coord[1]/3)
        #
        # ax.plot(xs_brillouin, ys_brillouin, color='gray', linestyle='dashed')  # , zorder=1

    if show_nb:
        xs_top = []
        ys_top = []

        # First hex
        xs_top.append(-Params.a_1[0])
        ys_top.append(-Params.a_1[1])

        xs_top.append(Params.a_1[0])
        ys_top.append(Params.a_1[1])

        xs_top.append(-Params.a_2[0])
        ys_top.append(-Params.a_2[1])

        xs_top.append(Params.a_2[0])
        ys_top.append(Params.a_2[1])

        xs_top.append(- Params.a_1[0] - Params.a_2[0])
        ys_top.append(- Params.a_1[1] - Params.a_2[1])

        xs_top.append(Params.a_1[0] + Params.a_2[0])
        ys_top.append(Params.a_1[1] + Params.a_2[1])

        # Second hex
        xs_top.append(2 * Params.a_1[0] + Params.a_2[0])
        ys_top.append(2 * Params.a_1[1] + Params.a_2[1])

        xs_top.append(Params.a_1[0] + 2 * Params.a_2[0])
        ys_top.append(Params.a_1[1] + 2 * Params.a_2[1])

        xs_top.append(-Params.a_1[0] + Params.a_2[0])
        ys_top.append(-Params.a_1[1] + Params.a_2[1])

        xs_top.append(Params.a_1[0] - Params.a_2[0])
        ys_top.append(Params.a_1[1] - Params.a_2[1])

        xs_top.append(-2 * Params.a_1[0] - Params.a_2[0])
        ys_top.append(-2 * Params.a_1[1] - Params.a_2[1])

        xs_top.append(-Params.a_1[0] - 2 * Params.a_2[0])
        ys_top.append(-Params.a_1[1] - 2 * Params.a_2[1])

        ax.scatter(xs_top, ys_top, label='Nb')

    if show_se:
        xs_top = []
        ys_top = []

        xs_bottom = []
        ys_bottom = []

        s1 = np.matrix((1 / 3) * Params.a_1 + (2 / 3) * Params.a_2)
        s2 = np.matrix((-2 / 3) * Params.a_1 - (1 / 3) * Params.a_2)
        s3 = np.matrix((1 / 3) * Params.a_1 + (-1 / 3) * Params.a_2)

        # Rotate where Se atoms are
        theta = np.pi / 6
        theta = 0
        R_z = np.matrix(((np.cos(theta), -np.sin(theta)),
                         (np.sin(theta), np.cos(theta))))

        s1 = R_z * s1.transpose()
        s2 = R_z * s2.transpose()
        s3 = R_z * s3.transpose()

        # Top first
        xs_top.append(s1[0])
        ys_top.append(s1[1])

        xs_top.append(s2[0])
        ys_top.append(s2[1])

        xs_top.append(s3[0])
        ys_top.append(s3[1])

        # Bottom first
        xs_bottom.append(-s1[0])
        ys_bottom.append(-s1[1])

        xs_bottom.append(-s2[0])
        ys_bottom.append(-s2[1])

        xs_bottom.append(-s3[0])
        ys_bottom.append(-s3[1])

        # Bottom second
        xs_bottom.append(2 * s1[0])
        ys_bottom.append(2 * s1[1])

        xs_bottom.append(2 * s2[0])
        ys_bottom.append(2 * s2[1])

        xs_bottom.append(2 * s3[0])
        ys_bottom.append(2 * s3[1])

        # Top second
        xs_top.append(-2 * s1[0])
        ys_top.append(-2 * s1[1])

        xs_top.append(-2 * s2[0])
        ys_top.append(-2 * s2[1])

        xs_top.append(-2 * s3[0])  # xs.append(-Params.a_1[0] - 2 * Params.a_2[0] + s1[0])
        ys_top.append(-2 * s3[1])  # ys.append(-Params.a_1[1] - 2 * Params.a_2[1] + s1[1])

        ax.scatter(xs_top, ys_top, color='#38cf5b', label='Se (upper)') # 88c999
        ax.scatter(xs_bottom, ys_bottom, color='#a7c9af', label='Se (lower)')


    fig.colorbar(c, ax=ax, label='')  # TODO: change/try a logarithmic color scale/bar .
    fig.gca().set_aspect('equal', adjustable='box')
    fig.tight_layout()
    fig.savefig(filename, rasterized=True, bbox_inches='tight', dpi=300)
    fig.legend()
    # fig.show()
    plt.close(fig)
    print('heat map plot: \n {} \n'.format(filename))


def gaussian(x, y, a):
    r2 = x**2 + y**2
    return np.exp(-r2/a)


def plot_fermi_surface_with_gaussian():
    H_arr_R, rs, weights = get_hamiltonian('../data/Hamiltonian.dat')

    # real lattice vectors
    a_0 = 3.44  # \AA (Angstrom)
    a_1 = a_0 * np.array([np.sqrt(3) / 2, -1 / 2])
    a_2 = a_0 * np.array([0, 1])

    # reciprocal lattice vectors
    b_1 = (2 * np.pi / a_0) * np.array([2 * np.sqrt(3) / 3, 0])
    b_2 = (2 * np.pi / a_0) * np.array([np.sqrt(3) / 3, 1])

    # Fermi-energy
    E_f = -3.9285  # eV
    dE_f = -50 * pow(10, -3)

    E_f = E_f + dE_f

    # make k-space (to be used in constructing H(k))
    kx_n = 50
    ky_n = 50
    kxs, kys = generate_axes(kx_n, ky_n, 4.2, 4.2)

    # make mesh to hold hamiltonians(k) (mesh of 2*2 matrices)
    Hs_k = np.matrix(np.zeros((kx_n, ky_n), dtype=np.matrix))

    es_lower = np.matrix(np.zeros((kx_n, ky_n), dtype=float))
    es_upper = np.matrix(np.zeros((kx_n, ky_n), dtype=float))

    for i in range(kx_n):
        for j in range(ky_n):
            Hs_k[i, j] = calculate_H_k(H_arr_R, rs, weights, a_1=a_1, a_2=a_2, k=np.array([kxs[i], kys[j]]))

            _, e_values = extract_eigenstates_from_hamiltonian(Hs_k[i, j])

            e_values = e_values.real

            es_lower[i, j] = min(e_values)
            es_upper[i, j] = max(e_values)

        if i % 10 == 0:
            print('finished ', i)

    for i in range(1, 20):
        # Plot gaussian
        alpha_gauss = i * 0.5
        phi_k = np.zeros((kx_n, ky_n), dtype=float)
        for i in range(kx_n):
            for j in range(ky_n):
                phi_k[i, j] = gaussian(kxs[i], kys[j], a=alpha_gauss)**2

        fig, ax = plt.subplots()
        c = ax.pcolormesh(kxs, kys, phi_k.transpose(), cmap='hot')

        # plot fermi-surface
        c_upper = ax.contour(kxs, kys, es_upper.transpose(), levels=[E_f], colors=['#ff7f0e'], linestyles='solid')
        c_lower = ax.contour(kxs, kys, es_lower.transpose(), levels=[E_f], colors=['#1f77b4'], linestyles='solid')
        ax.set_xlabel('$k_x$ $(\AA^{-1})$')
        ax.set_ylabel('$k_y$ $(\AA^{-1})$')

        c_upper.collections[0].set_label('$E_f = ${} eV (upper)'.format(E_f))

        c_lower.collections[0].set_label('$E_f = ${} eV (lower)'.format(E_f))

        # Plot BZ
        a_0 = 3.44  # \AA (Angstrom)
        a_1 = a_0 * np.array([np.sqrt(3) / 2, -1 / 2])
        a_2 = a_0 * np.array([0, 1])

        b_1 = (2 * np.pi / a_0) * np.array([2 * np.sqrt(3) / 3, 0])
        b_2 = (2 * np.pi / a_0) * np.array([np.sqrt(3) / 3, 1])

        M = 0.5 * b_1
        K = np.array([0.5 * b_1[0], 0.5 * b_1[0] * np.tan(30 * np.pi / 180)])

        cs = [K, (K[0], -K[1]), -2 * K + 2 * M, (-K[0], -K[1]), (-K[0], K[1]), 2 * K - 2 * M, K]
        xs_brillouin = np.array([])
        ys_brillouin = np.array([])

        for coord in cs:
            xs_brillouin = np.append(xs_brillouin, coord[0])
            ys_brillouin = np.append(ys_brillouin, coord[1])

        ax.plot(xs_brillouin, ys_brillouin, color='gray', linestyle='dashed')

        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=2)
        # ax.legend(loc='upper center')
        fig.tight_layout()
        fig.colorbar(c, ax=ax, label='')
        fig.gca().set_aspect('equal', adjustable='box')
        filepath = '../plots/contour_with_bz_with_gauss_kxn={}_dEf={}_a={}_gauss-squared.pdf'.format(kx_n, dE_f, alpha_gauss)
        print('filepath:\n', filepath)
        fig.savefig(filepath)
        fig.show()
        plt.close()


def plot_fermi_surface_with_gaussian_and_edge_plots():
    H_arr_R, rs, weights = get_hamiltonian('../data/Hamiltonian.dat')

    # real lattice vectors
    a_0 = 3.44  # \AA (Angstrom)
    a_1 = a_0 * np.array([np.sqrt(3) / 2, -1 / 2])
    a_2 = a_0 * np.array([0, 1])

    # reciprocal lattice vectors
    b_1 = (2 * np.pi / a_0) * np.array([2 * np.sqrt(3) / 3, 0])
    b_2 = (2 * np.pi / a_0) * np.array([np.sqrt(3) / 3, 1])

    # Fermi-energy
    E_f = -3.9285  # eV
    dE_f = -70 * pow(10, -3)

    E_f = E_f + dE_f

    # make k-space (to be used in constructing H(k))
    kx_n = 30
    ky_n = 30
    kxs, kys = generate_axes(kx_n, ky_n, 4.2, 4.2)

    # make mesh to hold hamiltonians(k) (mesh of 2*2 matrices)
    Hs_k = np.matrix(np.zeros((kx_n, ky_n), dtype=np.matrix))

    es_lower = np.zeros((kx_n, ky_n), dtype=float)
    es_upper = np.zeros((kx_n, ky_n), dtype=float)

    for i in range(kx_n):
        for j in range(ky_n):
            Hs_k[i, j] = calculate_H_k(H_arr_R, rs, weights, a_1=a_1, a_2=a_2, k=np.array([kxs[i], kys[j]]))

            _, e_values = extract_eigenstates_from_hamiltonian(Hs_k[i, j])

            e_values = e_values.real

            es_lower[i, j] = min(e_values)
            es_upper[i, j] = max(e_values)

        if i % 10 == 0:
            print('finished ', i)

    # es_lower_z = np.zeros((kx_n, ky_n), dtype=float)
    # for i in range(kx_n):
    #     for j in range(ky_n):
    #         es_lower_z[i, j] = es_lower[i, j]

    # plot_heatmap(kxs, kys, es_lower, filename='../plots/es_lower_kxn={}_dEf={}.pdf'.format(kx_n, dE_f))
    # plot_heatmap(kxs, kys, es_upper, filename='../plots/es_upper_kxn={}_dEf={}.pdf'.format(kx_n, dE_f))
    # exit()

    a_0 = 3.44  # \AA (Angstrom)
    a_1 = a_0 * np.array([np.sqrt(3) / 2, -1 / 2])
    a_2 = a_0 * np.array([0, 1])

    b_1 = (2 * np.pi / a_0) * np.array([2 * np.sqrt(3) / 3, 0])
    b_2 = (2 * np.pi / a_0) * np.array([np.sqrt(3) / 3, 1])

    M = 0.5 * b_1
    K = np.array([0.5 * b_1[0], 0.5 * b_1[0] * np.tan(30 * np.pi / 180)])

    lower_k = 0
    mid_k = 0.65
    upper_k = 1.4

    inner_v = 1
    outer_v = 0.58

    k_upper = 0.5
    k_val = 1

    for i in range(1, 2):
        # Plot gaussian
        alpha_gauss = 0.3
        phi_k = np.zeros((kx_n, ky_n), dtype=float)
        for i in range(kx_n):
            for j in range(ky_n):
                kx = kxs[i] - K[0]
                ky = kys[j] - K[1]
                k = np.sqrt(kx ** 2 + ky ** 2)

                # if lower_k < k <= mid_k:
                #     phi_k[i, j] = inner_v
                # elif mid_k < k <= upper_k:
                #     phi_k[i, j] = outer_v
                # else:
                #     phi_k[i, j] = 0

                if 0 < k < k_upper:
                    phi_k[i, j] = k_val

                ##
                kx = kxs[i] + K[0]
                ky = kys[j] + K[1]
                k = np.sqrt(kx ** 2 + ky ** 2)

                if 0 < k < k_upper:
                    phi_k[i, j] = k_val

                kx = kxs[i] + K[0]
                ky = kys[j] - K[1]
                k = np.sqrt(kx ** 2 + ky ** 2)

                if 0 < k < k_upper:
                    phi_k[i, j] = k_val

                kx = kxs[i] - K[0]
                ky = kys[j] + K[1]
                k = np.sqrt(kx ** 2 + ky ** 2)

                if 0 < k < 0.5:
                    phi_k[i, j] = k_val

                kx = kxs[i] - 0
                ky = kys[j] + np.sqrt(K[0]**2 + K[1]**2)
                k = np.sqrt(kx ** 2 + ky ** 2)

                if 0 < k < k_upper:
                    phi_k[i, j] = k_val

                kx = kxs[i] - 0
                ky = kys[j] - np.sqrt(K[0]**2 + K[1]**2)
                k = np.sqrt(kx ** 2 + ky ** 2)

                if 0 < k < k_upper:
                    phi_k[i, j] = k_val

                # else:
                #     phi_k[i, j] = 0
                #
                # phi_k[i, j] = np.exp(-((k-1.2)**2)/alpha_gauss)

                # phi_k[i, j] = gaussian(kxs[i], kys[j], a=alpha_gauss)**2


        fig, ax = plt.subplots()
        # gs = gridspec.GridSpec(2, 1)
        # plt.subplot(gs[0])
        c = ax.pcolormesh(kxs, kys, phi_k.transpose(), cmap='hot')

        # plot fermi-surface
        c_upper = ax.contour(kxs, kys, es_upper.transpose(), levels=[E_f], colors=['#ff7f0e'], linestyles='solid')
        c_lower = ax.contour(kxs, kys, es_lower.transpose(), levels=[E_f], colors=['#1f77b4'], linestyles='solid')
        ax.set_xlabel('$k_x$ $(\AA^{-1})$')
        ax.set_ylabel('$k_y$ $(\AA^{-1})$')

        c_upper.collections[0].set_label('$E_f = ${} eV (upper)'.format(E_f))

        c_lower.collections[0].set_label('$E_f = ${} eV (lower)'.format(E_f))

        # Plot BZ
        a_0 = 3.44  # \AA (Angstrom)
        a_1 = a_0 * np.array([np.sqrt(3) / 2, -1 / 2])
        a_2 = a_0 * np.array([0, 1])

        b_1 = (2 * np.pi / a_0) * np.array([2 * np.sqrt(3) / 3, 0])
        b_2 = (2 * np.pi / a_0) * np.array([np.sqrt(3) / 3, 1])

        M = 0.5 * b_1
        K = np.array([0.5 * b_1[0], 0.5 * b_1[0] * np.tan(30 * np.pi / 180)])

        cs = [K, (K[0], -K[1]), -2 * K + 2 * M, (-K[0], -K[1]), (-K[0], K[1]), 2 * K - 2 * M, K]
        xs_brillouin = np.array([])
        ys_brillouin = np.array([])

        for coord in cs:
            xs_brillouin = np.append(xs_brillouin, coord[0])
            ys_brillouin = np.append(ys_brillouin, coord[1])

        ax.plot(xs_brillouin, ys_brillouin, color='gray', linestyle='dashed')

        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=2)
        # ax.legend(loc='upper center')

        fig.tight_layout()
        fig.colorbar(c, ax=ax, label='')
        fig.gca().set_aspect('equal', adjustable='box')

        # plt.subplot(gs[1])
        # plt.plot([0, 1], [0, 1])

        filepath = '../plots/contour_with_bz_with_gauss_kxn={}_dEf={}_gauss-squared_a={}_4_{}-k-{}={}_{}-k-{}={}__{}.pdf'\
            .format(kx_n, dE_f, alpha_gauss, lower_k, mid_k, inner_v, mid_k, upper_k, outer_v, VERSION_NUMBER)
        print('filepath:\n', filepath)
        fig.savefig(filepath)
        # fig.show()
        plt.close()

        gauss_ys = np.zeros(kx_n, dtype=float)
        for i in range(kx_n):
            gauss_ys[i] = gaussian(kxs[i], 0, a=alpha_gauss)**2

        plt.plot(kxs, gauss_ys)
        filepath_gauss = '../plots/gauss-squared__1D_slice_kxn={}_dEf={}_a={}__{}.pdf'.format(kx_n, dE_f, alpha_gauss, VERSION_NUMBER)
        print('filepath_gauss: \n', filepath_gauss)
        plt.savefig(filepath_gauss)


def plot_heatmap(xs, ys, zs, filename, show_bz=False):
    fig, ax = plt.subplots()

    c = ax.pcolormesh(xs, ys, zs, cmap='hot')
    ax.set_xlabel('$q_x$ $(\AA^{-1})$')
    ax.set_ylabel('$q_y$ $(\AA^{-1})$')

    if show_bz:
        # K = Params.K
        # M = Params.M
        #
        # cs = [Params.K, (K[0], -K[1]), -2 * K + 2 * M, (-K[0], -K[1]), (-K[0], K[1]), 2 * K - 2 * M, Params.K]
        # xs_brillouin = np.array([])
        # ys_brillouin = np.array([])
        #
        # for coord in cs:
        #     xs_brillouin = np.append(xs_brillouin, coord[0])
        #     ys_brillouin = np.append(ys_brillouin, coord[1])
        #
        # ax.plot(xs_brillouin, ys_brillouin, color='gray', linestyle='dashed')  # , zorder=1
        #
        # xs_brillouin = np.array([])
        # ys_brillouin = np.array([])
        #
        # for coord in cs:
        #     xs_brillouin = np.append(xs_brillouin, coord[0]*2)
        #     ys_brillouin = np.append(ys_brillouin, coord[1]*2)
        #
        # ax.plot(xs_brillouin, ys_brillouin, color='gray', linestyle='dashed')  # , zorder=1
        pass

    fig.colorbar(c, ax=ax, label='')  # TODO: change/try a logarithmic color scale/bar .
    fig.gca().set_aspect('equal', adjustable='box')
    fig.tight_layout()
    fig.savefig(filename, rasterized=True)
    # fig.show()
    plt.close(fig)
    print('heat map plot: \n {} \n'.format(filename))


def plot_H_k_elements():
    from plotting import plot_combined_H_k_els
    from plotting import plot_heatmap

    kxn, kyn = 100, 100
    kx_length, ky_length = 8 * np.pi / Params.a_0, 8 * np.pi / Params.a_0
    # kx_length, ky_length = 0.05, 0.1
    # kx_centre, ky_centre = -0.37, -0.33
    kxs, kys = generate_axes(kxn, kyn, kx_length, ky_length)
    # kxs = np.linspace(kx_centre - kx_length / 2, kx_centre + kx_length / 2, kxn)
    # kys = np.linspace(ky_centre - ky_length / 2, ky_centre + ky_length / 2, kyn)
    kx_centre, ky_centre = 0, 0

    H_arr_R, rs, weights = get_hamiltonian('../data/Hamiltonian.dat')

    for m in range(2):
        for n in range(2):
            h_k_mn = np.zeros((kxn, kyn), dtype=complex)
            for i in range(kxn):
                for j in range(kyn):
                    h = calculate_H_k(H_arr_R, rs, weights, Params.a_1, Params.a_2, [kxs[i], kys[j]])
                    h_k_mn[i, j] = h[m, n]

            # filename = '../plots/ham/h_k_{}{}_k-centre={},{}_k-length={},{}_kxn={}'.format(m, n, kx_centre, ky_centre, kx_length, ky_length, kxn)
            filename = '../plots/ham/H_k/h_k_{}{}'.format(m, n)
            plot_heatmap(kxs, kys, h_k_mn.transpose().imag, filename=filename+'_imag.png', x_label='$k_x (\AA^{-1})$', y_label='$k_y (\AA^{-1})$', show_bz=True)
            plot_heatmap(kxs, kys, h_k_mn.transpose().real, filename=filename + '_real.png', x_label='$k_x (\AA^{-1})$',
                         y_label='$k_y (\AA^{-1})$', show_bz=True)
            plot_heatmap(kxs, kys, np.abs(h_k_mn.transpose()), filename=filename + '_abs.png', x_label='$k_x (\AA^{-1})$',
                         y_label='$k_y (\AA^{-1})$', show_bz=True)
            plot_combined_H_k_els(m, n)


def plot_H_r_elements():
    from plotting import plot_scatter_heatmap
    from plotting import plot_combined_H_r_els

    H_arr_R, rs, weights = get_hamiltonian('../data/Hamiltonian.dat')
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


def main():
    print('NbSe2 - Hamiltonian investigation')
    # plot_bands()
    start_time = time.time()

    plot_H_r_elements()
    plot_H_k_elements()
    # plot_fermi_surface_with_gaussian_and_edge_plots()
    # plot_bands()
    # plot_fermi_surface()
    time_end = time.time()
    dt = time_end - start_time
    print('time taken: {:.6f} seconds'.format(dt))


VERSION_NUMBER = '1.1.2_phi=K_k0.5'


if __name__ == '__main__':
    main()

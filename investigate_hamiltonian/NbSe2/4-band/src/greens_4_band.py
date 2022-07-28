import numpy as np
import os
from params import *
from files import FilePaths
from hamiltonian_4band import calculate_4_band_H_k
from hamiltonian_4band import get_4_band_hamiltonian
from hamiltonian_4band import extract_eigenstates_from_hamiltonian


def g_n(psi_n, phi, omega, eta, epsilon_n):
    psi_n_H = psi_n.getH()

    # numerator |\psi><\psi|
    outer = psi_n * psi_n_H

    numerator = outer * np.conjugate(phi) * phi

    # g_n(k,\omega)^-1 =  \omega + i\eta - \epsilon
    g = omega + 1j * eta - epsilon_n

    return numerator / g


def G_0(eigenvectors, eigenvalues, phi, omega, eta=0.01):
    G = np.matrix(np.zeros((4, 4), dtype=complex))
    for i in range(len(eigenvalues)):
        G = G + g_n(eigenvectors[i], phi=phi, omega=omega, eta=eta, epsilon_n=eigenvalues[i])
    return G


def generate_4_band_bare_greens(kxs, kys, phi_k):
    from files import get_4_bands_greens_data
    from files import write_4_bands_greens_data

    print("*** generate_Greens")

    omega = Params.omega

    kxn = len(kxs)
    kyn = len(kys)

    filepath = FilePaths.greens_cache + 'greens_4_band_NbSe2_omega={}_kxn={}_kyn={}_kxrange=({},{})_kyrange=({},{}).dat'\
        .format(Params.omega, kxn, kyn, min(kxs), max(kxs), min(kys), max(kys))

    def generate_greens():
        H_arr_R, rs, weights = get_4_band_hamiltonian(FilePaths.hamiltonian_NbSe2_4band_data)
        Gs = np.matrix(np.zeros((kxn, kyn), dtype=np.matrix))

        for i in range(kxn):  # Number of "Pixels" along the kx axis
            for j in range(kyn):  # Number of "Pixels" along the ky axis
                kx = kxs[i]
                ky = kys[j]

                phi_kx_ky = 1

                H_k = calculate_4_band_H_k(H_arr_R, rs, weights=weights, kx=kx, ky=ky)

                eigenvectors, eigenvalues = extract_eigenstates_from_hamiltonian(H_k)

                # Construct bare Green's function, G_0(k,\omega)
                Gs[i, j] = G_0(eigenvectors=eigenvectors, eigenvalues=eigenvalues, phi=phi_kx_ky,
                               omega=omega)  # 2*2 matrix

            if i % 20 == 0:
                print('finished column: {}'.format(i))

        return Gs

    Gs = np.matrix(np.zeros((kxn, kyn), dtype=np.matrix))

    if RunParams.USE_CACHE:
        # check if greens file of requested resolution exists
        if os.path.exists(filepath):
            print('reading greens from: ', filepath)
            Gs = get_4_bands_greens_data(filepath)
        else:
            print('path does not exist, so calculating greens and saving to {}'.format(filepath))
            Gs = generate_greens()
            write_4_bands_greens_data(filepath, kxn, kyn, Gs)
    else:
        Gs = generate_greens()

    # apply k-space modulated weighting phi(k) (each eigenstate weighted equally (dependent on k))
    for i in range(kxn):
        for j in range(kyn):
            Gs[i, j] = Gs[i, j] * phi_k[i, j] * phi_k[i, j].conjugate()

    return Gs


def generate_4_band_bare_greens_eigenstate_weighting(kxs, kys, phi_k):
    '''

    :param kxs:
    :param kys:
    :param phi_k: a 4 element (column) vector of weightings of each eigenstate defined at each k point (a (kxn, kyn) matrix of (1, 4) column vectors)
    :return: G_0(k)
    '''
    from files import get_4_bands_greens_data
    from files import write_4_bands_greens_data

    print("*** generate_Greens")

    omega = Params.omega

    kxn = len(kxs)
    kyn = len(kys)

    phi_str = ''
    for p in Params.eigen_phi:
        phi_str += str(p)

    filepath = FilePaths.greens_cache + 'greens_4_band_NbSe2_omega={}_kxn={}_kyn={}_kxrange=({},{})_kyrange=({},{}).dat'\
        .format(Params.omega, kxn, kyn, min(kxs), max(kxs), min(kys), max(kys), phi_str)

    def generate_greens():
        H_arr_R, rs, weights = get_4_band_hamiltonian(FilePaths.hamiltonian_NbSe2_4band_data)
        Gs = np.matrix(np.zeros((kxn, kyn), dtype=np.matrix))

        for i in range(kxn):  # Number of "Pixels" along the kx axis
            for j in range(kyn):  # Number of "Pixels" along the ky axis
                kx = kxs[i]
                ky = kys[j]

                phi_kx_ky = 1

                H_k = calculate_4_band_H_k(H_arr_R, rs, weights=weights, kx=kx, ky=ky)

                eigenvectors, eigenvalues = extract_eigenstates_from_hamiltonian(H_k)

                # Construct bare Green's function, G_0(k,\omega)
                Gs[i, j] = G_0(eigenvectors=eigenvectors, eigenvalues=eigenvalues, phi=phi_kx_ky,
                               omega=omega)  # 2*2 matrix

            if i % 20 == 0:
                print('finished column: {}'.format(i))

        return Gs

    Gs = np.matrix(np.zeros((kxn, kyn), dtype=np.matrix))

    if RunParams.USE_CACHE:
        # check if greens file of requested resoltion exists
        if os.path.exists(filepath):
            print('reading greens from: ', filepath)
            Gs = get_4_bands_greens_data(filepath)
        else:
            print('path does not exist, so calculating greens and saving to {}'.format(filepath))
            Gs = generate_greens()
            write_4_bands_greens_data(filepath, kxn, kyn, Gs)
    else:
        Gs = generate_greens()

    print('eigen_phi = {}'.format(Params.eigen_phi))
    print('eigenvalue_phi = {}'.format(Params.eigenvalue_phi))

    eigen_phi_vec = np.matrix(Params.eigen_phi)
    if eigen_phi_vec.shape[0] == 1 and eigen_phi_vec.shape[1] == 4:
        eigen_phi_vec = eigen_phi_vec.transpose()
    eigen_phi_mat = eigen_phi_vec * eigen_phi_vec.getH()
    print(eigen_phi_mat)
    # apply k-space modulated weighting phi(k) (each eigenstate weighted equally (dependent on k))
    for i in range(kxn):
        for j in range(kyn):
            for m in range(Gs[0, 0].shape[0]):
                for n in range(Gs[0, 0].shape[1]):
                    Gs[i, j][m, n] = Gs[i, j][m, n] * phi_k[i, j] * phi_k[i, j].conjugate() * eigen_phi_mat[m, n]

    return Gs


def generate_4_band_bare_greens_den(kxs, kys):
    '''

    :param kxs:
    :param kys:
    :return: G_0_k_den(k)
    '''
    print("*** generate_Greens")

    omega = Params.omega

    kxn = len(kxs)
    kyn = len(kys)

    H_arr_R, rs, weights = get_4_band_hamiltonian(FilePaths.hamiltonian_NbSe2_4band_data)

    gs = np.zeros((kxn, kyn), dtype=complex)

    eta = 0.01

    for i in range(kxn):  # Number of "Pixels" along the kx axis
        for j in range(kyn):  # Number of "Pixels" along the ky axis
            kx = kxs[i]
            ky = kys[j]

            H_k = calculate_4_band_H_k(H_arr_R, rs, weights=weights, kx=kx, ky=ky)

            eigenvectors, eigenvalues = extract_eigenstates_from_hamiltonian(H_k)
            for m in range(len(eigenvalues)):
                eigenvalues[m] *= Params.eigenvalue_phi[m]

            for eig in eigenvalues:
                gs[i, j] += 1 / (omega + 1j * eta - eig)

        if i % 20 == 0:
            print('finished column: {}'.format(i))

    return gs


def generate_4_band_bare_greens_num(kxs, kys):
    '''

    :param kxs:
    :param kys:
    :return: G_0(k)
    '''
    print("*** generate_Greens")

    omega = Params.omega

    kxn = len(kxs)
    kyn = len(kys)

    phi_str = ''
    for p in Params.eigen_phi:
        phi_str += str(p)

    eigen_phi_vec = np.matrix(Params.eigen_phi)
    if eigen_phi_vec.shape[0] == 1 and eigen_phi_vec.shape[1] == 4:
        eigen_phi_vec = eigen_phi_vec.transpose()
    eigen_phi_mat = eigen_phi_vec * eigen_phi_vec.getH()

    H_arr_R, rs, weights = get_4_band_hamiltonian(FilePaths.hamiltonian_NbSe2_4band_data)
    Gs_num = np.matrix(np.zeros((kxn, kyn), dtype=np.matrix))

    for i in range(kxn):  # Number of "Pixels" along the kx axis
        for j in range(kyn):  # Number of "Pixels" along the ky axis
            kx = kxs[i]
            ky = kys[j]

            H_k = calculate_4_band_H_k(H_arr_R, rs, weights=weights, kx=kx, ky=ky)

            eigenvectors, eigenvalues = extract_eigenstates_from_hamiltonian(H_k)

            Gs_num[i, j] = np.zeros((4, 4), dtype=complex)

            # Construct bare Green's function, G_0(k,\omega)
            for n in range(len(eigenvalues)):
                for m in range(len(eigenvectors[n])):
                    eigenvectors[n][m] *= Params.eigen_phi[m]
                psi_n = eigenvectors[n]
                psi_n_H = psi_n.getH()

                # numerator |\psi><\psi|
                outer = psi_n * psi_n_H
                Gs_num[i, j] += outer

        if i % 20 == 0:
            print('finished column: {}'.format(i))

    return Gs_num

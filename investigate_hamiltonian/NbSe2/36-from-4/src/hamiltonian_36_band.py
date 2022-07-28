import numpy as np
from scipy import linalg
from params import *
from files import *


def extract_floats_from_line(line):
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
        if (numpy_mat.shape[0] == 1) and (numpy_mat.shape[1] == len(eigenvalues)):
            numpy_mat = numpy_mat.transpose()

        eigenvectors[i] = numpy_mat

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


def construct_36_band_hamiltonian_from_4_band(H_arr_R, rs, R):
    '''
    Calculates the the H(R) 36*36 matrix from the 4 band model for a given R (of the new 3*3 unit cell)
    returns H_R (or None if R is invalid (/not specified in param rs))
    :param H_arr_R:
    :param rs:
    :param R: site vector of the origin of the super lattice cell
    '''

    H_R = np.zeros((36, 36), dtype=complex)

    def atom_number_to_r(atom_number):  # small r (in a_1 a_2 basis) (relates atom numbering to r (of unit cell)) (atom number withing unit cell)
        return [int(atom_number/3), atom_number % 3]

    # find min and max of the defined rs (of the pristine lattice) in the {a1, a2} basis
    max_0 = int(rs[0][0])
    min_0 = int(rs[0][0])
    max_1 = int(rs[0][1])
    min_1 = int(rs[0][1])
    for r in rs:
        if r[0] > max_0:
            max_0 = int(r[0])
        if r[1] > max_1:
            max_1 = int(r[1])

        if r[0] < min_0:
            min_0 = int(r[0])
        if r[1] < min_1:
            min_1 = int(r[1])

    # add padding to be able to include points not defined
    max_0 += 6
    min_0 -= 6

    max_1 += 6
    min_1 -= 6

    # create mapping between coordinate (in {a1, a2} basis) to the index of where it's defined in the array
    indices = np.full((max_0 - min_0 + 1, max_1 - min_0 + 1), -1, dtype=int)

    for i in range(len(rs)):
        indices[int(rs[i][0])-min_0, int(rs[i][1])-min_1] = i

    # Iterate of the 36*36 new hamiltonian (broken into 4*4 blocks (hence 9*9 of 4*4 at each index))
    for i in range(9):
        for j in range(9):
            # r vector (in a1 and a2 basis) of the 4*4 Hamiltonian at this element index of the new 36*36 Hamiltonian
            n_0 = atom_number_to_r(j)[0] - atom_number_to_r(i)[0] + R[0]
            n_1 = atom_number_to_r(j)[1] - atom_number_to_r(i)[1] + R[1]

            # OLD: check if new hamiltonian index has a possibility of being valid
            # check if hamiltonian is in included range (my still not be defined, but will be set to zero)
            if int(n_0)-min_0 < 0 or int(n_0)-min_0 > max_0 - min_0 or int(n_1)-min_1 < 0 or int(n_1)-min_1 > max_1 - min_0:
                return None

            index = indices[int(n_0)-min_0, int(n_1)-min_1]

            h = None
            if index == -1:
                h = np.zeros((4, 4), dtype=complex)  # not defined, so set to zero
            else:
                h = H_arr_R[index]

            for k in range(4):
                for l in range(4):
                    H_R[i * 4 + k, j * 4 + l] = h[k, l]

    # check H(R) != 0
    for i in range(36):
        for j in range(36):
            if H_R[i, j] != 0:
                return H_R

    return None


def calculate_36_band_H_k_from_supercell(ham_arr_R, rs, kx, ky):
    '''
    Calculates the 36 band supercell Hamiltonian in k-space for a given k-point
    giving the supercell 36 band Hamiltonian in real space with the real-coords
    of where each H(r) is defined.

    :param ham_arr_R:  Supercell 36 band Hamiltonian
    :param rs: in a1 & a2 basis (of the supercell origins
    :param kx:
    :param ky:
    :return:
    '''
    # real lattice vectors
    # a_0 = Params.a_0
    a_1 = Params.a_1
    a_2 = Params.a_2

    h_k = np.zeros(ham_arr_R[0].shape, dtype=complex)

    k = np.array([kx, ky])

    for r_i in range(len(rs)):
        r = rs[r_i]
        R = r[0] * a_1 + r[1] * a_2

        phase = -1j * np.dot(k, R)

        h_k += ham_arr_R[r_i] * np.exp(phase)

    return h_k


def get_supercell_36band_hamiltonian():
    print('*** supercell Hamiltoian')
    from hamiltonian_4band import get_4band_hamiltonian
    H_4_band_arr_R, rs, _ = get_4band_hamiltonian(FilePaths.hamiltonian_NbSe2_4band_data)

    new_H_R_array = []
    new_Rs = []

    total_valid = 0
    for i in range(-20, 20):
        for j in range(-20, 20):
            R = [i * 3, j * 3]
            H = construct_36_band_hamiltonian_from_4_band(H_4_band_arr_R, rs, R)

            if H is not None:
                total_valid += 1
                new_H_R_array.append(H)
                new_Rs.append(R)

    print('{} Rs'.format(len(new_Rs)))

    xs = []
    ys = []

    new_xs = []
    new_ys = []

    for r in rs:
        xs.append(r[0])
        ys.append(r[1])

    for r in new_Rs:
        new_xs.append(r[0])
        new_ys.append(r[1])

    import matplotlib.pyplot as plt
    plt.scatter(xs, ys)
    plt.scatter(new_xs, new_ys)
    plt.savefig('scatter_of_points.png')

    return new_H_R_array, new_Rs
import numpy as np
from scipy import linalg
from params import *



def extract_floats_from_line(line):
    els = line.split(' ')

    vs = np.array([])

    for e in els:
        if e == '' or e == '\n':  # check for '\n' might not be needed (but added just in case)
            continue

        vs = np.append(vs, float(e))

    return vs


def get_hamiltonian(filepath):
    print('*** get hamiltonian: {}'.format(filepath))

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

        vs = extract_floats_from_line(line)

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


def construct_18_from_2(H_arr_R, rs, R):
    '''
    Calculates the the H(R) 18*18 matrix from the 2 band model for a given R (of the new 3*3 unit cell)
    returns H_R, and 0 = OK, 1 = invalid/failed
    :param H_arr_R:
    :param rs:
    :param R: site vector of the origin of the super lattice cell
    '''

    H_R = np.zeros((18, 18), dtype=complex)

    def atom_number_to_r(atom_number):  # small r (in a_1 a_2 basis) (relates atom numbering to r (of unit cell))
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
        indices[int(rs[i][0]) - min_0, int(rs[i][1]) - min_1] = i

    # Iterate of the 18*18 new hamiltonian (broken into 2*2 blocks (hence 9*9 of 2*2 at each index))
    for i in range(9):
        for j in range(9):
            # r vector (in a1 and a2 basis) of the 2*2 Hamiltonian at this element index of the new 18*18 Hamiltonian
            n_0 = atom_number_to_r(j)[0] - atom_number_to_r(i)[0] + R[0]
            n_1 = atom_number_to_r(j)[1] - atom_number_to_r(i)[1] + R[1]

            # check if hamiltonian is in included range (my still not be defined, but will be set to zero)
            if int(n_0) - min_0 < 0 or int(n_0) - min_0 > max_0 - min_0 or int(n_1) - min_1 < 0 or int(
                    n_1) - min_1 > max_1 - min_0:
                return None

            index = indices[int(n_0) - min_0, int(n_1) - min_1]

            h = None
            if index == -1:
                h = np.zeros((2, 2), dtype=complex)
            else:
                h = H_arr_R[index]

            for k in range(2):
                for l in range(2):
                    H_R[i * 2 + k, j * 2 + l] = h[k, l]

            # H_R[i * 2, j * 2] = h[0, 0]
            # H_R[i*2 + 1, j*2] = h[1, 0]
            # H_R[i*2, j*2 + 1] = h[0, 1]
            # H_R[i*2 + 1, j*2 + 1] = h[1, 1]

    # Check H(R) != 0
    for i in range(18):
        for j in range(18):
            if H_R[i, j] != 0:
                return H_R

    return None


def calculate_H_k(ham_arr_R, rs, weights, kx, ky):
    # real lattice vectors
    a_0 = Params.a_0
    a_1 = Params.a_1
    a_2 = Params.a_2
    # a_0 = 3.44  # \AA (Angstrom)
    # a_1 = a_0 * np.array([np.sqrt(3) / 2, -1 / 2])
    # a_2 = a_0 * np.array([0, 1])

    h_k = np.zeros((2, 2), dtype=complex)

    k = np.array([kx, ky])

    for r_i in range(len(rs)):
        r = rs[r_i]
        R = r[0] * a_1 + r[1] * a_2

        phase = -1j * np.dot(k, R)

        h_k += ham_arr_R[r_i] * np.exp(phase) / weights[r_i]

    return h_k


def calculate_H_k_from_supercell(ham_arr_R, rs, kx, ky):
    # real lattice vectors
    # a_0 = Params.a_0
    a_1 = Params.a_1   # *3 b/c of super-cell is 3*3 of unit cell
    a_2 = Params.a_2

    h_k = np.zeros(ham_arr_R[0].shape, dtype=complex)

    k = np.array([kx, ky])

    for r_i in range(len(rs)):
        r = rs[r_i]
        R = r[0] * a_1 + r[1] * a_2

        phase = -1j * np.dot(k, R)

        h_k += ham_arr_R[r_i] * np.exp(phase)

    return h_k


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
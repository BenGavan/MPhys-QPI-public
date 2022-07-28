from params import *


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
    from scipy import linalg
    eigenstates = linalg.eig(hamiltonian)

    eigenvalues = eigenstates[0]
    eigenvects = eigenstates[1]

    eigenvectors = [None] * len(eigenvects)

    for i in range(len(eigenvects)):
        numpy_mat = np.matrix(eigenvects[:, i])
        if (numpy_mat.shape[0] == 1) and (numpy_mat.shape[1] == 4):
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

    for i in range(n-1):
        if eigenvalues[i] > eigenvalues[i+1] or eigenvalues[i].imag > 0.0001:
            print(eigenvalues)
            exit()

    return eigenvectors, eigenvalues


def get_4_band_hamiltonian(filepath):
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
            number_bands = int(vs[0])
            print('Number of bands = {}'.format(number_bands))
            continue

        if line_num == 2 and len(vs) == 1:
            number_hopping_vectors = int(vs[0])
            print('# hopping vectors = {}'.format(number_hopping_vectors))

            H_arr_R = np.array([np.zeros((number_bands, number_bands), dtype=complex) for _ in range(number_hopping_vectors)])
            hopping_vectors = np.array([np.zeros(2, dtype=float) for _ in range(number_hopping_vectors)])
            continue

        if len(weights) != number_hopping_vectors:
            weights = np.append(weights, vs)
            continue

        H_line += 1
        H_arr_index = int(H_line/(number_bands**2))
        if H_line % (number_bands**2) == 0:  # only need to assign hopping vector at the start of each possition (repeasted 16(=4*4) times correspinding to the 4 elements of the hamilitonian at a given point)
            hopping_vectors[H_arr_index] = vs[:2]
        H_arr_R[H_arr_index][int(vs[3])-1, int(vs[4])-1] = vs[5] + 1j * vs[6]

    f.close()

    rs = hopping_vectors

    return H_arr_R, rs, weights


def calculate_4_band_H_k(ham_arr_R, rs, weights, kx, ky):
    # real lattice vectors
    a_1 = Params.a_1
    a_2 = Params.a_2

    h_k = np.zeros((4, 4), dtype=complex)

    k = np.array([kx, ky])

    for r_i in range(len(rs)):
        r = rs[r_i]
        R = r[0] * a_1 + r[1] * a_2

        phase = -1j * np.dot(k, R)

        h_k += ham_arr_R[r_i] * np.exp(phase) / weights[r_i]

    return h_k

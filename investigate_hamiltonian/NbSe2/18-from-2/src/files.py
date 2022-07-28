from params import *
from matrix_utils import *


class FilePaths:
    orbital_data = '../data/ld1.wfc'
    hamiltonian_NbSe2_data = '../data/Hamiltonian_2band.dat'
    bands_plots = '../plots/bands/'
    contour_plots = '../plots/contour/'


def append_arr_to_file(filename, vs):
    f = open(filename, 'a')

    for v in vs:
        f.write(str(v))
        f.write(' ')
    f.write('\n')

    f.close()


def append_matrix_to_file(filename, m):
    f = open(filename, 'a')

    xn = m.shape[0]
    yn = m.shape[1]

    for i in range(xn):
        for j in range(yn):
            f.write(str(m[i, j]))
            f.write(' ')
        f.write('\n')

    f.close()


def write_phi_r_data(filename, xn, yn, x_length, y_length, kxn, kyn, kx_length, ky_length, phi_r):
    f = open(filename, 'a')

    f.write(str(xn))
    f.write('\n')
    f.write(str(yn))
    f.write('\n')
    f.write(str(x_length))
    f.write('\n')
    f.write(str(y_length))
    f.write('\n')
    f.write(str(kxn))
    f.write('\n')
    f.write(str(kyn))
    f.write('\n')
    f.write(str(kx_length))
    f.write('\n')
    f.write(str(ky_length))
    f.write('\n')

    for i in range(xn):
        for j in range(yn):
            f.write(str(phi_r[i, j]))
            f.write(' ')
        f.write('\n')

    f.close()


def read_phi_k_data(filename):
    kxn = None
    kyn = None

    kx_length = None
    ky_length = None

    f = open(filename, 'r')

    phi_k = None

    line_counter = -1
    for line in f:
        line_counter += 1
        if line_counter <= 3:
            vs = extract_floats_from_line(line)
            if len(vs) == 1:
                if line_counter == 0:
                    kxn = int(vs[0])
                elif line_counter == 1:
                    kyn = int(vs[0])
                    phi_k = np.zeros((kxn, kyn), dtype=complex)
                elif line_counter == 2:
                    kx_length = float(vs[0])
                elif line_counter == 3:
                    ky_length = float(vs[0])
                continue

        vs = extract_complex_from_line(line)

        for i in range(len(vs)):
            phi_k[line_counter - 4][i] = vs[i]

    return kxn, kyn, kx_length, ky_length, phi_k


# appends to  'filename' xs, ys, zs, space seperated with new line between data
#  xs = kx sample values
#  ys = ky sample values
#  zs = Hamiltonian eigenvalues (singular (selected one only from pair (either min or max)))
def write_heatmap_data(filename, xs, ys, zs):
    print('writing data to: ', filename)
    append_arr_to_file(filename, xs)
    append_arr_to_file(filename, ys)
    append_matrix_to_file(filename, zs)


def get_heatmap_data(filename):
    xs = np.array([])
    ys = np.array([])
    zs = None

    f = open(filename, 'r')
    line_counter = 0
    for line in f:
        els = line.split(' ')[:-1]
        for i in range(len(els)):
            if line_counter == 0:
                xs = np.append(xs, float(els[i]))
            elif line_counter == 1:
                ys = np.append(ys, float(els[i]))
            else:
                zs[line_counter - 2][i] = float(els[i])

        if line_counter == 1:
            zs = np.zeros((len(xs), len(ys)))

        line_counter += 1

    f.close()

    return xs, ys, zs


def write_complex_data(filename, xs, ys, zs):
    print('writing data to: ', filename)
    append_arr_to_file(filename, xs)
    append_arr_to_file(filename, ys)
    append_matrix_to_file(filename, zs)


def get_complex_data(filename):
    xs = np.array([])
    ys = np.array([])
    zs = None

    f = open(filename, 'r')
    line_counter = 0
    for line in f:
        els = line.split(' ')[:-1]
        for i in range(len(els)):
            if line_counter == 0:
                xs = np.append(xs, float(els[i]))
            elif line_counter == 1:
                ys = np.append(ys, float(els[i]))
            else:
                zs[line_counter - 2][i] = complex(els[i])

        if line_counter == 1:
            zs = np.zeros((len(xs), len(ys)), dtype=complex)

        line_counter += 1

    f.close()

    return xs, ys, zs


def write_greens_data(filename, xs, ys, gs):
    print('writing data to: ', filename)
    append_arr_to_file(filename, xs)
    append_arr_to_file(filename, ys)
    gs_flat = flatten_mesh_of_matrices(gs)
    append_matrix_to_file(filename, gs_flat[0, 0])
    append_matrix_to_file(filename, gs_flat[0, 1])
    append_matrix_to_file(filename, gs_flat[1, 0])
    append_matrix_to_file(filename, gs_flat[1, 1])


def get_greens_data(filename):
    kxs = np.array([])
    kys = np.array([])
    gs_flat = np.matrix(np.zeros((2, 2), dtype=np.matrix))

    f = open(filename, 'r')
    line_counter = 0
    ky_i = 0
    for line in f:
        els = line.split(' ')[:-1]
        # print(line_counter, els)
        for i in range(len(els)):
            if line_counter == 0:
                kxs = np.append(kxs, float(els[i]))
            elif line_counter == 1:
                kys = np.append(kys, float(els[i]))
            else:
                g_i = int((line_counter - 2) / len(kxs))

                x_i = line_counter - 2 - (len(kxs)*g_i)
                gs_flat[g_i % 2, int((g_i - (g_i % 2)) / 2)][x_i, i] = complex(els[i])

        if line_counter > 1:
            ky_i += 1
            if (line_counter - 2) % len(kys) == 0:
                ky_i = 0

        if line_counter == 1:
            for i in range(2):
                for j in range(2):
                    gs_flat[i, j] = np.matrix(np.zeros((len(kxs), len(kys)), dtype=complex))

        line_counter += 1

    f.close()

    return kxs, kys, unflatten_to_mesh_of_matrices(gs_flat)


def extract_floats_from_line(line):
    els = line.split(' ')

    vs = np.array([])

    for e in els:
        if e == '' or e == '\n':  # check for '\n' might not be needed (but added just in case)
            continue

        vs = np.append(vs, float(e))

    return vs


def extract_complex_from_line(line):
    els = line.split(' ')

    vs = np.array([], dtype=complex)

    for e in els:
        if e == '' or e == '\n':  # check for '\n' might not be needed (but added just in case)
            continue

        c_str = e.split(',')

        r = float(c_str[0].split('(')[1])
        i = float(c_str[1].split(')')[0])

        vs = np.append(vs, r + 1j*i)
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


def get_orbital_data(filepath):
    f = open(filepath)

    rs = np.array([])
    Rs = np.array([])

    line_counter = -1

    R_n = 0

    for line in f:
        line_counter += 1
        if line_counter == 0:
            continue

        vs = extract_floats_from_line(line)

        r = vs[0]
        rR = vs[1:]

         # if you only want to plot upto a certain r value - modify here
        # if r > 10 or r < 3:
        #     continue

        R_n = len(rR)

        #  Data in the format of r, r*R(r) (we want r, R(r))
        R = rR / r

        rs = np.append(rs, r)
        Rs = np.append(Rs, R)

    return rs, Rs, R_n


def get_5s_orbital(filepath=FilePaths.orbital_data):
    rs, Rs, R_n = get_orbital_data(filepath)

    R_5s = [Rs[j] for j in range(2, len(Rs), R_n)]

    return rs, R_5s


def get_orbital(orbital, filepath=FilePaths.orbital_data):
    rs, Rs, R_n = get_orbital_data(filepath)

    R_5s = [Rs[j] for j in range(orbital.value, len(Rs), R_n)]

    return rs, R_5s


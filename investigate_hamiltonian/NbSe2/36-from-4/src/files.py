from params import *
from matrix_utils import *


class FilePaths:
    hamiltonian_NbSe2_2band_data = '../data/Hamiltonian_2band.dat'
    hamiltonian_NbSe2_4band_data = '../data/NbSe2_hr_(4-band).dat'
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

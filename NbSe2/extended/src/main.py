# from numba import jit
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.fftpack import ifft2
from scipy.fftpack import fft2
from scipy.fftpack import fftshift
# from scipy.fftpack import ifftshift
from scipy.interpolate import interp1d
import os
import enum
import time
import point_in_polygon as pp

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


class Scattering(enum.Enum):
    Scalar = 0
    Magnetic = 1
    Scalar_and_Spin_orbit = 2


class Orbital(enum.Enum):
    orbital_5S = 2
    orbital_5P = 1
    gaussian = -1


class Params:
    a_0 = 3.44  # \AA (Angstrom)
    a_1 = a_0 * np.array([np.sqrt(3) / 2, -1 / 2])
    a_2 = a_0 * np.array([0, 1])

    b_1 = (2 * np.pi / a_0) * np.array([2 * np.sqrt(3) / 3, 0])
    b_2 = (2 * np.pi / a_0) * np.array([np.sqrt(3) / 3, 1])

    M = 0.5 * b_1
    K = np.array([0.5 * b_1[0], 0.5 * b_1[0] * np.tan(30 * np.pi / 180)])

    c = 60
    v0 = 0.1
    E_f = -3.9285  # eV
    delta_E_f = -50 * pow(10, -3)  # eV (should be 50 meV but in eV)
    omega = E_f + delta_E_f

    # alpha_gauss = np.pi**2 / (2*a_0**2)
    alpha_gauss = 0.3

    lower_k = 0
    mid_k = 0.65
    upper_k = 1.58

    inner_v = 1
    outer_v = 1

    k_upper = 0.56
    k_val = 1

    Gam_upper = 0.65
    Gam_val = 1


class FilePaths:
    orbital_data = '../data/ld1.wfc'
    hamiltonian_NbSe2_data = ''
    greens_cache = '/Users/ben/Desktop/MPhys-QPI/project/QPI/NbSe2/extended/data/greens/'
    rho_cache = ''


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


def extract_values_from_line(line):
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

        vs = extract_values_from_line(line)

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


'''
Calculates the bare Green's function G_0(k, w) = \sigma_n \frac{}{omega + i\eta - \epsilon}
 - psi = eigenstate (2*2 matrix)
'''


def g_n(psi_n, phi, omega, eta, epsilon_n):
    psi_n_H = psi_n.getH()

    # numerator |\psi><\psi|
    outer = psi_n * psi_n_H

    # print('--------')
    # print('phi -------=')
    # print(phi)
    # print('np.conjugate(phi) -------=')
    # print(np.conjugate(phi))
    # print('--------')
    # exit()

    numerator = outer * np.conjugate(phi) * phi

    # g_n(k,\omega)^-1 =  \omega + i\eta - \epsilon
    g = omega + 1j * eta - epsilon_n

    return numerator / g


def G_0(eigenvectors, eigenvalues, phi, omega, eta=0.01):
    G = np.matrix([[0, 0], [0, 0]])
    for i in range(len(eigenvalues)):
        G = G + g_n(eigenvectors[i], phi=phi, omega=omega, eta=eta, epsilon_n=eigenvalues[i])
    return G


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
        numpy_mat = np.matrix(eigenvects[i])
        if (numpy_mat.shape[0] == 1) and (numpy_mat.shape[1] == 2):
            numpy_mat = numpy_mat.transpose()

        eigenvectors[i] = numpy_mat

    return eigenvectors, eigenvalues


def construct_m_matrix(scattering_type, Gs, kxs, kys):
    kx_n = len(kxs)
    ky_n = len(kys)

    U = U_matrix(scattering_type)

    u_G_u_sum = np.matrix(np.zeros((U.shape[0], U.shape[1]), dtype=complex))

    # u_G_u_p = u * 1
    for i in range(kx_n):
        for j in range(ky_n):
            kx = kxs[i]
            ky = kys[j]

            u = u_vector(scattering_type, kx, ky)

            u_G_u_sum = u_G_u_sum + u * Gs[i, j] * u.transpose()

    # normalize
    u_G_u_sum = u_G_u_sum / (kx_n * ky_n)

    # M = U * u_G_u_sum
    return U * u_G_u_sum


def u_vector(scattering_type, kx, ky):
    if scattering_type == Scattering.Scalar:
        return np.matrix(np.identity(2))
    elif scattering_type == Scattering.Magnetic:
        return np.matrix(np.identity(2))
    elif scattering_type == Scattering.Scalar_and_Spin_orbit:
        return np.matrix(np.concatenate((np.identity(2), kx*np.identity(2), ky*np.identity(2))))


def v_vector(scattering_type, kx, ky):
    return U_matrix(scattering_type) * u_vector(scattering_type, kx, ky)


def U_matrix(scattering_type, v0=Params.v0, c=Params.c):
    if scattering_type == Scattering.Scalar:
        return np.matrix(np.identity(2)) * v0
    elif scattering_type == Scattering.Magnetic:
        return np.matrix(pauli[2]) * v0
    elif scattering_type == Scattering.Scalar_and_Spin_orbit:
        return np.concatenate((np.concatenate((v0*np.identity(2), np.zeros((2, 2)), np.zeros((2, 2))), axis=1),
                        np.concatenate((np.zeros((2, 2)), np.zeros((2, 2)), 1j * c * v0 * pauli[2]), axis=1),
                        np.concatenate((np.zeros((2, 2)),  -1j * c * v0 * pauli[2], np.zeros((2, 2))), axis=1)
                        ), axis=0)


def T(scattering_type, Gs, kxs, kys, n=51):
    M = construct_m_matrix(scattering_type, Gs, kxs, kys)

    T = np.matrix(np.zeros((M.shape[0], M.shape[1]), dtype=complex))

    for i in range(n):
        T = T + M ** i

    return T


def make_g_0_r_minus_r(G_0_r, G_0_minus_r):
    """
    (second line of eq.A3)
    :param g_0_r: kx_n*ky_n matrix of 2*2 matrices
    :param g_0_minus_r: kx_n*ky_n matrix of 2*2 matrices
    :return: kx_n*ky_n matrix of 2*2 matrices
    """

    kx_n = G_0_r.shape[0]
    ky_n = G_0_r.shape[1]

    G_0_r_minus_r = np.full((kx_n, ky_n), 0, dtype=np.matrix)

    for i in range(kx_n):  # Number of "Pixels" along the kx axis
        for j in range(ky_n):  # Number of "Pixels" along the ky axis
            G_0_r_minus_r[i, j] = G_0_r[i, j] * G_0_minus_r[i, j]

    return G_0_r_minus_r


def flatten_mesh_of_matrices(m):
    '''
    transforms a matrix (mesh) of matrices into matrix of matrices (meshes)
    :param m: matrix of matrices
    :return:
    '''
    kx_n = m.shape[0]
    ky_n = m.shape[1]

    element_shape_x = m[0, 0].shape[0]
    element_shape_y = m[0, 0].shape[1]

    # make empty matrix of meshes
    g = np.matrix(np.zeros((element_shape_x, element_shape_y), dtype=np.matrix))

    # invert mesh of matrices to matrix of meshes
    for i in range(element_shape_x):  # Number of "Pixels" along the kx axis
        for j in range(element_shape_y):  # Number of "Pixels" along the ky axis
            g[i, j] = np.matrix(np.zeros((kx_n, ky_n), dtype=complex))  # new blank mesh
            # populate new blank mesh
            for k in range(kx_n):
                for l in range(ky_n):
                    g[i, j][k, l] = m[k, l][i, j]

    return g


def unflatten_to_mesh_of_matrices(m):
    '''
    "unflattens" matrix of meshes to mesh of matrices (at each mesh point there is a matrix)
    :param m: matrix of meshes
    :return:
    '''

    m = np.matrix(m)

    kx_n = m[0, 0].shape[0]
    ky_n = m[0, 0].shape[1]

    element_shape_x = m.shape[0]
    element_shape_y = m.shape[1]

    gs = np.matrix(np.zeros((kx_n, ky_n), dtype=np.matrix))

    for i in range(kx_n):  # Number of "Pixels" along the kx axis
        for j in range(ky_n):  # Number of "Pixels" along the ky axis
            gs[i, j] = np.matrix(
                np.zeros((element_shape_x, element_shape_y), dtype=complex))  # new blank matrix at (kx,ky)
            #  populate new () blank matrix at (kx, ky)
            for k in range(element_shape_x):
                for l in range(element_shape_y):
                    gs[i, j][k, l] = m[k, l][i, j]

    return gs


def fftransform(x):
    return fft2(x)


def ifftransform(x):
    return ifft2(x)


def fft_of_mesh_of_matrices(xs):
    '''
    g_ij = FFT [x_ij]
    where x and g are matrix elements of the the matrices at each mesh point
    don't know the dimensions of the matrices at each mesh point, but have to assume there are of the same dimensions at
    each mesh point
    :param xs:
    :return:
    '''
    xs_flat = flatten_mesh_of_matrices(xs)

    element_shape_x = xs_flat.shape[0]
    element_shape_y = xs_flat.shape[1]

    xs_flat_fft = np.zeros((element_shape_x, element_shape_y), dtype=np.matrix)

    for i in range(element_shape_x):
        for j in range(element_shape_y):
            xs_flat_fft[i, j] = fftransform(xs_flat[i, j])

    xs_fft = unflatten_to_mesh_of_matrices(xs_flat_fft)
    return xs_fft


def ifft_of_mesh_of_matrices(xs):
    '''
    g_ij = IFFT [x_ij]
    where x and g are matrix elements of the the matrices at each mesh point
    don't know the dimensions of the matrices at each mesh point, but have to assume there are of the same dimensions at
    each mesh point
    :param xs:
    :return:
    '''
    xs_flat = flatten_mesh_of_matrices(xs)

    element_shape_x = xs_flat.shape[0]
    element_shape_y = xs_flat.shape[1]

    xs_flat_fft = np.zeros((element_shape_x, element_shape_y), dtype=np.matrix)

    for i in range(element_shape_x):
        for j in range(element_shape_y):
            xs_flat_fft[i, j] = ifftransform(xs_flat[i, j])

    xs_fft = unflatten_to_mesh_of_matrices(xs_flat_fft)
    return xs_fft


def generate_greens(H_arr_R, rs, weights, kxs, kys, phi_k, omega=Params.omega):
    print("*** generate_Greens")

    kxn = len(kxs)
    kyn = len(kys)

    filepath = FilePaths.greens_cache + 'greens_NbSe2_omega={}_kxn={}_kyn={}_xrange=({},{})_yrange=({},{}).dat'\
        .format(omega, kxn, kyn, min(kxs), max(kxs), min(kys), max(kys))

    if USE_CACHE:
        # check if greens file of requested resoltion exists
        if os.path.exists(filepath):
            print('reading greens from: ', filepath)
            _, _, Gs = get_greens_data(filepath)

            for i in range(kxn):
                for j in range(kyn):
                    Gs[i, j] = Gs[i, j] * phi_k[i, j] * phi_k[i, j].conjugate()

            return Gs

        print('path does not exist, so calculating greens and saving to {}'.format(filepath))

        Gs = np.matrix(np.zeros((kxn, kyn), dtype=np.matrix))

        for i in range(kxn):  # Number of "Pixels" along the kx axis
            for j in range(kyn):  # Number of "Pixels" along the ky axis
                kx = kxs[i]
                ky = kys[j]

                phi_kx_ky = 1

                H_k = calculate_H_k(H_arr_R, rs, weights=weights, kx=kx, ky=ky)

                eigenvectors, eigenvalues = extract_eigenstates_from_hamiltonian(H_k)

                # Construct bare Green's function, G_0(k,\omega)
                Gs[i, j] = G_0(eigenvectors=eigenvectors, eigenvalues=eigenvalues, phi=phi_kx_ky,
                               omega=omega)  # 2*2 matrix

            if i % 20 == 0:
                print('finished column: {}'.format(i))

        write_greens_data(filepath, kxs, kys, Gs)

        for i in range(kxn):
            for j in range(kyn):
                Gs[i, j] = Gs[i, j] * phi_k[i, j] * phi_k[i, j].conjugate()

        return Gs


    Gs = np.matrix(np.zeros((kxn, kyn), dtype=np.matrix))

    for i in range(kxn):  # Number of "Pixels" along the kx axis
        for j in range(kyn):  # Number of "Pixels" along the ky axis
            kx = kxs[i]
            ky = kys[j]

            phi_kx_ky = phi_k[i, j]

            H_k = calculate_H_k(H_arr_R, rs, weights=weights, kx=kx, ky=ky)

            eigenvectors, eigenvalues = extract_eigenstates_from_hamiltonian(H_k)

            # Construct bare Green's function, G_0(k,\omega)
            Gs[i, j] = G_0(eigenvectors=eigenvectors, eigenvalues=eigenvalues, phi=phi_kx_ky, omega=omega)  # 2*2 matrix

        if i % 20 == 0:
            print('finished column: {}'.format(i))

    if USE_CACHE:
        # cache Green's function for (xn, yn)
        write_greens_data(filepath, kxs, kys, Gs)

    return Gs


def calculate_rho(g_0_q, g_0_conj_minus_q, t, t_c):
    qx_n = g_0_q.shape[0]
    qy_n = g_0_q.shape[1]

    rho_q = np.full((qx_n, qy_n), 0, dtype=complex)

    # multiply each matrix element by T
    for i in range(qx_n):  # Number of "Pixels" along the kx axis
        for k in range(qy_n):  # Number of "Pixels" along the ky axis
            for n in range(t.shape[1]):
                for m in range(t.shape[0]):
                    rho_q[i, k] += ((-1 / (2 * np.pi * 1j)) * (
                            (t[m, n] * g_0_q[i, k][n, m]) - (t_c[m, n] * g_0_conj_minus_q[i, k][n, m]))
                                    )

    return rho_q


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


def construct_phi(orbital, xs, ys):
    print('*** construct phi(r)')

    xn = len(xs)
    yn = len(ys)

    phi_r = np.zeros((xn, yn), dtype=complex)

    if orbital == Orbital.gaussian:
        for i in range(xn):
            for j in range(yn):
                phi_r[i, j] = gaussian(xs[i], ys[j], a=Params.alpha_gauss)

        return phi_r

    # get 5s orbital data
    rs, Rs = get_orbital(orbital)

    plt.plot(rs, Rs)
    plt.savefig('orbital_temp.pdf')
    plt.close()

    R_data = interp1d(rs, Rs, kind='linear')

    def R_fit(r):  # x in Angstrom
        # x_bohr = x
        r_bohr = (r * pow(10, -10)) / (5.2917721067 * pow(10, -11))
        if 0 < r_bohr < 100:
            return R_data(r_bohr)
        return 0

    def Y_1_1(x, y, z):
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        return -0.5 * np.sqrt(3 / (2 * np.pi)) * (x + 1j * y) / r

    def Y_1_neg1(x, y, z):
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        return 0.5 * np.sqrt(3 / (2 * np.pi)) * (x - 1j*y) / r

    def Y_1_0(x, y, z):
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        return 0.5 * np.sqrt(3 / np.pi) * (z / r)

    def Theta(x, y, z):
        if orbital == Orbital.orbital_5S:
            return 1
        elif orbital == Orbital.orbital_5P:
            # y_10 = Y_1_0(x, y, z)
            # return y_10
            y_11 = Y_1_1(x, y, z)
            y_1_neg1 = Y_1_neg1(x, y, z)
            # return (1 / np.sqrt(2)) * 1j * (y_1_neg1 - y_11)
            # return (1 / np.sqrt(2)) * (y_11 + y_1_neg1 + 1j * (y_1_neg1 - y_11))
            return (1 / np.sqrt(2)) * (abs(y_11 + y_1_neg1) + abs(y_1_neg1 - y_11))
            # return (1 / np.sqrt(2)) * (y_10 + y_11 + y_1_neg1 + 1j * (y_1_neg1 - y_11))

    for i in range(xn):
        for j in range(yn):
            x = xs[i]
            y = ys[j]

            r2 = x**2 + y**2 + z**2
            r = np.sqrt(r2)

            phi_r[i, j] = R_fit(r) * Theta(x, y, z)

    return phi_r


def make_g_0_q(g_0_r_minus_r, xs, ys, qxs, qys):  # fft
    print('*** make_g_0_q')

    print(xs)
    print(ys)
    print(qxs)
    print(qys)

    xn = len(xs)
    yn = len(ys)

    qxn = len(qxs)
    qyn = len(qys)

    q_0_q = np.zeros((qxn, qyn), dtype=np.matrix)

    for i in range(qxn):
        for j in range(qyn):
            # For each q element, iterate over all of known r space
            for m in range(xn):
                for n in range(yn):
                    k = np.array([qxs[i], qys[j]])
                    r = np.array([xs[m], ys[n]])

                    phase = -1j * np.dot(k, r)

                    q_0_q[i, j] += g_0_r_minus_r[m, n] * np.exp(phase)

        if i % 20 == 0:
            print('finished fft of row:', i)

    return q_0_q


def gaussian(x, y, a):
    r2 = x**2 + y**2
    return np.exp(-r2/a)
    # return (1 / (a * np.sqrt(2 * np.pi))) * np.exp(-r2 / (2 * a**2))


def qpi(scattering_type, orbital):
    if scattering_type == Scattering.Scalar:
        print('scalar QPI')
    elif scattering_type == Scattering.Magnetic:
        print('magnetic QPI')
    elif scattering_type == Scattering.Scalar_and_Spin_orbit:
        print('scalar and spin orbit QPI')

    start_time = time.time()

    # Get Hamiltonian data
    H_arr_R, rs, weights = get_hamiltonian('../data/Hamiltonian.dat')

    # Generate k-space axes
    kx_n = 600
    ky_n = 600

    kx_length = 8 * np.pi / Params.a_0  # \AA^-1
    ky_length = 8 * np.pi / Params.a_0  # \AA^-1  # (also done 8* for 600)

    kxs, kys = generate_axes(kx_n, ky_n, x_length=kx_length, y_length=ky_length)

    qxs, qys = generate_axes(kx_n, ky_n, x_length=kx_length, y_length=ky_length)  # would expect 2*

    print('kx_length = ', kx_length)
    print('ky_length = ', ky_length)

    phi_k = np.zeros((kx_n, ky_n), dtype=float)
    for i in range(kx_n):
        for j in range(ky_n):
            # phi_k[i, j] = gaussian(kxs[i], kys[j], a=Params.alpha_gauss)
            # phi_k[i, j] = 1

            # kx = kxs[i]
            # ky = kys[j]
            # k = np.sqrt(kx**2 + ky**2)
            # # phi_k[i, j] = np.exp(-((k - 1.2) ** 2) / Params.alpha_gauss)
            # #
            # if Params.lower_k < k < Params.mid_k:
            #     phi_k[i, j] = Params.inner_v
            # elif Params.mid_k < k < Params.upper_k:
            #     phi_k[i, j] = Params.outer_v
            # else:
            #     phi_k[i, j] = 0
            ########################################################
            kx = kxs[i]
            ky = kys[j]
            k = np.sqrt(kx ** 2 + ky ** 2)

            if 0 < k <= Params.Gam_upper:
                phi_k[i, j] = Params.Gam_val
            ########################################################
            # elif mid_k < k <= upper_k:
            #     phi_k[i, j] = outer_v
            # else:
            #     phi_k[i, j] = 0

            ########################################################
            ########################################################
            ########################################################
            kx = kxs[i] - Params.K[0]
            ky = kys[j] - Params.K[1]
            k = np.sqrt(kx ** 2 + ky ** 2)

            if 0 < k < Params.k_upper:
                phi_k[i, j] = Params.k_val

            ##
            kx = kxs[i] + Params.K[0]
            ky = kys[j] + Params.K[1]
            k = np.sqrt(kx ** 2 + ky ** 2)

            if 0 < k < Params.k_upper:
                phi_k[i, j] = Params.k_val

            kx = kxs[i] + Params.K[0]
            ky = kys[j] - Params.K[1]
            k = np.sqrt(kx ** 2 + ky ** 2)

            if 0 < k < Params.k_upper:
                phi_k[i, j] = Params.k_val

            kx = kxs[i] - Params.K[0]
            ky = kys[j] + Params.K[1]
            k = np.sqrt(kx ** 2 + ky ** 2)

            if 0 < k < Params.k_upper:
                phi_k[i, j] = Params.k_val

            kx = kxs[i] - 0
            ky = kys[j] + np.sqrt(Params.K[0] ** 2 + Params.K[1] ** 2)
            k = np.sqrt(kx ** 2 + ky ** 2)

            if 0 < k < Params.k_upper:
                phi_k[i, j] = Params.k_val

            kx = kxs[i] - 0
            ky = kys[j] - np.sqrt(Params.K[0] ** 2 + Params.K[1] ** 2)
            k = np.sqrt(kx ** 2 + ky ** 2)

            if 0 < k < Params.k_upper:
                phi_k[i, j] = Params.k_val
            ########################################################
            ########################################################
            ########################################################

            #
            # if phi_k[i, j]**2 > 0.025:
            #     phi_k[i, j] = 1
            # else:
            #     phi_k[i, j] = 0

    plot_heatmap(kxs, kys, abs(phi_k.transpose()), filename="../plots/debug/phi_k_kxn={}_col={}_a={}_k_val={}__{}.pdf".format(kx_n, orbital.value, Params.alpha_gauss, Params.k_val, VERSION_NUMBER), show_bz=True)

    print('*** Greens')

    G_0_k = generate_greens(H_arr_R, rs, weights, kxs, kys, phi_k=phi_k)

    G_0_k_trace = np.zeros((kx_n, ky_n), dtype=complex)
    for i in range(kx_n):
        for j in range(ky_n):
            G_0_k_trace[i, j] = G_0_k[i, j].trace()

    plot_heatmap(kxs, kys, abs(G_0_k_trace.transpose()), filename='../plots/debug/G_trace_kxn={}_a={}_k_val={}__{}.pdf'.format(kx_n, Params.alpha_gauss, Params.k_val, VERSION_NUMBER), show_bz=True)

    G_0_primed_k = np.matrix(np.zeros((kx_n, ky_n), dtype=np.matrix))
    G_0_primed_k_primed = np.matrix(np.zeros((kx_n, ky_n), dtype=np.matrix))

    # Apply u and v vectors/matrices to G
    for i in range(kx_n):
        for j in range(ky_n):
            G_0_primed_k[i, j] = G_0_k[i, j] * (u_vector(scattering_type, kxs[i], kys[j]).transpose())
            G_0_primed_k_primed[i, j] = v_vector(scattering_type, kxs[i], kys[j]) * G_0_k[i, j]

    print('*** FFTs')

    g_0_r = fft_of_mesh_of_matrices(G_0_primed_k_primed)
    g_0_minus_r = ifft_of_mesh_of_matrices(G_0_primed_k)

    g_0_r_c = ifft_of_mesh_of_matrices(G_0_primed_k_primed.conjugate())
    g_0_minus_r_c = fft_of_mesh_of_matrices(G_0_primed_k.conjugate())

    # construct g_0(r,-r,w)
    g_0_r_minus_r = make_g_0_r_minus_r(g_0_r, g_0_minus_r)  # matrix of matrices
    g_0_r_minus_r_c = make_g_0_r_minus_r(g_0_r_c, g_0_minus_r_c)  # matrix of matrices

    ''' Take FFT[g(r,-r,w)] and IFFT[g*(r,-r,w)] '''
    # FFT[g(r,-r,w)] = G_0(k,k-q)
    g_0_r_minus_r_fft = fftshift(fft_of_mesh_of_matrices(g_0_r_minus_r)) / (kx_n*ky_n)
    # FFT[g*(r,-r,w)] = G*_0(k,k+q)
    g_0_r_minus_r_fft_c = fftshift(fft_of_mesh_of_matrices(g_0_r_minus_r_c)) / (kx_n*ky_n)

    g_0_q = g_0_r_minus_r_fft
    g_0_conj_minus_q = g_0_r_minus_r_fft_c

    # g_0_q_manual = make_g_0_q(g_0_r_minus_r, xs, ys, qxs, qys)
    # g_0_conj_minus_q_manual = make_g_0_q(g_0_r_minus_r_c, xs, ys, qxs, qys)

    print('*** T')

    t = T(scattering_type, G_0_k, kxs, kys, n=101)
    t_congugate = t.conjugate()

    print('*** abs(rho(q))')

    rho_q = calculate_rho(g_0_q=g_0_q, g_0_conj_minus_q=g_0_conj_minus_q, t=t, t_c=t_congugate)
    # rho_q_manual = calculate_rho(g_0_q=g_0_q_manual, g_0_conj_minus_q=g_0_conj_minus_q_manual, t=t, t_c=t_congugate)

    rho_q_abs = abs(rho_q)

    # Scale fixing
    # for i in range(kx_n):
    #     for j in range(ky_n):
    #         # if rho_q_abs[i, j] >= 0.0000020:
    #         #     rho_q_abs[i, j] = 0.0000005
    #         kx = kxs[i]
    #         ky = kys[j]
    #         k = np.sqrt(kx**2 + ky**2)
    #         if k < 2:
    #             rho_q_abs[i, j] = 0

    print('*** plotting')

    filename = ''
    if scattering_type == Scattering.Scalar:
        filename = '../plots/scalar/scalar_omega={}eV_kxn={}_col={}_a={}__{}.pdf'
    elif scattering_type == Scattering.Magnetic:
        filename = '../plots/magnetic/magnetic_omega={}eV_kxn={}_col={}_a={}__{}.pdf'
    elif scattering_type == Scattering.Scalar_and_Spin_orbit:
        filename = '../plots/spin-orbit_scalar/spin-orbit_scalar_omega={}eV_c=' + str(Params.c) + '_n=101_kxn={}_col={}_a={}__k_val={}_{}.pdf'

    plot_heatmap(qxs, qys, rho_q_abs.transpose(), filename=filename.format(Params.omega, kx_n, orbital.value, Params.alpha_gauss, Params.k_val, VERSION_NUMBER), show_bz=True)
    # plot_heatmap(qxs, qys, rho_q_abs_manual, filename=filename.format(z, kx_n, orbital.value, VERSION_NUMBER) + '_manual.pdf')

    # write_data('../data/scalar_conjugate_before_fix_{}.dat'.format(kx_n), xs, ys, rho_q)

    time_end = time.time()
    dt = time_end - start_time
    average_dt = dt / ((kx_n) * (ky_n))
    print('time taken: {:.6f} seconds'.format(dt))
    print('average time per q vector: {:.6f} seconds'.format(average_dt))


def plot_heatmap(xs, ys, zs, filename, show_bz=False):
    fig, ax = plt.subplots()

    c = ax.pcolormesh(xs, ys, zs, cmap='hot')
    ax.set_xlabel('$q_x$ $(\AA^{-1})$')
    ax.set_ylabel('$q_y$ $(\AA^{-1})$')

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

        xs_brillouin = np.array([])
        ys_brillouin = np.array([])

        for coord in cs:
            xs_brillouin = np.append(xs_brillouin, coord[0]*2)
            ys_brillouin = np.append(ys_brillouin, coord[1]*2)

        ax.plot(xs_brillouin, ys_brillouin, color='gray', linestyle='dashed')  # , zorder=1

    fig.colorbar(c, ax=ax, label='')  # TODO: change/try a logarithmic color scale/bar .
    fig.gca().set_aspect('equal', adjustable='box')
    fig.tight_layout()
    fig.savefig(filename, rasterized=True)
    # fig.show()
    plt.close(fig)
    print('heat map plot: \n {} \n'.format(filename))


def plt_from_file(filename):
    xs, ys, zs = get_heatmap_data(filename)
    plot_heatmap(xs, ys, zs.transpose(), filename + '.pdf')


def pad_matrix(m):
    print('*** pad matrix')

    xn = m.shape[0]
    yn = m.shape[1]

    element_shape = m[0, 0].shape

    if (xn % 2 != 0) or (yn % 2 != 0):
        import sys
        import os
        print('ERROR: only matrices with even dimensions can be padded - got dimensions ({}, {})'.format(xn, yn))
        sys.exit(os.EX_DATAERR)

    xn_pad = 2 * xn
    yn_pad = 2 * yn

    m_pad = np.zeros((xn_pad, yn_pad), dtype=np.matrix)

    for i in range(xn_pad):
        for j in range(yn_pad):
            if i < xn/2 or i >= 3*xn/2 or j < yn/2 or j >= 3*yn/2:
                m_pad[i, j] = np.zeros(element_shape, dtype=complex)
                continue

            m_pad[i, j] = m[i-int(xn/2), j-int(yn/2)]

    return m_pad


def main():
    orbital = Orbital.gaussian
    # print('NbSe2 - extended/ - {} \AA - col={}, alpha = {}, c = {}, omega = {} (E_f = {}, delat_E_f = {})'
    #       .format(VERSION_NUMBER, orbital.value, Params.alpha_gauss, Params.c, Params.omega, Params.E_f,
    #               Params.delta_E_f))
    # qpi(Scattering.Scalar_and_Spin_orbit, orbital)


    for i in range(10):
        Params.k_val = 0 + i * 0.1
        print('NbSe2 - extended/ - {} \AA - col={}, alpha = {}, c = {}, omega = {} (E_f = {}, delat_E_f = {})'
              .format(VERSION_NUMBER, orbital.value, Params.alpha_gauss, Params.c, Params.omega, Params.E_f,
                      Params.delta_E_f))
        qpi(Scattering.Scalar, orbital)


USE_CACHE = True
# VERSION_NUMBER = '1.20.5_Scalar+SO_c={}_dEf={}_{}-k-{}={}_{}-k-{}={}__'.format(Params.c, Params.delta_E_f, Params.lower_k, Params.mid_k, Params.inner_v, Params.mid_k, Params.upper_k, Params.outer_v)
# VERSION_NUMBER = '1.20.5_Scalar+SO_c={}_dEf={}_{}-K_k-{}={}'.format(Params.c, Params.delta_E_f, 0, Params.k_upper, Params.k_val)
VERSION_NUMBER = '1.20.5_Scalar+SO_c={}_dEf={}_{}-K_k-{}={}__{}-G_k-{}={}'.format(Params.c, Params.delta_E_f, 0, Params.k_upper, Params.k_val, 0, Params.Gam_upper, Params.Gam_val)
# VERSION_NUMBER = '1.20.5_Scalar+SO_c={}_dEf={}__{}-G_k-{}={}'.format(Params.c, Params.delta_E_f, 0, Params.Gam_upper, Params.Gam_val)

if __name__ == '__main__':
    main()

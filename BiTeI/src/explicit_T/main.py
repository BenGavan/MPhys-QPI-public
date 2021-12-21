import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import linalg
from scipy.fftpack import ifft2
from scipy.fftpack import fft2
from scipy.fftpack import fft
from scipy.fftpack import fftshift
from scipy.fftpack import ifftshift
import os
import enum
import time

# from tqdm import tqdm

'''
Notes:
 - All matrices and vectors should conform to the standard convension where <psi| = (...)
   - when introducing a new matrix, if not automatically in the format, change as soon as defined.
'''


class Scattering(enum.Enum):
    Scalar = 0
    Magnetic = 1


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
def write_data(filename, xs, ys, zs):
    print('writing data to: ', filename)
    append_arr_to_file(filename, xs)
    append_arr_to_file(filename, ys)
    append_matrix_to_file(filename, zs)


def get_data(filename):
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
    #
    # print('-------')
    # # print('kxs = ', kxs)
    # # print('kys = ', kys)
    # print('-------')
    # # print('gs_flat = ', gs_flat)
    # print('-------')
    # print('Gs = ', unflatten_to_mesh_of_matrices(gs_flat))
    # print('-------')

    return kxs, kys, unflatten_to_mesh_of_matrices(gs_flat)


# TODO: Decouple data generation and plotting
# TODO: https://eigen.tuxfamily.org/index.php?title=Main_Page

m = 0.0168  # in ev A^(-2)
alpha_4 = -2.03  # A^(-2)
alpha_6 = 87.5  # A^(-4)
v = 3.13  # eV A^(-1)
beta_3 = -2.01  # A^(-2)
beta_5 = 323  # A^(-4)
lamb = -41.7  # eV A(-3)
gamma_5 = 2.43  # A^(-2)
E_0 = -0.352  # eV

pauli = np.array((
    ((0, 1), (1, 0)),
    ((0, -1j), (1j, 0)),
    ((1, 0), (0, -1))
))


def E(k):
    return 1 + alpha_4 * k ** 2 + alpha_6 * k ** 4


def V(k):
    return v * (1 + beta_3 * k ** 2 + beta_5 * k ** 4)


def Lambda(k):
    return lamb * (1 + gamma_5 * k ** 2)


def Hamiltonian(kx, ky):
    k2 = kx ** 2 + ky ** 2
    k = np.sqrt(k2)
    return (E_0 + (k2 / (2 * m)) * E(k)) * np.identity(2) + V(k) * (kx * pauli[1] - ky * pauli[0]) + Lambda(k) * (
            3 * kx ** 2 - ky ** 2) * ky * pauli[2]


'''
Calculates the bare Green's function G_0(k, w) = \sigma_n \frac{}{omega + i\eta - \epsilon}
 - psi = eigenstate (2*2 matrix)
'''


def g_n(psi_n, omega, eta, epsilon_n):
    psi_n_H = psi_n.getH()

    # numerator |\psi><\psi|
    outer = psi_n * psi_n_H

    # g_n(k,\omega)^-1 =  \omega + i\eta - \epsilon
    g = omega + 1j * eta - epsilon_n

    return outer / g


def G_0(eigenvectors, eigenvalues, omega=-0.01, eta=0.01):
    G = np.matrix([[0, 0], [0, 0]])
    for i in range(len(eigenvalues)):
        G = G + g_n(eigenvectors[i], omega=omega, eta=eta, epsilon_n=eigenvalues[i])
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


def T_explicit(scattering_type, G_0, v_0=0.1):
    V = None

    if scattering_type == Scattering.Scalar:
        V = np.matrix(np.identity(2)) * v_0
    elif scattering_type == Scattering.Magnetic:
        V = np.matrix(pauli[2]) * v_0

    kx_n = G_0.shape[0]
    ky_n = G_0.shape[1]

    G_sum = np.full((2, 2), 0, dtype=complex)

    for i in range(kx_n):  # Number of "Pixels" along the kx axis
        for j in range(ky_n):  # Number of "Pixels" along the ky axis
            G_sum = G_sum + G_0[i, j]

    X = np.full((2, 2), 0, dtype=complex)

    G_sum = G_sum / (kx_n * ky_n)

    print('-------')
    print('G_sum = ', G_sum)
    print('-------')

    X = np.identity(2) - V * G_sum
    X = np.matrix(X)

    print('-------')
    print('X = ', X)
    print('-------')

    return linalg.inv(X) * V
    # return V / X  # TODO: Test for using inv() (as we would expect)

# def T(G_0, n, v_0=0.1):
#     V = np.identity(2) * v_0
#     if n == 1:
#         return V
#     elif n >= 2:
#         kx_n = len(G_0)
#         ky_n = len(G_0[0])
#
#         G_sum = np.full((2, 2), 0, dtype=complex)
#
#         for i in range(kx_n):  # Number of "Pixels" along the kx axis
#             for j in range(ky_n):  # Number of "Pixels" along the ky axis
#                 G_sum = G_sum + G_0[i, j]
#
#         return V + V * G_sum * T(G_0, n-1)


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
    # kx_n = x.shape[0]
    # ky_n = x.shape[1]
    # return fftshift(fft2(x)) / (kx_n * ky_n)
    # return fftshift(fft2(x)) / np.sqrt(kx_n * ky_n)
    return fft2(x)


def ifftransform(x):
    # kx_n = x.shape[0]
    # ky_n = x.shape[1]
    # return ifftshift(ifft2(x)) / np.sqrt(kx_n * ky_n)
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
    # TODO: implement FFT of matrix (mesh) with matrix elements
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
    # TODO: implement FFT of matrix (mesh) with matrix elements
    xs_flat = flatten_mesh_of_matrices(xs)

    element_shape_x = xs_flat.shape[0]
    element_shape_y = xs_flat.shape[1]

    xs_flat_fft = np.zeros((element_shape_x, element_shape_y), dtype=np.matrix)

    for i in range(element_shape_x):
        for j in range(element_shape_y):
            xs_flat_fft[i, j] = ifftransform(xs_flat[i, j])

    xs_fft = unflatten_to_mesh_of_matrices(xs_flat_fft)
    return xs_fft


def generate_greens(xs, ys, omega=-0.01):
    print("*** generate_Greens")

    x_n = len(xs)
    y_n = len(ys)

    filepath = '/Users/ben/Desktop/MPhys-QPI/project/QPI/python/data/greens__omega={}_xn={}_yn={}_xrange=({},{})_yrange=({},{}).dat' \
        .format(omega, x_n, y_n, min(xs), max(xs), min(ys), max(ys))

    # check if greens file of requested resoltion exists
    if os.path.exists(filepath):
        print('reading greens from: ', filepath)
        _, _, Gs = get_greens_data(filepath)
        return Gs

    print('{} does not exist, so calculating greens'.format(filepath))

    Gs = np.matrix(np.zeros((x_n, y_n), dtype=np.matrix))

    for i in range(x_n):  # Number of "Pixels" along the kx axis
        for j in range(y_n):  # Number of "Pixels" along the ky axis
            kx = xs[i]
            ky = ys[j]

            H = Hamiltonian(kx, ky)

            eigenvectors, eigenvalues = extract_eigenstates_from_hamiltonian(H)

            # Construct bare Green's function, G_0(k,\omega)
            Gs[i, j] = G_0(eigenvectors=eigenvectors, eigenvalues=eigenvalues, omega=omega)  # 2*2 matrix

        if i % 20 == 0:
            print('finished column: {}'.format(i))

    # cache Green's function for (xn, yn)
    write_greens_data(filepath, xs, ys, Gs)

    print('finished Greens')
    return Gs


def generate_axes(kx_n, ky_n, kx_length, ky_length):
    '''

    :param kx_n: # pixes along x axis
    :param ky_n: # pixes along y axis
    :param kx_length:
    :param ky_length:
    :return:
    '''
    xs = np.linspace(-kx_length / 2, kx_length / 2, kx_n)
    ys = np.linspace(-ky_length / 2, ky_length / 2, ky_n)

    return xs, ys


def qpi_explicit_T(scattering_type):
    start_time = time.time()

    kx_n = 300
    ky_n = 300

    kxs, kys = generate_axes(kx_n, ky_n, kx_length=1, ky_length=1)

    G_0_k = generate_greens(kxs, kys)

    print('*** FFTs')

    g_0_r = fft_of_mesh_of_matrices(G_0_k)
    g_0_minus_r = ifft_of_mesh_of_matrices(G_0_k)

    g_0_r_c = ifft_of_mesh_of_matrices(G_0_k.conjugate())
    g_0_minus_r_c = fft_of_mesh_of_matrices(G_0_k.conjugate())

    # construct g_0(r,-r,w)
    g_0_r_minus_r = make_g_0_r_minus_r(g_0_r, g_0_minus_r)  # matrix of (2*2) matrices
    g_0_r_minus_r_c = make_g_0_r_minus_r(g_0_r_c, g_0_minus_r_c)  # matrix of (2*2) matrices

    ''' Take FFT[g(r,-r,w)] and IFFT[g*(r,-r,w)] '''
    # FFT[g(r,-r,w)] = G_0(k,k-q)
    g_0_r_minus_r_fft = fftshift(fft_of_mesh_of_matrices(g_0_r_minus_r)) / (kx_n * ky_n)
    # FFT[g*(r,-r,w)] = G*_0(k,k+q)
    g_0_r_minus_r_fft_c = fftshift(fft_of_mesh_of_matrices(g_0_r_minus_r_c)) / (kx_n * ky_n)

    g_0_q = g_0_r_minus_r_fft
    g_0_conj_minus_q = g_0_r_minus_r_fft_c

    print('*** T')

    t = T_explicit(scattering_type, G_0_k)
    t_congugate = t.conjugate()

    rho_q = np.full((kx_n, ky_n), 0, dtype=complex)

    # multiply each matrix element by T
    for i in range(kx_n):  # Number of "Pixels" along the kx axis
        for k in range(ky_n):  # Number of "Pixels" along the ky axis
            for n in range(t.shape[1]):
                for m in range(t.shape[0]):
                    rho_q[i, k] += ((-1 / (2 * np.pi * 1j)) * (
                            (t[m, n] * g_0_q[i, k][n, m]) - (t_congugate[m, n] * g_0_conj_minus_q[i, k][n, m]))
                                    )

    rho_q_abs = np.full((kx_n, ky_n), 0, dtype=float)

    # take the modulus (magnitude) of \rho(q)
    for i in range(kx_n):  # Number of "Pixels" along the kx axis
        for k in range(ky_n):  # Number of "Pixels" along the ky axis
            rho_q_abs[i, k] = abs(rho_q[i, k])

    filepath = ''
    if scattering_type == Scattering.Scalar:
        filepath = '../../plots/explicit_T/scalar/scalar_new_derivation_{}.pdf'.format(kx_n)
    if scattering_type == Scattering.Magnetic:
        filepath = '../../plots/explicit_T/magnetic/magnetic_new_derivation_{}.pdf'.format(kx_n)

    plot_heatmap(xs, ys, rho_q_abs.transpose(), filename=filepath)

    # write_data('../data/scalar_conjugate_before_fix_{}.dat'.format(kx_n), xs, ys, rho_q)

    time_end = time.time()
    dt = time_end - start_time
    average_dt = dt / ((kx_n) * (ky_n))
    print('time taken: {:.6f} seconds'.format(dt))
    print('average time per q vector: {:.6f} seconds'.format(average_dt))

    print('rho_q = ', rho_q)


def plot_heatmap(xs, ys, zs, filename=None):
    fig, ax = plt.subplots()

    # c = ax.pcolormesh(xs, ys, rho_q, cmap='RdBu', vmin=z_min, vmax=z_max)
    # c = ax.pcolormesh(xs, ys, rho_q, cmap='RdBu', edgecolors=None)
    c = ax.pcolormesh(xs, ys, zs, cmap='hot')
    # ax.set_title('title')
    ax.set_xlabel('$q_x$ $(\AA^{-1})$')
    ax.set_ylabel('$q_y$ $(\AA^{-1})$')

    # ax.axis([x.min(), x.max(), y.min(), y.max()])
    fig.colorbar(c, ax=ax, label='')
    fig.gca().set_aspect('equal', adjustable='box')
    fig.tight_layout()
    if filename is None:
        filename = '../plots/no_name_given_hot_{}.pdf'.format(-1)
    fig.savefig(filename)
    fig.show()
    print('heat map plot: ', filename)


def plt_from_file(filename):
    xs, ys, zs = get_data(filename)
    plot_heatmap(xs, ys, zs.transpose())


def main():
    qpi_explicit_T(Scattering.Magnetic)
    return


if __name__ == '__main__':
    main()

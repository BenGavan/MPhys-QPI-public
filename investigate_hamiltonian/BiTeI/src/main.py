import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from sympy.matrices import Matrix
import time
import enum
# from tqdm import tqdm

'''
Notes:
 - All matrices and vectors should conform to the standard convension where <psi| = (...)
   - when introducing a new matrix, if not automatically in the format, change as soon as defined.
'''

# TODO: Decouple data generation and plotting
# TODO: https://eigen.tuxfamily.org/index.php?title=Main_Page


m = 0.0168  #in ev A^(-2)
alpha_4 = -2.03  # A^(-2)
alpha_6 = 87.5  # A^(-4)
v = 3.13  #eV A^(-1)
beta_3 = -2.01  #A^(-2)
beta_5 = 323  #A^(-4)
lamb = -41.7  # eV A(-3)
gamma_5 = 2.43  #A^(-2)
E_0 = -0.352  #eV

pauli = np.array((
    ((0, 1), (1, 0)),
    ((0, -1j), (1j, 0)),
    ((1, 0), (0, -1))
))


def E(k):
    return 1 + alpha_4*k**2 + alpha_6*k**4


def V(k):
    return v * (1 + beta_3*k**2 + beta_5*k**4)


def Lambda(k):
    return lamb * (1 + gamma_5*k**2)


def Hamiltonian(k_x, k_y):
    k2 = k_x**2 + k_y**2
    k = np.sqrt(k2)
    return (E_0 + (k2/(2*m))*E(k)) * np.identity(2) + V(k)*(k_x*pauli[1] - k_y*pauli[0]) + Lambda(k) * (3*k_x**2 - k_y**2)*k_y*pauli[2]


'''
Calculates the bare Green's function G_0(k, w) = \sigma_n \frac{}{omega + i\eta - \epsilon}
 - psi = eigenstate (2*2 matrix)
'''
def g_n(psi_n, omega, eta, epsilon_n):
    psi_n_H = psi_n.getH()

    # numerator |\psi><\psi|
    outer = psi_n * psi_n_H

    # g_n(k,\omega)^-1 =  \omega + i\eta - \epsilon
    g = omega + 1j*eta - epsilon_n

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


def generate_energy_contour_data():

    k_x_length = 0.6
    k_y_length = 0.6

    k_x_n = 20
    k_y_n = 20

    eigen_vals_lower = np.zeros((k_x_n + 1, k_y_n + 1))
    eigen_vals_upper = np.zeros((k_x_n + 1, k_y_n + 1))

    eigen_vals_lower_sci = np.zeros((k_x_n + 1, k_y_n + 1))
    eigen_vals_upper_sci = np.zeros((k_x_n + 1, k_y_n + 1))

    xs = np.array([])
    ys = np.array([])

    for i in range(k_x_n + 1):
        xs = np.append(xs, -(k_x_length / 2) + (k_x_length / k_x_n) * i)
    for i in range(k_y_n + 1):
        ys = np.append(ys, -(k_y_length / 2) + (k_y_length / k_y_n) * i)

    print(xs)
    print(ys)

    for i in range(k_x_n + 1):  # Number of "Pixels" along the k_x axis
        k_x = -(k_x_length / 2) + (k_x_length / k_x_n) * i
        for j in range(k_y_n + 1):  # Number of "Pixels" along the k_y axis
            k_y = -(k_y_length / 2) + (k_y_length / k_y_n) * j

            H = Hamiltonian(k_x, k_y)

            vs = []

            for r in H:
                for x in r:
                    vs.append(x)

            m = Matrix(2, 2, vs)
            m_diags = m.diagonalize()
            m_diag = m_diags[1]

            eigens = [m_diag[0], m_diag[3]]

            # es = linalg.eigvals(H) # No longer used (might change back -  much more performant)

            eigen_vals_lower[i][j] = min(eigens)
            eigen_vals_upper[i][j] = max(eigens)

            print(eigen_vals_upper)

    write_data('upper.dat', xs, ys, eigen_vals_upper)
    write_data('lower.dat', xs, ys, eigen_vals_lower)


def generate_energy_contour_data_sci():
    k_x_length = 0.64
    k_y_length = 0.64

    k_x_n = 200
    k_y_n = 200

    eigen_vals_lower = np.zeros((k_x_n + 1, k_y_n + 1))
    eigen_vals_upper = np.zeros((k_x_n + 1, k_y_n + 1))

    xs = np.array([])
    ys = np.array([])

    for i in range(k_x_n + 1):
        xs = np.append(xs, -(k_x_length / 2) + (k_x_length / k_x_n) * i)

    for i in range(k_y_n + 1):
        ys = np.append(ys, -(k_y_length / 2) + (k_y_length / k_y_n) * i)

    print(xs)
    print(ys)

    start_time = time.time()

    for i in range(k_x_n + 1):  # Number of "Pixels" along the k_x axis
        k_x = -(k_x_length / 2) + (k_x_length / k_x_n) * i
        for j in range(k_y_n + 1):  # Number of "Pixels" along the k_y axis
            k_y = -(k_y_length / 2) + (k_y_length / k_y_n) * j

            H = Hamiltonian(k_x, k_y)

            # vs = []
            #
            # for r in H:
            #     for x in r:
            #         vs.append(x)
            #
            # m = Matrix(2, 2, vs)
            # m_diags = m.diagonalize()
            # m_diag = m_diags[1]

            # eigens = [float(m_diag[0]), float(m_diag[3])]

            es = linalg.eigvals(H)  # No longer used (might change back -  much more performant)

            if i % 1000 == 0:
                mid_time = time.time()
                dt = mid_time - start_time
                print('{:.4f} seconds'.format(dt / pow(10, 0)))
                # print(es)

            eigens = [float(es[0]), float(es[1])]
            eigen_vals_lower[j][i] = min(eigens)
            eigen_vals_upper[j][i] = max(eigens)

            # print(eigen_vals_upper)

    end_time = time.time()

    dt = end_time - start_time
    print('Time taken: {:.3f} seconds'.format(dt))

    write_data('upper_sci.dat', xs, ys, eigen_vals_upper)
    write_data('lower_sci.dat', xs, ys, eigen_vals_lower)


def generate_band_structure_data_perp():
    # M - G - K
    # (k_x, 0) -> (0,0) -> (0, k_y)

    no_sample = 100
    dk = 0.3/no_sample

    # --------- G-M --------- #
    # K_y = 0 (G-M is along x axis)
    k_x = 0
    k_y = 0

    k_xs = []

    G_M_zs_lower = np.array([])
    G_M_zs_upper = np.array([])

    for i in range(no_sample):
        H = Hamiltonian(k_x, k_y)

        es = linalg.eigvals(H)  # No longer used (might change back -  much more performant)

        eigens = [float(es[0]), float(es[1])]
        G_M_zs_lower = np.append(G_M_zs_lower, min(eigens))
        G_M_zs_upper = np.append(G_M_zs_upper, max(eigens))

        k_xs.append(-k_x)
        k_x += dk

    plt.plot(k_xs, G_M_zs_lower, color='dodgerblue')
    plt.plot(k_xs, G_M_zs_upper, color='darkorange')


    # --------- G-K --------- #
    # K_x = 0 (G-K is along y axis)
    k_x = 0
    k_y = 0

    k_ys = []

    G_K_zs_lower = np.array([])
    G_K_zs_upper = np.array([])

    no_sample *= 1.5
    no_sample = int(no_sample)

    for i in range(no_sample):
        H = Hamiltonian(k_x, k_y)

        es = linalg.eigvals(H)  # No longer used (might change back -  much more performant)

        eigens = [float(es[0]), float(es[1])]
        G_K_zs_lower = np.append(G_K_zs_lower, min(eigens))
        G_K_zs_upper = np.append(G_K_zs_upper, max(eigens))

        k_ys.append(k_y)
        k_y += dk


    plt.plot(k_ys, G_K_zs_lower, color='dodgerblue')
    plt.plot(k_ys, G_K_zs_upper, color='darkorange')

    plt.ylabel('E (eV)')

    x = [-0.4, -0.3, 0, 0.3, 0.4]
    labels = ['$\overline{M}$', '(0.3, 0)', '$\Gamma$', '(0, 0.3)', '$\overline{K}$']
    plt.xticks(x, labels, rotation='horizontal')

    plt.ylim(bottom=-0.5, top=max([max(G_M_zs_lower)]))

    plt.savefig('band_M-G-K_perp.pdf')
    plt.show()


def generate_band_structure_data_45():
    # M - G - K
    # (k_x, 0) -> (0,0) -> (0, k_y)

    no_sample = 100
    dk = 0.3/no_sample

    # --------- G-M --------- #
    # K_y = 0 (G-M is along x axis)
    k_x = 0
    k_y = 0

    k_xs = []

    G_M_zs_lower = np.array([])
    G_M_zs_upper = np.array([])

    for i in range(no_sample):
        H = Hamiltonian(k_x, k_y)

        es = linalg.eigvals(H)  # No longer used (might change back -  much more performant)

        eigens = [float(es[0]), float(es[1])]
        G_M_zs_lower = np.append(G_M_zs_lower, min(eigens))
        G_M_zs_upper = np.append(G_M_zs_upper, max(eigens))

        k_xs.append(-k_x)
        k_x += dk

    plt.plot(k_xs, G_M_zs_lower, color='dodgerblue')
    plt.plot(k_xs, G_M_zs_upper, color='darkorange')


    # --------- G-K --------- #
    # K_x = 0 (G-K is along y axis)
    k_x = 0
    k_y = 0

    k_s = []

    G_K_zs_lower = np.array([])
    G_K_zs_upper = np.array([])

    no_sample *= 1.5
    no_sample = int(no_sample)

    for i in range(no_sample):
        H = Hamiltonian(k_x, k_y)

        es = linalg.eigvals(H)  # No longer used (might change back -  much more performant)

        eigens = [float(es[0]), float(es[1])]
        G_K_zs_lower = np.append(G_K_zs_lower, min(eigens))
        G_K_zs_upper = np.append(G_K_zs_upper, max(eigens))

        k_s.append(np.sqrt(k_x**2 + k_y**2))
        k_y += dk
        k_x += dk


    plt.plot(k_s, G_K_zs_lower, color='dodgerblue', label='lower')
    plt.plot(k_s, G_K_zs_upper, color='darkorange', label='upper')

    plt.ylabel('E (eV)')

    x = [-0.4, -0.3, 0, 0.3, 0.4]
    labels = ['$\overline{M} \leftarrow$', '$|k|=0.3$', '$\Gamma$', '$|k|=0.3$', r'$\rightarrow\overline{K}$']
    plt.xticks(x, labels, rotation='horizontal')

    plt.ylim(bottom=-0.5, top=max([max(G_M_zs_lower)]))
    plt.xlim(left=-0.4, right=0.4)
    plt.legend()
    plt.savefig('band_M-G-K_45_report.pdf', bbox_inches='tight')
    plt.show()


def append_arr_to_file(filename, vs):
    f = open(filename, 'a')

    for v in vs:
        f.write(str(v))
        f.write(' ')
    f.write('\n')

    f.close()


def append_matrix_to_file(filename, m):
    f = open(filename, 'a')

    for row in m:
        for e in row:
            f.write(str(e))
            f.write(' ')
        f.write('\n')

    f.close()


# appends to  'filename' xs, ys, zs, space seperated with new line between data
#  xs = k_x sample values
#  ys = k_y sample values
#  zs = Hamiltonian eigenvalues (singular (selected one only from pair (either min or max)))
def write_data(filename, xs, ys, zs):
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
                zs[line_counter-2][i] = float(els[i])

        if line_counter == 1:
            zs = np.zeros((len(xs), len(ys)))

        line_counter += 1

    f.close()

    return xs, ys, zs


def plot_min_max_eigen_contours(upper_filename, lower_filename):
    xs_upper, ys_upper, zs_upper = get_data(upper_filename)
    xs_lower, ys_lower, zs_lower = get_data(lower_filename)

    fig, ax = plt.subplots()
    c_upper = ax.contour(xs_upper, ys_upper, zs_upper, levels=[-0.2, 0, 0.2], colors=['darkred', 'red', 'orange'])
    c_lower = ax.contour(xs_lower, ys_lower, zs_lower, levels=[-0.2, 0, 0.2], colors=['darkblue', 'blue', 'cornflowerblue'])
    ax.set_xlabel('$k_x$ $(\AA^{-1})$')
    ax.set_ylabel('$k_y$ $(\AA^{-1})$')

    c_upper.collections[0].set_label('-0.2 eV')
    c_upper.collections[1].set_label('0 eV')
    c_upper.collections[2].set_label('0.2 eV')

    c_lower.collections[0].set_label('-0.2 eV')
    c_lower.collections[1].set_label('0 eV')
    c_lower.collections[2].set_label('0.2 eV')

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=2)
    # fig.colorbar(c_upper, ax=ax)
    # fig.colorbar(c_lower, ax=ax)
    fig.tight_layout()
    fig.gca().set_aspect('equal', adjustable='box')
    fig.savefig('contour_both.pdf')
    fig.show()



def plot(filename):
    xs, ys, zs = get_data(filename)

    fig, ax = plt.subplots()
    c = ax.contour(xs, ys, zs, levels=[-0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3])
    ax.set_xlabel('$k_x$ $(\AA^{-1})$')
    ax.set_ylabel('$k_y$ $(\AA^{-1})$')
    fig.colorbar(c, ax=ax)
    fig.tight_layout()
    fig.savefig('out_lower_sci.pdf')


def main():
    generate_band_structure_data_45()
    generate_energy_contour_data()


if __name__ == '__main__':
    main()

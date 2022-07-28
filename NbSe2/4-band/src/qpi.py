import numpy as np
import time
from scipy.fftpack import fftshift
from params import *
from pauli_matrices import *
from matrix_utils import *


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
    if scattering_type == Scattering.Scalar_2_bands:
        return np.matrix(np.identity(2))
    if scattering_type == Scattering.Scalar_4_band:
        return np.matrix(np.identity(4))
    elif scattering_type == Scattering.Magnetic_2_bands:
        return np.matrix(np.identity(2))
    elif scattering_type == Scattering.Scalar_and_Spin_orbit_2_bands:
        return np.matrix(np.concatenate((np.identity(2), kx*np.identity(2), ky*np.identity(2))))
    elif scattering_type == Scattering.Scalar_and_Spin_orbit_4_bands:
        return np.matrix(np.concatenate((np.identity(4), kx*np.identity(4), ky*np.identity(4))))
    elif scattering_type == Scattering.Scalar_and_Spin_orbit_6_bands:
        return np.matrix(np.concatenate((np.identity(6), kx*np.identity(6), ky*np.identity(6))))
    elif scattering_type == Scattering.Scalar_and_Spin_orbit_18_bands:
        return np.matrix(np.concatenate((np.identity(18), kx*np.identity(18), ky*np.identity(18))))
    print('u-vector for given scattering does not exist')


def v_vector(scattering_type, kx, ky):
    return U_matrix(scattering_type) * u_vector(scattering_type, kx, ky)


def U_matrix(scattering_type, v0=Params.v0, c=Params.c):
    if scattering_type == Scattering.Scalar_2_bands:
        return np.matrix(np.identity(2)) * v0
    if scattering_type == Scattering.Scalar_4_band:
        return np.matrix(np.identity(4)) * v0
    elif scattering_type == Scattering.Magnetic_2_bands:
        return np.matrix(pauli[2]) * v0
    elif scattering_type == Scattering.Scalar_and_Spin_orbit_2_bands:
        return np.concatenate((np.concatenate((v0*np.identity(2), np.zeros((2, 2)), np.zeros((2, 2))), axis=1),
                        np.concatenate((np.zeros((2, 2)), np.zeros((2, 2)), 1j * c * v0 * pauli[2]), axis=1),
                        np.concatenate((np.zeros((2, 2)),  -1j * c * v0 * pauli[2], np.zeros((2, 2))), axis=1)
                        ), axis=0)
    elif scattering_type == Scattering.Scalar_and_Spin_orbit_4_bands:
        d = 4
        pauli_z_4 = make_pauli_z_4()
        return np.concatenate((np.concatenate((v0 * np.identity(d), np.zeros((d, d)), np.zeros((d, d))), axis=1),
                               np.concatenate((np.zeros((d, d)), np.zeros((d, d)), 1j * c * v0 * pauli_z_4), axis=1),
                               np.concatenate((np.zeros((d, d)), -1j * c * v0 * pauli_z_4, np.zeros((d, d))), axis=1)
                               ), axis=0)
    elif scattering_type == Scattering.Scalar_and_Spin_orbit_6_bands:
        d = 6
        pauli_z_6 = make_pauli_z_6()
        return np.concatenate((np.concatenate((v0 * np.identity(d), np.zeros((d, d)), np.zeros((d, d))), axis=1),
                               np.concatenate((np.zeros((d, d)), np.zeros((d, d)), 1j * c * v0 * pauli_z_6), axis=1),
                               np.concatenate((np.zeros((d, d)), -1j * c * v0 * pauli_z_6, np.zeros((d, d))), axis=1)
                               ), axis=0)
    elif scattering_type == Scattering.Scalar_and_Spin_orbit_18_bands:
        d = 18
        pauli_z_18 = make_pauli_z_18()
        return np.concatenate((np.concatenate((v0 * np.identity(d), np.zeros((d, d)), np.zeros((d, d))), axis=1),
                               np.concatenate((np.zeros((d, d)), np.zeros((d, d)), 1j * c * v0 * pauli_z_18), axis=1),
                               np.concatenate((np.zeros((d, d)), -1j * c * v0 * pauli_z_18, np.zeros((d, d))), axis=1)
                               ), axis=0)


def T(scattering_type, Gs, kxs, kys, n=51):
    M = construct_m_matrix(scattering_type, Gs, kxs, kys)

    T_matrix = np.matrix(np.zeros((M.shape[0], M.shape[1]), dtype=complex))

    for i in range(n):
        T_matrix = T_matrix + M ** i

    return T_matrix


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


def qpi(scattering_type, G_0_k, kxs, kys):
    if scattering_type == Scattering.Scalar_2_bands:
        print('2-band scalar QPI')
    if scattering_type == Scattering.Scalar_4_band:
        print('4-band scalar QPI')
    elif scattering_type == Scattering.Magnetic_2_bands:
        print('magnetic QPI')
    elif scattering_type == Scattering.Scalar_and_Spin_orbit_2_bands:
        print('scalar and spin orbit QPI 2 bands')
    elif scattering_type == Scattering.Scalar_and_Spin_orbit_4_bands:
        print('scalar and spin orbit QPI 4 bands')
    elif scattering_type == Scattering.Scalar_and_Spin_orbit_6_bands:
        print('scalar and spin orbit QPI 6 bands')
    elif scattering_type == Scattering.Scalar_and_Spin_orbit_18_bands:
        print('scalar and spin orbit QPI 18 bands')

    start_time = time.time()

    kx_n = len(kxs)
    ky_n = len(kys)

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

    del G_0_primed_k_primed
    del G_0_primed_k

    print('* construct g_0(r,-r,w)')
    # construct g_0(r,-r,w)
    g_0_r_minus_r = make_g_0_r_minus_r(g_0_r, g_0_minus_r)  # matrix of matrices

    del g_0_r
    del g_0_minus_r

    ''' Take FFT[g(r,-r,w)] and IFFT[g*(r,-r,w)] '''
    print('* Take FFT[g(r,-r,w)] and IFFT[g*(r,-r,w)]')

    # FFT[g(r,-r,w)] = G_0(k,k-q)
    g_0_q = fftshift(fft_of_mesh_of_matrices(g_0_r_minus_r)) / (kx_n*ky_n)
    # FFT[g*(r,-r,w)] = G*_0(k,k+q)
    g_0_conj_minus_q = fftshift(fft_of_mesh_of_matrices(g_0_r_minus_r.conjugate())) / (kx_n * ky_n)

    del g_0_r_minus_r

    print('*** T')

    t = T(scattering_type, G_0_k, kxs, kys, n=101)
    t_congugate = t.conjugate()
    print('max abs(T) element (non-normalization) = ', np.max(np.abs(t)))

    g_0_q_trace = np.zeros((kx_n, ky_n), dtype=complex)
    for i in range(kx_n):
        for j in range(ky_n):
            g_0_q_trace[i, j] = g_0_q[i, j].trace()

    print('max abs(g_0_q_trace) element = ', np.max(np.abs(g_0_q_trace)))
    del g_0_q_trace

    print('*** abs(rho(q))')

    rho_q = calculate_rho(g_0_q=g_0_q, g_0_conj_minus_q=g_0_conj_minus_q, t=t, t_c=t_congugate)

    del t
    del t_congugate
    del g_0_q
    del g_0_conj_minus_q

    time_end = time.time()
    dt = time_end - start_time
    average_dt = dt / ((kx_n) * (ky_n))
    print('time taken: {:.6f} seconds'.format(dt))
    print('average time per q vector: {:.6f} seconds'.format(average_dt))
    return rho_q
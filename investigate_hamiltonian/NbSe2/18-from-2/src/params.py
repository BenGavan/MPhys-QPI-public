import numpy as np


class Params:
    a_0 = 3.44  # \AA (Angstrom)
    a_1 = a_0 * np.array([np.sqrt(3) / 2, -1 / 2])
    a_2 = a_0 * np.array([0, 1])

    b_1 = (2 * np.pi / a_0) * np.array([2 * np.sqrt(3) / 3, 0])
    b_2 = (2 * np.pi / a_0) * np.array([np.sqrt(3) / 3, 1])

    a_0_sc = a_0 * 3
    a_1_sc = a_1 * 3
    a_2_sc = a_2 * 3

    b_1_sc = b_1 / 3
    b_2_sc = b_2 / 3

    G = np.array([0, 0])
    M = 0.5 * b_1
    K = np.array([0.5 * b_1[0], 0.5 * b_1[0] * np.tan(30 * np.pi / 180)])

    G_sc = G / 3
    M_sc = M / 3
    K_sc = K / 3

    c = 60
    v0 = 0.1
    E_f = -3.9285  # eV
    delta_E_f = -50 * pow(10, -3)  # eV (should be 50 meV but in eV)
    omega = E_f + delta_E_f


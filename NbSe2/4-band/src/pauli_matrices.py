import numpy as np


pauli = np.array((
    ((0, 1), (1, 0)),
    ((0, -1j), (1j, 0)),
    ((1, 0), (0, -1))
))


def make_pauli_z_18():
    pz = np.zeros((18, 18), dtype=complex)
    for i in range(9):
        for j in range(9):
            if i != j:
                continue
            for k in range(2):
                for l in range(2):
                    pz[i*2 + k, j*2 + l] = pauli[2][k, l]

    return pz


def make_pauli_z_6():
    pz = np.zeros((6, 6), dtype=complex)
    for i in range(3):
        for j in range(3):
            if i != j:
                continue
            for k in range(2):
                for l in range(2):
                    pz[i*2 + k, j*2 + l] = pauli[2][k, l]

    return pz


def make_pauli_z_4():
    pz = np.zeros((4, 4), dtype=complex)
    pz[0, 0] = 1
    pz[1, 1] = 1
    pz[2, 2] = -1
    pz[3, 3] = -1
    return pz
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from sympy.physics.paulialgebra import Pauli

if __name__ == '__main__':
    pass

m = 0.0168 #in ev A^(-2)
alpha_4 = -2.03 # A^(-2)
alpha_6 = 87.5 # A^(-4)
v = 3.13 #eV A^(-1)
beta_3 = -2.01 #A^(-2)
beta_5 = 323 #A^(-4)
lamb = -41.7 # eV A(-3)
gamma_5 = 2.43 #A^(-2)
E_0 = -0.352 #eV

pauli = np.array((
                                   ((0, 1), (1, 0)),
                                   ((0, -1j), (1j, 0)),
                                   ((1, 0), (0, -1))
                                  ))

I = np.array(((1,0), (0,1)))

print(pauli[1]*3)
print(I)
print((pauli[0]**2)*I)


def E(k):
  return 1 + alpha_4*k**2 + alpha_6*k**4

def V(k):
  return 1

def Lambda(k):
  return -1

def Hamiltonian(k_x, k_y):
  k2 = k_x**2 + k_y**2
  k = np.sqrt(k2)
  return  V(k)*(k_x*pauli[1] - k_y*pauli[0]) + Lambda(k) * (3*k_x**2 - k_y**2)*k_y*pauli[2]
  # return (E_0 + (k2/(2*m))*E(k)) * np.identity(2) + V(k)*(k_x*pauli[1] - k_y*pauli[0]) + Lambda(k) * (3*k_x**2 - k_y**2)*k_y*pauli[2]

print(Hamiltonian(2,2))



########################################################################
########################################################################
########################################################################
########################################################################

import numpy as np
import sympy as sp
from sympy.matrices import Matrix



from scipy import linalg


# Pauli Matrices
sigma_1 = np.matrix([[0, 1], [1, 0]])
sigma_2 = np.matrix([[0, 1], [1, 0]])
sigma_1 = np.matrix([[0, 1], [1, 0]])
sigma = np.array([sigma_1])


def sympy_testing():
    M = Matrix(2, 2, [1, 1, 1, 1])
    dM = M.diagonalize()
    P = dM[0]
    P_inv = P.inv()
    D = dM[1]

    print("P", P)
    print("P^-1", P_inv)
    print("D", D)

    print(dM)
    exit()

    h = hamiltonian(1, 2)
    print(h)
    print(sigma_1)
    print(sigma[0])

    print(sp.sqrt(8))

    M = Matrix(3, 3, [1, 2, 3, 4, 5, 6, 7, 8, 9])
    print(M[0], M[1])
    exit()
    dM = M.diagonalize()
    print(M)
    print("-------")
    print(dM)

    M = Matrix(3, 3, [1, 1, 1, 1, 1, 1, 1, 1, 1])
    dM = M.diagonalize()
    print(M)
    print("-------")
    print(dM)
    print("dM size = ", len(dM))
    print(dM[0])
    print(dM[1])



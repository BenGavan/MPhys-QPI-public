import numpy as np
import enum


class Scattering(enum.Enum):
    Scalar_2_bands = 0
    Scalar_4_band = 1
    Magnetic_2_bands = 2
    Scalar_and_Spin_orbit_2_bands = 3
    Scalar_and_Spin_orbit_4_bands = 4
    Scalar_and_Spin_orbit_6_bands = 5
    Scalar_and_Spin_orbit_18_bands = 6


class Orbital(enum.Enum):
    orbital_3D = 3
    orbital_5S = 2
    orbital_5P = 1
    gaussian = -1


class RunParams:
    USE_CACHE = True
    PLOT_PHI_K = True
    PLOT_PHI_R = True
    PLOT_GREENS = True
    USE_PHI_K_FILE = False
    WRITE_18_BAND_RHO_Q = True
    WRITE_4_BAND_RHO_Q = True
    WRITE_2_BAND_RHO_Q = True
    READ_18_BAND_RHO_Q = True
    READ_4_BAND_RHO_Q = True
    READ_2_BAND_RHO_Q = True


class Params:
    a_0 = 3.44  # \AA (Angstrom)
    a_1 = a_0 * np.array([np.sqrt(3) / 2, -1 / 2])
    a_2 = a_0 * np.array([0, 1])

    b_1 = (2 * np.pi / a_0) * np.array([2 * np.sqrt(3) / 3, 0])
    b_2 = (2 * np.pi / a_0) * np.array([np.sqrt(3) / 3, 1])

    G = np.array([0, 0])
    M = 0.5 * b_1
    K = np.array([0.5 * b_1[0], 0.5 * b_1[0] * np.tan(30 * np.pi / 180)])

    a_0_sc = a_0 * 3
    a_1_sc = a_1 * 3
    a_2_sc = a_2 * 3

    b_1_sc = b_1 / 3
    b_2_sc = b_2 / 3

    G_sc = G / 3
    M_sc = M / 3
    K_sc = K / 3

    c = 60
    v0 = 0.1
    E_f = -2.719  # eV
    delta_E_f = -50 * pow(10, -3)  # eV (should be 50 meV but in eV)
    omega = E_f + delta_E_f

    kxn = 300
    kyn = 300

    eigenbasis_phi = [0, 0, 0, 0]

    # # alpha_gauss = np.pi**2 / (2*a_0**2)
    alpha_gauss = 2
    alpha_central_gauss = 3.6
    # sub_gauss_alpha = 1
    #
    # alpha_outer_gauss = 1
    # alpha_inner_gauss = 1
    #
    # sigma_outer_gauss = 0.9
    # sigma_inner_gauss = 0.55
    #
    # sigma = 2
    #
    # # lower_k = 0
    # # mid_k = 0.65
    # # upper_k = 1.58
    #
    # # inner_v = 1
    # # outer_v = 1
    #
    k_pocket_radius = 0.25
    # k_pocket_weighting = 0.25
    #
    # Gam_pocket_radius = 0.65
    # Gam_pocket_weighting = 0.25
    #
    # z = 0

    xn_sub = 150
    x_length_sub = 40

    basis_factor = 3.5

    lower_triangle_weighting = 0
    upper_triangle_weighting = 1
    small_triangle_weighting = 0

    lower_triangle_alpha = 1
    upper_triangle_alpha = 1
    small_triangle_alpha = 1

    threshold = 10


def set_rho_q_run_params(read, write):
    RunParams.WRITE_18_BAND_RHO_Q = write
    RunParams.WRITE_4_BAND_RHO_Q = write
    RunParams.WRITE_2_BAND_RHO_Q = write
    RunParams.READ_18_BAND_RHO_Q = read
    RunParams.READ_4_BAND_RHO_Q = read
    RunParams.READ_2_BAND_RHO_Q = read

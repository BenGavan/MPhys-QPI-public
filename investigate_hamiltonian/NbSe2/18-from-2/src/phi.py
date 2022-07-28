import numpy as np
from params import *
from files import *
from plotting import *


def unnormalized_gaussian(x, y, a):
    r2 = x ** 2 + y ** 2 + Params.z ** 2
    return np.exp(-r2 / a)


def gaussian(x, y, a):
    r2 = x**2 + y**2
    n = 10 / pow(a*np.pi, 3/2)
    return n * np.exp(-r2/a)
    # return (1 / (a * np.sqrt(2 * np.pi))) * np.exp(-r2 / (2 * a**2))


def marr_wavelet(x, y, s):
    r2 = x**2 + y**2
    r2s = r2 / (2 * s**2)
    A = (1 / (np.pi * pow(s, 4)))
    return A * (1 - r2s) * np.exp(-r2s)


def modified_ricker_wavelet(x, y, s):
    r2 = x ** 2 + y ** 2
    # r2s = (r2 / (2 * s ** 2))**2
    r2s = (r2 / (2 * s ** 2))**2
    A = (1 / (np.pi * pow(s, 4)))
    return A * r2s * np.exp(-r2s)


def warped_hex(x, y, a):
    # r2 = x**2 + y**2
    theta = np.arctan(x / y)

    return (3 + np.cos(3*theta) ** 2) * unnormalized_gaussian(x, y, a) - unnormalized_gaussian(x, y, 1)


def warped_tri(x, y, a):
    # r2 = x**2 + y**2
    theta = 0
    if x >= 0 and y >= 0:
        theta = np.arctan(y / x)
    elif x <= 0:
        theta = np.pi + np.arctan(y / x)
    elif x >= 0 and y <= 0:
        theta = 2 * np.pi - abs(np.arctan(y / x))

    theta += np.pi

    # theta = abs(np.arctan(x / y)) * y / np.abs(y)

    # return np.cos(1.5 * theta) ** 2
    #
    return (3 + np.cos(1.5*theta) ** 2) * (unnormalized_gaussian(x, y, a) - 0.5 * unnormalized_gaussian(x, y, 1))


def construct_wavelet(xs, ys, s):
    xn = len(xs)
    yn = len(ys)

    phi_k = np.zeros((xn, yn), dtype=float)

    for i in range(xn):
        for j in range(yn):
            x = xs[i]
            y = ys[j]
            phi_k[i, j] = modified_ricker_wavelet(x, y, s)

    return phi_k


def construct_gauss_sombrero(xs, ys, sigma_outer, sigma_inner):
    xn = len(xs)
    yn = len(ys)

    phi_k = np.zeros((xn, yn), dtype=float)

    def gauss(x, y, s):
        r2 = x**2 + y**2
        a = 2 * (s**2)
        return np.exp(-r2/a)

    for i in range(xn):
        for j in range(yn):
            x = xs[i]
            y = ys[j]
            phi_k[i, j] = gauss(x, y, sigma_outer) - gauss(x, y, sigma_inner)

    return phi_k


def construct_selective_phi(xs, ys):
    xn = len(xs)
    yn = len(ys)

    phi_k = np.zeros((xn, yn), dtype=float)
    # phi_k_primed = np.zeros((kx_n, ky_n), dtype=float)

    for i in range(xs):
        for j in range(ys):
            #
            # ########################################################
            # ########################################################
            # ########################################################
            # # Centre Gamma point
            # kx = kxs[i]
            # ky = kys[j]
            # k = np.sqrt(kx ** 2 + ky ** 2)
            #
            # if 0 < k < Params.Gam_pocket_radius:
            #     phi_k[i, j] = Params.Gam_pocket_weighting
            # #
            # # ########################################################
            # kx = kxs[i] - Params.K[0]
            # ky = kys[j] - Params.K[1]
            # k = np.sqrt(kx ** 2 + ky ** 2)
            #
            # if 0 < k < Params.k_pocket_radius:
            #     phi_k[i, j] = Params.k_pocket_weighting
            # #
            # kx = kxs[i] + Params.K[0]
            # ky = kys[j] + Params.K[1]
            # k = np.sqrt(kx ** 2 + ky ** 2)
            #
            # if 0 < k < Params.k_pocket_radius:
            #     phi_k[i, j] = Params.k_pocket_weighting
            #
            # kx = kxs[i] + Params.K[0]
            # ky = kys[j] - Params.K[1]
            # k = np.sqrt(kx ** 2 + ky ** 2)
            #
            # if 0 < k < Params.k_pocket_radius:
            #     phi_k[i, j] = Params.k_pocket_weighting
            #
            # kx = kxs[i] - Params.K[0]
            # ky = kys[j] + Params.K[1]
            # k = np.sqrt(kx ** 2 + ky ** 2)
            #
            # if 0 < k < Params.k_pocket_radius:
            #     phi_k[i, j] = Params.k_pocket_weighting
            #
            # kx = kxs[i] - 0
            # ky = kys[j] + np.sqrt(Params.K[0] ** 2 + Params.K[1] ** 2)
            # k = np.sqrt(kx ** 2 + ky ** 2)
            #
            # if 0 < k < Params.k_pocket_radius:
            #     phi_k[i, j] = Params.k_pocket_weighting
            #
            # kx = kxs[i] - 0
            # ky = kys[j] - np.sqrt(Params.K[0] ** 2 + Params.K[1] ** 2)
            # k = np.sqrt(kx ** 2 + ky ** 2)
            #
            # if 0 < k < Params.k_pocket_radius:
            #     phi_k[i, j] = Params.k_pocket_weighting
            # # ########################################################

            # Apply gaussian over only K and \Gamma
            phi_k[i, j] = 1
            phi_k[i, j] = phi_k[i, j] * gaussian(xs[i], ys[j], Params.alpha_gauss)

            # phi_k[i, j] = 1

            ########################################################
            ########################################################
    return phi_k


def construct_phi_r_nb_warped(xs, ys):
    xn = len(xs)
    yn = len(ys)

    phi_r = np.zeros((xn, yn), dtype=float)

    for i in range(xn):
        for j in range(yn):
            x = xs[i]
            y = ys[j]

            phi_r[i, j] += warped_hex(x, y, Params.alpha_gauss)

    return phi_r


def construct_phi_r_se_warped(xs, ys):
    xn = len(xs)
    yn = len(ys)

    phi_r = np.zeros((xn, yn), dtype=float)

    for i in range(xn):
        for j in range(yn):
            x = xs[i]
            y = ys[j]

            phi_r[i, j] += warped_tri(x, y, Params.alpha_gauss)


    return phi_r


def construct_phi_r_nb(xs, ys):
    xn = len(xs)
    yn = len(ys)

    phi_r = np.zeros((xn, yn), dtype=float)

    alpha = 1
    beta = 15
    gamma = 10

    for i in range(xn):
        for j in range(yn):
            x = xs[i]
            y = ys[j]
            # phi_r[i, j] = phi_r[i, j] + gaussian(x, y, Params.alpha_central_gauss) * central_magnitude
            phi_r[i, j] = phi_r[i, j] + unnormalized_gaussian(x, y, 10) * beta - unnormalized_gaussian(x, y, Params.sub_gauss_alpha) * gamma
            # phi_r[i, j] += unnormalized_gaussian(x, y, Params.alpha_gauss)

            x = xs[i] - Params.a_1[0]
            y = ys[j] - Params.a_1[1]
            phi_r[i, j] += unnormalized_gaussian(x, y, Params.alpha_gauss) * alpha

            x = xs[i] + Params.a_1[0]
            y = ys[j] + Params.a_1[1]
            phi_r[i, j] += unnormalized_gaussian(x, y, Params.alpha_gauss) * alpha

            x = xs[i] - Params.a_2[0]
            y = ys[j] - Params.a_2[1]
            phi_r[i, j] += unnormalized_gaussian(x, y, Params.alpha_gauss) * alpha

            x = xs[i] + Params.a_2[0]
            y = ys[j] + Params.a_2[1]
            phi_r[i, j] += unnormalized_gaussian(x, y, Params.alpha_gauss) * alpha

            x = xs[i] - Params.a_1[0] - Params.a_2[0]
            y = ys[j] - Params.a_1[1] - Params.a_2[1]
            phi_r[i, j] += unnormalized_gaussian(x, y, Params.alpha_gauss) * alpha

            x = xs[i] + Params.a_1[0] + Params.a_2[0]
            y = ys[j] + Params.a_1[1] + Params.a_2[1]
            phi_r[i, j] += unnormalized_gaussian(x, y, Params.alpha_gauss) * alpha

    return phi_r


def construct_phi_r_se(xs, ys):
    xn = len(xs)
    yn = len(ys)

    phi_r = np.zeros((xn, yn), dtype=float)

    # Def where Se atoms are (from a1, a2 basis to )
    s1 = np.matrix((1 / 3) * Params.a_1 + (2 / 3) * Params.a_2)
    s2 = np.matrix((-2 / 3) * Params.a_1 - (1 / 3) * Params.a_2)
    s3 = np.matrix((1 / 3) * Params.a_1 + (-1 / 3) * Params.a_2)

    # Rotate where Se atoms are
    theta = np.pi / 6
    theta = 0
    R_z = np.matrix(((np.cos(theta), -np.sin(theta)),
                     (np.sin(theta), np.cos(theta))))

    s1 = R_z * s1.transpose()
    s2 = R_z * s2.transpose()
    s3 = R_z * s3.transpose()

    xs_top = []
    ys_top = []

    xs_bottom = []
    ys_bottom = []

    # # Top first
    xs_top.append(s1[0])
    ys_top.append(s1[1])

    xs_top.append(s2[0])
    ys_top.append(s2[1])

    xs_top.append(s3[0])
    ys_top.append(s3[1])

    # Bottom first
    xs_bottom.append(-s1[0])
    ys_bottom.append(-s1[1])

    xs_bottom.append(-s2[0])
    ys_bottom.append(-s2[1])

    xs_bottom.append(-s3[0])
    ys_bottom.append(-s3[1])

    # Bottom second
    xs_bottom.append(2 * s1[0])
    ys_bottom.append(2 * s1[1])

    xs_bottom.append(2 * s2[0])
    ys_bottom.append(2 * s2[1])

    xs_bottom.append(2 * s3[0])
    ys_bottom.append(2 * s3[1])

    # Top second
    xs_top.append(-2 * s1[0])
    ys_top.append(-2 * s1[1])

    xs_top.append(-2 * s2[0])
    ys_top.append(-2 * s2[1])

    xs_top.append(-2 * s3[0])  # xs.append(-Params.a_1[0] - 2 * Params.a_2[0] + s1[0])
    ys_top.append(-2 * s3[1])  # ys.append(-Params.a_1[1] - 2 * Params.a_2[1] + s1[1])

    for i in range(xn):
        for j in range(yn):
            # x = xs[i]
            # y = ys[j]
            # phi_r[i, j] += unnormalized_gaussian(x, y, Params.alpha_central_gauss) * 4 - unnormalized_gaussian(x, y, 1)
            #
            # x = xs[i] - s1[0]
            # y = ys[j] - s1[1]
            # phi_r[i, j] += unnormalized_gaussian(x, y, Params.alpha_gauss) * 0.5
            #
            # x = xs[i] - s2[0]
            # y = ys[j] - s2[1]
            # phi_r[i, j] += unnormalized_gaussian(x, y, Params.alpha_gauss) * 0.5
            #
            # x = xs[i] - s3[0]
            # y = ys[j] - s3[1]
            # phi_r[i, j] += unnormalized_gaussian(x, y, Params.alpha_gauss) * 0.5

            # central
            # x = xs[i]
            # y = ys[j]
            #
            # r2 = x ** 2 + y ** 2
            # phi_r[i, j] += np.exp(-r2 / Params.alpha_central_gauss)


            for se_x, se_y in zip(xs_top, ys_top):
                    x = xs[i] - se_x
                    y = ys[j] - se_y

                    r2 = x ** 2 + y ** 2
                    phi_r[i, j] += np.exp(-r2 / Params.alpha_gauss)
                    # phi_r[i, j] += unnormalized_gaussian(x, y, Params.alpha_gauss)


            for se_x, se_y in zip(xs_bottom, ys_bottom):
                    x = xs[i] - se_x
                    y = ys[j] - se_y

                    r2 = x ** 2 + y ** 2
                    phi_r[i, j] += np.exp(-r2 / Params.alpha_gauss)

    return phi_r


def plot_phi_r(xn, yn, x_length, y_length, phi_r_func):
    xs, ys = generate_axes(xn, yn, x_length, y_length)
    phi_r = phi_r_func(xs, ys)

    plot_heatmap(xs, ys, (phi_r.transpose()),
             filename=get_phi_r_sub_plot_filepath(xn=xn, x_length=x_length),
             x_label='$x$ $(\AA)$', y_label='$y$ $(\AA)$', show_bz=False, show_nb=True, show_se=True)


def make_and_save_phi_r(xn, yn, x_length, y_length, kxn, kyn, kx_length, ky_length, phi_r_func):
    xs, ys = generate_axes(xn, yn, x_length, y_length)
    phi_r = phi_r_func(xs, ys)

    plot_heatmap(xs, ys, abs(phi_r.transpose()), 'phi_r.pdf', x_label='x', y_label='y')

    write_phi_r_data('phi_r.dat', xn, yn, x_length, y_length, kxn, kyn, kx_length, ky_length, phi_r)



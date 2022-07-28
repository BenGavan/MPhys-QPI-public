import numpy as np
from scipy.interpolate import interp1d
from params import *
from files import *
from plotting import *


def unnormalized_gaussian(x, y, a):
    r2 = x ** 2 + y ** 2
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
    return A * (1 - 1.5*r2s) * np.exp(-r2s)


def modified_ricker_wavelet(x, y, s):
    r2 = x ** 2 + y ** 2
    # r2s = (r2 / (2 * s ** 2))**2
    r2s = (r2 / (2 * s ** 2))**2
    A = (1 / (np.pi * pow(s, 4)))
    return A * r2s * np.exp(-r2s)


def three_peak_ricker_wavelet(x, y, s):
    r2 = x ** 2 + y ** 2
    rs2 = r2 / (s ** 2)
    rs4 = rs2**2
    return (1 - rs2 + (1/8) * rs4) * np.exp(-rs2/4)


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


def construct_blank_phi(xs, ys):
    xn = len(xs)
    yn = len(ys)

    phi_r = np.zeros((xn, yn), dtype=float)
    return phi_r


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

    def Y_2_2(x, y, z):
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        return 0.25 * np.sqrt(15 / (2*np.pi)) * pow(x + 1j*y, 2) / (r**2)

    def Y_neg2_2(x, y, z):
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        return 0.25 * np.sqrt(15 / (2*np.pi)) * pow(x - 1j*y, 2) / (r**2)

    def Y_neg1_2(x, y, z):
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        return 0.5 * np.sqrt(15 / (2 * np.pi)) * (x - 1j * y) * z / (r ** 2)

    def Y_1_2(x, y, z):
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        return -0.5 * np.sqrt(15 / (2 * np.pi)) * (x + 1j * y) * z / (r ** 2)

    def Y_0_2(x, y, z):
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        return 0.25 * np.sqrt(5 / (2 * np.pi)) * (3*z**2 - r**2) / (r ** 2)


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
        elif orbital == Orbital.orbital_3D:
            return -1

    for i in range(xn):
        for j in range(yn):
            x = xs[i]
            y = ys[j]

            z = 0

            r2 = x**2 + y**2 + z**2
            r = np.sqrt(r2)

            phi_r[i, j] = R_fit(r) * Theta(x, y, z)

    return phi_r


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
            # phi_r[i, j] = phi_r[i, j] + unnormalized_gaussian(x, y, 10) * beta - unnormalized_gaussian(x, y, Params.sub_gauss_alpha) * gamma
            phi_r[i, j] += unnormalized_gaussian(x, y, Params.alpha_gauss)

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
    # xs_bottom.append(-s1[0])
    # ys_bottom.append(-s1[1])
    #
    # xs_bottom.append(-s2[0])
    # ys_bottom.append(-s2[1])
    #
    # xs_bottom.append(-s3[0])
    # ys_bottom.append(-s3[1])

    # Bottom second
    # xs_bottom.append(2 * s1[0])
    # ys_bottom.append(2 * s1[1])
    #
    # xs_bottom.append(2 * s2[0])
    # ys_bottom.append(2 * s2[1])
    #
    # xs_bottom.append(2 * s3[0])
    # ys_bottom.append(2 * s3[1])

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
            x = xs[i]
            y = ys[j]

            r2 = x ** 2 + y ** 2
            phi_r[i, j] += np.exp(-r2 / Params.alpha_central_gauss)


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


def plot_phi_r(xn, yn, x_length, y_length, phi_r_func, version_number):
    xs, ys = generate_axes(xn, yn, x_length, y_length)
    phi_r = phi_r_func(xs, ys)

    plot_heatmap(xs, ys, (phi_r.transpose()),
             filename=get_phi_r_sub_plot_filepath(xn=xn, x_length=x_length, version_number=version_number),
             x_label='$x$ $(\AA)$', y_label='$y$ $(\AA)$', show_bz=False, show_nb=True, show_se=True)


def make_and_save_phi_r(xn, yn, x_length, y_length, kxn, kyn, kx_length, ky_length, phi_r_func):
    xs, ys = generate_axes(xn, yn, x_length, y_length)
    phi_r = phi_r_func(xs, ys)

    plot_heatmap(xs, ys, abs(phi_r.transpose()), 'phi_r.pdf', x_label='x', y_label='y')

    write_phi_r_data('phi_r.dat', xn, yn, x_length, y_length, kxn, kyn, kx_length, ky_length, phi_r)


def make_select_paper_phi_k(kxs, kys):
    kxn = len(kxs)
    kyn = len(kys)

    K = Params.K
    M = Params.M
    G = Params.G

    b1 = Params.b_1
    b2 = Params.b_2

    bz_cs = [K, (K[0], -K[1]), -2 * K + 2 * M, (-K[0], -K[1]), (-K[0], K[1]), 2 * K - 2 * M]
    ms_cs = [M]

    # init phi(k)
    phi_k = np.zeros((kxn, kyn), dtype=float)

    # weird extended regions
    from point_in_polygon import Point
    from point_in_polygon import is_point_in_poly

    cs = [G, [b1[0]/2, 0.3], b2, G]
    ps = []
    for c in cs:
        ps.append(Point(c[0], c[1]))

    for j in range(6):
        theta = np.pi / 3
        # theta = 0
        R_z = np.matrix(((np.cos(theta), -np.sin(theta)),
                         (np.sin(theta), np.cos(theta))))

        for i in range(len(ps)):
            n = (R_z * np.matrix([ps[i].x, ps[i].y]).transpose()).transpose()
            ps[i].x = n[0, 0]
            ps[i].y = n[0, 1]

        for i in range(kxn):
            for j in range(kyn):
                kx = kxs[i]
                ky = kys[j]
                if is_point_in_poly(Point(kx, ky), ps):
                    phi_k[i, j] = 1

    # central intensity
    for i in range(kxn):
        for j in range(kyn):
            kx = kxs[i]
            ky = kys[j]
            k = np.sqrt(kx ** 2 + ky ** 2)
            if k < 0.6:
                phi_k[i, j] = 2

    return phi_k


def make_cdw_third_k_phi_k(kxs, kys):
    kxn = len(kxs)
    kyn = len(kys)

    K = Params.K / 3
    M = Params.M / 3

    cs = [K, (K[0], -K[1]), -2 * K + 2 * M, (-K[0], -K[1]), (-K[0], K[1]), 2 * K - 2 * M]

    K = Params.K * 2 / 3
    M = Params.M * 2 / 3

    import point_in_polygon

    r1 = [(K[0], 0.05), (K[0], -0.05), (-K[0], -0.05), (-K[0], 0.05), (K[0], 0.05)]
    r1_points = [point_in_polygon.Point(r[0], r[1]) for r in r1]
    point_in_polygon.plot(point_in_polygon.Point(0, 0), r1_points, 'r1.png')

    # M = np.array([0, M[0]])
    # r2 = [M-(K-M)*0.05, M+(K-M)*0.05, -M+(-K-M)*0.05, -M-(-K-M)*0.05, M-(K-M)*0.05]
    r2 = [(0.7, 1), (0.65, 1), (-0.7, -1), (-0.65, -1), (0.7, 1)]
    r2_points = [point_in_polygon.Point(r[0], r[1]) for r in r2]
    point_in_polygon.plot(point_in_polygon.Point(0, 0), r2_points, 'r2.png')

    r3 = [(-0.7, 1), (-0.65, 1), (0.7, -1), (0.65, -1), (-0.7, 1)]
    r3_points = [point_in_polygon.Point(r[0], r[1]) for r in r3]
    point_in_polygon.plot(point_in_polygon.Point(0, 0), r3_points, 'r3.png')
    #
    # K = Params.K / 3
    # M = Params.M / 3
    # #
    # # import point_in_polygon
    # cs = [K, (K[0], -K[1]), -2 * K + 2 * M, (-K[0], -K[1]), (-K[0], K[1]), 2 * K - 2 * M, K]
    # bz3_ps = [point_in_polygon.Point(k[0], k[1]) for k in cs]
    #

    #
    # if RunParams.PLOT_PHI_R:
    #     xs_sub, ys_sub = generate_axes(Params.xn_sub, Params.xn_sub, Params.x_length_sub, Params.x_length_sub)
    #     # phi_r = np.zeros((10, 10), dtype=float)
    #     phi_r_sub = construct_phi_r_se(xs_sub, ys_sub)
    #
    #     plot_heatmap(xs_sub, ys_sub, (phi_r_sub.transpose()),
    #                  filename=get_phi_r_sub_plot_filepath(xn=Params.xn_sub, x_length=Params.x_length_sub,
    #                                                       version_number=get_version_number()),
    #                  x_label='$x$ $(\AA)$', y_label='$y$ $(\AA)$', show_bz=False, show_nb=True, show_se=True)
    #
    #     plot_heatmap(xs, ys, abs(phi_r_sub.transpose()),
    #                  filename=get_phi_r_plot_filepath(xn=xn, x_length=x_length, version_number=get_version_number()),
    #                  x_label='$x$ $(\AA)$', y_label='$y$ $(\AA)$', show_bz=False)
    #
    #
    phi_k = np.zeros((kxn, kyn), dtype=float)
    for i in range(kxn):
        for j in range(kyn):
    #         # kx = kxs[i]
    #         # ky = kys[j]
    #         # phi_k[i, j] = warped_hex(kx, ky, Params.alpha_gauss)
    #
    #         # if np.sqrt(kx**2 + ky**2) < 0.6:
    #         #     phi_k[i, j] = 1
    #


            # if point_in_polygon.is_point_in_poly(point_in_polygon.Point(kx, ky), bz3_ps):
            #     phi_k[i, j] = 1
    #
            for p in cs:
                kx = kxs[i] - p[0]
                ky = kys[j] - p[1]

                k = np.sqrt(kx**2 + ky**2)
                # phi_k[i, j] += unnormalized_gaussian(kx, ky, Params.alpha_gauss)
                if k < Params.k_pocket_radius:
                    phi_k[i, j] = 1
    #
            kx = kxs[i]
            ky = kys[j]
            k = np.sqrt(kx**2 + ky**2)
    #         # phi_k[i, j] += unnormalized_gaussian(kx, ky, Params.alpha_gauss)
    #         # if k < 0.6:
    #         #     phi_k[i, j] = 1
            if k > 0.55:
                phi_k[i, j] = 0
    #
            # point in polygon
            kx = kxs[i]
            ky = kys[j]
            if point_in_polygon.is_point_in_poly(point_in_polygon.Point(kx, ky), r1_points):
                phi_k[i, j] = 0

            if point_in_polygon.is_point_in_poly(point_in_polygon.Point(kx, ky), r2_points):
                phi_k[i, j] = 0

            if point_in_polygon.is_point_in_poly(point_in_polygon.Point(kx, ky), r3_points):
                phi_k[i, j] = 0

    # phi_k = construct_phi_k(kxn, kyn, kx_length, ky_length)
    #
    # for i in range(kxn):
    #     for j in range(kyn):
    #         kx = kxs[i]
    #         ky = kys[j]
    #         phi_k[i, j] = phi_k[i, j] - 5 * gaussian(kx, ky, a=1)
    #         if phi_k[i, j] < 0:
    #             phi_k[i, j] = 0
    #
    # if RunParams.PLOT_PHI_K:
    #     plot_heatmap(kxs, kys, abs(phi_k.transpose()),
    #                  filename=get_phi_k_plot_filepath(kxn=kxn, kx_length=kx_length,
    #                                                   version_number=get_version_number()),
    #                  x_label='$k_x$ $(\AA^{-1})$', y_label='$k_y$ $(\AA^{-1})$', show_bz=True)
    return phi_k


def make_discrete_SoD_triangles_phi_r(xs, ys):
    xn = len(xs)
    yn = len(ys)

    from point_in_polygon import is_point_in_poly
    import point_in_polygon

    a1 = Params.a_1 * Params.basis_factor
    a2 = Params.a_2 * Params.basis_factor

    lower_triangle_coords = [2*a1 + a2, -a1 - 2*a2, -a1 + a2]
    upper_triangle_coords = [a1 - a2, -2*a1 - a2, a1 + 2*a2]

    # rotate
    theta = np.pi / 6
    # theta = 0
    R_z = np.matrix(((np.cos(theta), -np.sin(theta)),
                     (np.sin(theta), np.cos(theta))))

    for i in range(len(lower_triangle_coords)):
        n = (R_z * np.matrix(lower_triangle_coords[i]).transpose()).transpose()
        lower_triangle_coords[i][0] = n[0, 0]
        lower_triangle_coords[i][1] = n[0, 1]

    for i in range(len(upper_triangle_coords)):
        n = (R_z * np.matrix(upper_triangle_coords[i]).transpose()).transpose()
        upper_triangle_coords[i][0] = n[0, 0]
        upper_triangle_coords[i][1] = n[0, 1]

    lower_triangle_points = []
    upper_triangle_points = []

    for c in lower_triangle_coords:
        lower_triangle_points.append(point_in_polygon.Point(c[0], c[1]))

    for c in upper_triangle_coords:
        upper_triangle_points.append(point_in_polygon.Point(c[0], c[1]))

    lower_triangle_points.append(lower_triangle_points[0])
    upper_triangle_points.append(upper_triangle_points[0])

    point_in_polygon.plot(lower_triangle_points[1], lower_triangle_points, 'lower_tri.png')

    phi_r = np.zeros((xn, yn), dtype=float)
    for i in range(xn):
        for j in range(yn):
            p = point_in_polygon.Point(xs[i], ys[j])

            if is_point_in_poly(p, lower_triangle_points):
                phi_r[i, j] = -1

            if is_point_in_poly(p, upper_triangle_points):
                phi_r[i, j] = +1

    plot_heatmap(xs, ys, phi_r.transpose(), 'phi_r.png', 'x $\AA$', 'y $\AA$', show_nb=True)

    return phi_r


def make_SoD_triangles_from_gaussians_phi_r(xs, ys):
    xn = len(xs)
    yn = len(ys)

    inner_tri_rs = get_inner_triangle_positions()
    upper_tri_rs = get_upper_triangle_positions()
    lower_tri_rs = get_lower_triangle_positions()

    print(Params.lower_triangle_weighting, Params.upper_triangle_weighting, Params.small_triangle_weighting)

    phi_r = np.zeros((xn, yn), dtype=float)

    for i in range(xn):
        for j in range(yn):
            for r in lower_tri_rs:
                x = xs[i] - r[0]
                y = ys[j] - r[1]
                phi_r[i, j] += Params.lower_triangle_weighting * unnormalized_gaussian(x, y, Params.lower_triangle_alpha)

            for r in upper_tri_rs:
                x = xs[i] - r[0]
                y = ys[j] - r[1]
                phi_r[i, j] += Params.upper_triangle_weighting * unnormalized_gaussian(x, y, Params.upper_triangle_alpha)

            for r in inner_tri_rs:
                x = xs[i] - r[0]
                y = ys[j] - r[1]
                phi_r[i, j] += Params.small_triangle_weighting * unnormalized_gaussian(x, y, Params.small_triangle_alpha)

    return phi_r


def make_triple_SoD_discrete_triangles(xs, ys):
    xn = len(xs)
    yn = len(ys)

    inner_tri_rs = get_inner_triangle_positions()
    upper_tri_rs = get_upper_triangle_positions()
    lower_tri_rs = get_lower_triangle_positions()

    f = Params.basis_factor

    for i in range(len(inner_tri_rs)):
        inner_tri_rs[i] *= f

    for i in range(len(upper_tri_rs)):
        upper_tri_rs[i] *= f

    for i in range(len(lower_tri_rs)):
        lower_tri_rs[i] *= f

    from point_in_polygon import Point
    from point_in_polygon import is_point_in_poly

    inner_tri_ps = []
    for r in inner_tri_rs:
        inner_tri_ps.append(Point(r[0], r[1]))
    inner_tri_ps.append(inner_tri_ps[0])

    upper_tri_ps = []
    for r in upper_tri_rs:
        upper_tri_ps.append(Point(r[0], r[1]))
    upper_tri_ps.append(upper_tri_ps[0])

    lower_tri_ps = []
    for r in lower_tri_rs:
        lower_tri_ps.append(Point(r[0]*3, r[1]*3))
    lower_tri_ps.append(lower_tri_ps[0])

    phi_r = np.zeros((xn, yn), dtype=float)

    for i in range(xn):
        for j in range(yn):
            p = Point(xs[i], ys[j])
            if is_point_in_poly(p, lower_tri_ps):
                phi_r[i, j] = Params.lower_triangle_weighting

            if is_point_in_poly(p, upper_tri_ps):
                phi_r[i, j] = Params.upper_triangle_weighting

            if is_point_in_poly(p, inner_tri_ps):
                phi_r[i, j] = Params.small_triangle_weighting

    return phi_r


def make_upper_SoD_discrete_triangles(xs, ys):
    xn = len(xs)
    yn = len(ys)

    inner_tri_rs = get_inner_triangle_positions()
    upper_tri_rs = get_upper_triangle_positions()

    f = Params.basis_factor

    for i in range(len(inner_tri_rs)):
        inner_tri_rs[i] *= f

    for i in range(len(upper_tri_rs)):
        upper_tri_rs[i] *= f

    from point_in_polygon import Point
    from point_in_polygon import is_point_in_poly

    inner_tri_ps = []
    for r in inner_tri_rs:
        inner_tri_ps.append(Point(r[0], r[1]))
    inner_tri_ps.append(inner_tri_ps[0])

    upper_tri_ps = []
    for r in upper_tri_rs:
        upper_tri_ps.append(Point(r[0], r[1]))
    upper_tri_ps.append(upper_tri_ps[0])

    phi_r = np.zeros((xn, yn), dtype=float)

    for i in range(xn):
        for j in range(yn):
            p = Point(xs[i], ys[j])

            if is_point_in_poly(p, upper_tri_ps):
                phi_r[i, j] = Params.upper_triangle_weighting

            if is_point_in_poly(p, inner_tri_ps):
                phi_r[i, j] = Params.small_triangle_weighting

    return phi_r


def get_inner_triangle_positions():  # dark blue
    rs = []

    a1 = Params.a_1
    a2 = Params.a_2

    rs.append(-a1)
    rs.append(a1 + a2)
    rs.append(-a2)

    return rs


def get_upper_triangle_positions():  # orange
    rs = []

    a1 = Params.a_1
    a2 = Params.a_2

    rs.append(2*a2)
    rs.append(2*a1)
    rs.append(-2*a1 - 2*a2)

    return rs


def get_lower_triangle_positions():  # green
    rs = []

    a1 = Params.a_1
    a2 = Params.a_2

    rs.append(2*a1 + 2*a2)
    rs.append(-2*a2)
    rs.append(-2*a1)

    return rs


def make_nine_gausses_phi_r(xs, ys):
    xn = len(xs)
    yn = len(ys)

    a1 = Params.a_1
    a2 = Params.a_2

    gauss_positions = []
    for i in range(0, 3):
        for j in range(0, 3):
            gauss_positions.append(i*a1 + j*a2)

    phi_r = np.zeros((xn, yn), dtype=float)

    for i in range(xn):
        for j in range(yn):
            for p in gauss_positions:
                x = xs[i] - p[0]
                y = ys[j] - p[1]
                phi_r[i, j] += unnormalized_gaussian(x, y, Params.alpha_gauss)

    return phi_r


def make_single_central_gaussian_phi_r(xs, ys):
    xn = len(xs)
    yn = len(ys)

    phi_r = np.zeros((xn, yn), dtype=float)

    for i in range(xn):
        for j in range(yn):
            phi_r[i, j] += unnormalized_gaussian(xs[i], ys[j], Params.alpha_gauss)

    return phi_r


def make_fft_inspired_phi_r(xs, ys):
    xn = len(xs)
    yn = len(ys)

    phi_r = np.zeros((xn, yn), dtype=float)

    for i in range(xn):
        for j in range(yn):
            x = xs[i]
            y = ys[j]

            phi_r[i, j] += three_peak_ricker_wavelet(x, y, 4.5)

            # phi_r[i, j] += modified_ricker_wavelet(x, y, 12)

            r = np.sqrt(x**2 + y**2)
            # if s=4.5, r>11
            # if s=5, r>13
            # if r > 11:
            #     theta = np.arctan(x / y) + np.pi/6
            #     phi_r[i, j] *= (0.5 + 0.7*np.cos(3*theta) ** 2)

            # theta = np.arctan(x / y) + np.pi / 6
            # phi_r[i, j] *= (0.5 + 0.7 * np.cos(3 * theta) ** 2)

    return phi_r


def get_Nb_atom_possitions():
    xs = []
    ys = []
    # Central Nb
    xs.append(0)
    ys.append(0)

    # First hex
    xs.append(-Params.a_1[0])
    ys.append(-Params.a_1[1])

    xs.append(Params.a_1[0])
    ys.append(Params.a_1[1])

    xs.append(-Params.a_2[0])
    ys.append(-Params.a_2[1])

    xs.append(Params.a_2[0])
    ys.append(Params.a_2[1])

    xs.append(- Params.a_1[0] - Params.a_2[0])
    ys.append(- Params.a_1[1] - Params.a_2[1])

    xs.append(Params.a_1[0] + Params.a_2[0])
    ys.append(Params.a_1[1] + Params.a_2[1])

    # Second hex
    # xs.append(2 * Params.a_1[0] + Params.a_2[0])
    # ys.append(2 * Params.a_1[1] + Params.a_2[1])

    xs.append(Params.a_1[0] + 2 * Params.a_2[0])
    ys.append(Params.a_1[1] + 2 * Params.a_2[1])

    # xs.append(-Params.a_1[0] + Params.a_2[0])
    # ys.append(-Params.a_1[1] + Params.a_2[1])

    xs.append(Params.a_1[0] - Params.a_2[0])
    ys.append(Params.a_1[1] - Params.a_2[1])

    xs.append(-2 * Params.a_1[0] - Params.a_2[0])
    ys.append(-2 * Params.a_1[1] - Params.a_2[1])

    # xs.append(-Params.a_1[0] - 2 * Params.a_2[0])
    # ys.append(-Params.a_1[1] - 2 * Params.a_2[1])

    return xs, ys

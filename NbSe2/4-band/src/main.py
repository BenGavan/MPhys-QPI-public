from scipy.fftpack import fftshift
import time
from phi import *

# TODO: Decouple data generation and plotting
# TODO: https://eigen.tuxfamily.org/index.php?title=Main_Page

'''
Notes:
 - All matrices and vectors should conform to the standard convension where <psi| = (...)
   - when introducing a new matrix, if not automatically in the format, change as soon as defined.
'''

def generate_axes(xn, yn, x_length, y_length):
    '''
    Generates a pair of axes of a specified length and resolution
    :param xn: # pixes along x axis
    :param yn: # pixes along y axis
    :param x_length:
    :param y_length:
    :return: 2 arrays - one for each axes
    '''
    xs = np.linspace(-x_length / 2, x_length / 2, xn)
    ys = np.linspace(-y_length / 2, y_length / 2, yn)

    return xs, ys


def generate_real_axis(xn, yn, x_length, y_length):
    xs = np.linspace(-x_length / 2, x_length / 2, xn)
    ys = np.linspace(-y_length / 2, y_length / 2, yn)

    return xs, ys


def make_4_band_qpi(scattering, kxn, kyn, kx_length, ky_length, phi_k):
    # from greens_4_band import generate_4_band_bare_greens
    from greens_4_band import generate_4_band_bare_greens_eigenstate_weighting
    from qpi import qpi

    kxs, kys = generate_axes(kxn, kyn, kx_length, ky_length)

    g_0_k = generate_4_band_bare_greens_eigenstate_weighting(kxs, kys, phi_k=phi_k)

    # trace of bg
    G_0_k_trace = np.zeros((kxn, kyn), dtype=float)
    for i in range(kxn):
        for j in range(kyn):
            G_0_k_trace[i, j] = abs(g_0_k[i, j].trace())

    g_0_k = g_0_k * 80 / abs(np.max(G_0_k_trace))

    for i in range(kxn):
        for j in range(kyn):
            G_0_k_trace[i, j] = abs(g_0_k[i, j].trace())


    plot_heatmap(kxs, kys, abs(G_0_k_trace),
                 filename=get_bgtrace_4_band_plot_filepath(kxn=kxn, kx_length=kx_length,
                                                           version_number=get_version_string()),
                 x_label='$k_x$ $(\AA^{-1})$',
                 y_label='$k_y$ $(\AA^{-1})$', show_bz=True, show_4_band_fermi=True)

    del G_0_k_trace

    # take real before abs to guarantee only taking the imaginary part of the Green's function
    rho_q = qpi(scattering, g_0_k, kxs, kys).real
    rho_q_abs = np.abs(rho_q)
    # rho_q_abs = rho_q
    # rho_q_abs = np.abs(qpi(scattering, g_0_k, kxs, kys))

    if RunParams.WRITE_4_BAND_RHO_Q:
        print('*** oops... WRITE_4_BAND_RHO_Q has not been implemented')

    filename = get_qpi_4_band_filepath(scattering, kxn=kxn, kx_length=kx_length,
                                         version_number=get_version_string())

    plot_heatmap(kxs, kys, rho_q_abs, filename=filename, x_label='$q_x$ $(\AA^{-1})$',
                 y_label='$q_y$ $(\AA^{-1})$', show_bz=True, colormap='hot')

    # filename = get_qpi_4_band_filepath(scattering, kxn=kxn, kx_length=kx_length,
    #                                    version_number=get_version_string()) + '.pdf'
    #
    # from custom_colormap import get_continuous_cmap
    # hex_list = ['#ffffff', '#ffffff', '#f50fcb']
    # float_list = [0, 0.1, 1]
    #
    # plot_heatmap(kxs, kys, rho_q_abs, filename=filename, x_label='$q_x$ $(\AA^{-1})$',
    #              y_label='$q_y$ $(\AA^{-1})$', show_bz=True, colormap=get_continuous_cmap(hex_list, float_list=float_list))


def make_phi_r(xs, ys):
    '''
    wrapper of whatever function is being used as phi(r)
    :param xs:
    :param ys:
    :return: phi(r)
    '''
    # return np.zeros((len(xs), len(ys)), dtype=float)
    return make_single_central_gaussian_phi_r(xs, ys)
    # return make_triple_SoD_discrete_triangles(xs, ys)
    # return make_SoD_triangles_from_gaussians_phi_r(xs, ys)


def make_phi_k(kx_length, ky_length, kxn, kyn):
    dkx = kx_length / kxn
    dky = ky_length / kyn

    # Generate Real Axes (from k-space) (the resolution of k-space = 1/range in real space)
    xn = kxn
    yn = kyn

    # return np.ones((kxn, kyn), dtype=float)

    x_length = 2 * np.pi / dkx  # a/4 angstrom
    y_length = 2 * np.pi / dky  # a/4 angstrom

    xs, ys = generate_axes(xn, yn, x_length=x_length, y_length=y_length)

    phi_r = make_phi_r(xs, ys)

    if RunParams.PLOT_PHI_R:
        filepath = get_phi_r_plot_filepath(xn, x_length, version_number=get_version_string())
        plot_heatmap(xs, ys, phi_r, filepath, 'x $\AA$', 'y $\AA$')

    phi_k = fftshift(fftransform(phi_r))
    # kxs, kys = generate_axes(kxn, kyn, kx_length, ky_length)
    # phi_k = np.zeros((kxn, kyn), dtype=float)
    # return np.ones((kxn, kyn), dtype=float)

    # K = Params.K
    # M = Params.M
    #
    # cs = [K, (K[0], -K[1]), -2 * K + 2 * M, (-K[0], -K[1]), (-K[0], K[1]), 2 * K - 2 * M, K]
    # # select k-pockets
    # for i in range(kxn):
    #     for j in range(kyn):
    #         for c in cs:
    #             kx = kxs[i] - c[0]
    #             ky = kys[j] - c[1]
    #             k = np.sqrt(kx ** 2 + ky ** 2)
    #             if k < 0.54:
    #                 phi_k[i, j] = 1

    # for i in range(kxn):
    #     for j in range(kyn):
    #         kx = kxs[i]
    #         ky = kys[j]
    #         k = np.sqrt(kx ** 2 + ky ** 2)
    #         if k < 0.6:
    #             phi_k[i, j] = 1

    return phi_k


def run():
    start_time = time.time()
    print(get_version_string())
    kxn = Params.kxn
    kyn = Params.kyn

    kx_length = 4 * np.pi / Params.a_0  # \AA^-1
    ky_length = 4 * np.pi / Params.a_0  # \AA^-1

    kxs, kys = generate_axes(kxn, kyn, kx_length, ky_length)

    dkx = kx_length / kxn
    xn = kxn
    x_length = 2 * np.pi / dkx

    scattering = Scattering.Scalar_and_Spin_orbit_4_bands

    if RunParams.PLOT_PHI_R:
        xs_sub, ys_sub = generate_axes(Params.xn_sub, Params.xn_sub, Params.x_length_sub, Params.x_length_sub)

        phi_r_sub = make_phi_r(xs_sub, ys_sub)

        plot_heatmap(xs_sub, ys_sub, (phi_r_sub),
                     filename=get_phi_r_sub_plot_filepath(xn=Params.xn_sub, x_length=Params.x_length_sub,
                                                          version_number=get_version_string()),
                     x_label='$x$ $(\AA)$', y_label='$y$ $(\AA)$', show_bz=False, show_nb=True, show_se=True)

        plot_heatmap(xs_sub, ys_sub, abs(phi_r_sub),
                     filename=get_phi_r_plot_filepath(xn=xn, x_length=x_length, version_number=get_version_string()),
                     x_label='$x$ $(\AA)$', y_label='$y$ $(\AA)$', show_bz=False)

    print('phi(k)')
    phi_k = make_phi_k(kx_length, ky_length, kxn, kyn)
    # kxn, kyn, kx_length, ky_length, phi_k = read_phi_k_data('temp/data/phi_k_triple_SoD_basis-factor={}_lower={}_small={}_upper={}.dat'.format(Params.basis_factor, Params.lower_triangle_weighting, Params.small_triangle_weighting, Params.upper_triangle_weighting))
    # kxs, kys = generate_axes(kxn, kyn, kx_length, ky_length)

    # plot phi_k
    if RunParams.PLOT_PHI_K:
        plot_heatmap(kxs, kys, abs(phi_k),
                     filename=get_phi_k_plot_filepath(kxn=kxn, kx_length=kx_length,
                                                      version_number=get_version_string()),
                     x_label='$k_x$ $(\AA^{-1})$', y_label='$k_y$ $(\AA^{-1})$', show_bz=True)


    make_4_band_qpi(scattering, kxn, kyn, kx_length, ky_length, phi_k=phi_k)

    # plot_combined_18_band_and_2_band(scattering_type=scattering, kxn=kxn, kx_length=kx_length,
    #                                  version_number=get_version_string())
    plot_combined_4_band(scattering_type=scattering, kxn=kxn, kx_length=kx_length, version_number=get_version_string())
    # plot_combined_18_band(scattering_type=scattering, kxn=kxn, kx_length=kx_length, version_number=get_version_string())

    # plot_combined_18_band_split_into_9_2by2(version_string=get_version_string())

    time_end = time.time()
    dt = time_end - start_time
    print('time taken: {:.6f} seconds'.format(dt))


def plot_phi_r_from_file():
    xn, yn, x_length, y_length, phi_r = read_phi_k_data('phi_r_cdw_small.dat')
    xs, ys = generate_axes(xn, yn, x_length, y_length)

    plot_heatmap(xs, ys, np.abs(phi_r), filename='phi_r_test_abs.png', x_label='$x$', y_label='$y$',
                 show_bz=False, show_nb=True, show_se=True)

    plot_heatmap(xs, ys, phi_r.real, filename='phi_r_test_real.png', x_label='$x$', y_label='$y$',
                 show_bz=False, show_nb=True, show_se=True)

    plot_heatmap(xs, ys, phi_r.imag, filename='phi_r_test_imag.png', x_label='$x$', y_label='$y$',
                 show_bz=False, show_nb=True, show_se=True)

    dkx = 2*np.pi / x_length
    dky = 2*np.pi / y_length
    kxn = xn
    kyn = yn
    kx_length = dkx * kxn
    ky_length = dky * kyn

    kxs, kys = generate_axes(kxn, kyn, kx_length, ky_length)

    phi_k_from_phi_r = fftshift(fftransform(phi_r))
    phi_k_from_abs_phi_r = fftshift(fftransform(np.abs(phi_r)))
    phi_k_from_real_phi_r = fftshift(fftransform(phi_r.real))
    phi_k_from_imag_phi_r = fftshift(fftransform(phi_r.imag))

    plot_heatmap(kxs, kys, np.abs(phi_k_from_phi_r), filename='phi_k_test_abs_from_phi_r.png', x_label='$k_x$', y_label='$k_y$',
                 show_bz=False, show_nb=False, show_se=False)
    plot_heatmap(kxs, kys, np.abs(phi_k_from_abs_phi_r), filename='phi_k_test_abs_from_abs_phi_r.png', x_label='$k_x$', y_label='$k_y$',
                 show_bz=False, show_nb=False, show_se=False)
    plot_heatmap(kxs, kys, np.abs(phi_k_from_real_phi_r), filename='phi_k_test_abs_from_real_phi_r.png', x_label='$k_x$', y_label='$k_y$',
                 show_bz=False, show_nb=False, show_se=False)
    plot_heatmap(kxs, kys, np.abs(phi_k_from_imag_phi_r), filename='phi_k_test_abs_from_imag_phi_r.png', x_label='$k_x$', y_label='$k_y$',
                 show_bz=False, show_nb=False, show_se=False)


def plot_phi_k_from_file():
    kxn, kyn, kx_length, ky_length, phi_k = read_phi_k_data('temp/phi_k_SoD_1.dat')
    kxs, kys = generate_axes(kxn, kyn, kx_length, ky_length)

    plot_heatmap(kxs, kys, np.abs(phi_k), filename='temp/plots/phi_k_abs_from_dft.png', x_label='$x$', y_label='$y$',
                 show_bz=True, show_nb=False, show_se=False)


def write_phi_k_to_file():
    kxn = Params.kxn
    kyn = Params.kyn

    kx_length = 4 * np.pi / Params.a_0  # \AA^-1
    ky_length = 4 * np.pi / Params.a_0  # \AA^-1

    kxs, kys = generate_axes(kxn, kyn, kx_length, ky_length)

    xn = kxn
    yn = kyn
    x_length = 40
    y_length = x_length

    phi_k = make_cdw_third_k_phi_k(kxs, kys)
    plot_heatmap(kxs, kys, np.abs(phi_k), filename='phi_k_test.png', x_label='$k_x$', y_label='$k_y$',
                 show_bz=True)
    write_phi_r_data('phi_k_cdw.dat', kxn, kyn, kx_length, ky_length, xn, yn, x_length, y_length, phi_k)


def write_phi_r_to_file():
    kx_length = 4 * np.pi / Params.a_0  # \AA^-1
    ky_length = 4 * np.pi / Params.a_0  # \AA^-1

    xn = 300
    yn = 300

    x_length = 40
    y_length = 40

    xs, ys = generate_axes(xn, yn, x_length, y_length)

    phi_r = make_phi_r(xs, ys)

    write_phi_r_data('temp/phi_r_triple_SoD_basis-factor={}_lower={}_small={}_upper={}.dat'
                     .format(Params.basis_factor, Params.lower_triangle_weighting, Params.small_triangle_weighting, Params.upper_triangle_weighting),
                     xn, yn, x_length, y_length, 300, 300, kx_length, ky_length, phi_r)

    plot_heatmap(xs, ys, phi_r, 'phi_r.png', 'x', 'y', show_se=True, show_nb=True)


def init():
    Params.basis_factor = 1

    Params.small_triangle_weighting = 2
    Params.upper_triangle_weighting = 1
    Params.lower_triangle_weighting = -1

    Params.small_triangle_alpha = 7
    Params.upper_triangle_alpha = 6
    Params.lower_triangle_alpha = 1

    Params.alpha_gauss = 5

    Params.kxn = 300
    Params.kyn = 300

    Params.eigenbasis_phi = [1, 1, 1, 1]

    set_rho_q_run_params(read=False, write=False)
    RunParams.USE_CACHE = False


def main():
    init()
    append_run_to_history_file(get_version_string())

    print('a = {}'.format(Params.alpha_gauss))
    # print('basis factor = {}'.format(Params.basis_factor))
    # for i in range(1, 10, 1):
    #     Params.alpha_gauss = i
    #     print('a = {}'.format(Params.alpha_gauss))
    run()


# TODO: By applying a threshold to |Tr(G_0(k))|, it reveals asymmetry in the intensities of the Gamma pocket
#  -> maybe only select the k-pockets and see what that gives

def get_version_string():
    return '_{}_Scalar+SO_4_band_just-diag_all-eigvs_eig=3,4_central-gauss_alpha={}_dEf={}_kxn={}_'\
        .format(get_version_number(), Params.alpha_gauss, Params.delta_E_f, Params.kxn)
    # return '_{}_Scalar+SO_18_bands_just-diag_c={}___dEf={}_kxn={}___small={}_upper={}_lower={}'.format(get_version_number(), Params.c, Params.delta_E_f,
    #                                                                                                                                                                Params.kxn,
    #                                                                                                                                                                Params.basis_factor, Params.small_triangle_weighting, Params.upper_triangle_weighting, Params.lower_triangle_weighting)


def get_version_number():
    return '1.94.00'

# TODO: look at experimental data to motive SoD better
# TODO:   - should the lower (negative) triangle be larger than we currently have it?
# TODO:   - reason: the trigonal clusters at localised to ~3*3 atoms, but the dark regions extend much further


if __name__ == '__main__':
    main()

import matplotlib.pyplot as plt
from PIL import Image
from params import *
from utils import *
from files import *

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
from sys import argv


def plot_surface(xs, ys, zss):
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    import matplotlib.pyplot as plt
    import numpy as np

    X, Y = np.meshgrid(xs, ys)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    for zs in zss:
        surf = ax.plot_surface(X, Y, zs, rstride=1, cstride=1, linewidth=0, antialiased=False)
        # cmap=cm.coolwarm,

    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


def plot_heatmap(xs_top, ys_top, zs, filename, x_label, y_label, show_bz=False, show_nb=False, show_se=False):
    fig, ax = plt.subplots()

    c = ax.pcolormesh(xs_top, ys_top, zs, cmap='hot')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # ax.scatter([-0.39, -0.38, -0.37, -0.36, -0.35], [-0.34, -0.34, -0.34,  -0.34, -0.34], marker='x')
    # ax.scatter([-0.39], [-0.34], marker='x')

    if show_bz:
        K = Params.K
        M = Params.M

        cs = [Params.K, (K[0], -K[1]), -2 * K + 2 * M, (-K[0], -K[1]), (-K[0], K[1]), 2 * K - 2 * M, Params.K]
        xs_brillouin = np.array([])
        ys_brillouin = np.array([])

        for coord in cs:
            xs_brillouin = np.append(xs_brillouin, coord[0])
            ys_brillouin = np.append(ys_brillouin, coord[1])

        ax.plot(xs_brillouin, ys_brillouin, color='gray', linestyle='dashed')  # , zorder=1

        xs_brillouin = np.array([])
        ys_brillouin = np.array([])

        for coord in cs:
            xs_brillouin = np.append(xs_brillouin, coord[0]*2/3)
            ys_brillouin = np.append(ys_brillouin, coord[1]*2/3)

        ax.plot(xs_brillouin, ys_brillouin, color='gray', linestyle='dashed')  # , zorder=1

        xs_brillouin = np.array([])
        ys_brillouin = np.array([])

        for coord in cs:
            xs_brillouin = np.append(xs_brillouin, coord[0] * 1 / 3)
            ys_brillouin = np.append(ys_brillouin, coord[1] * 1 / 3)

        ax.plot(xs_brillouin, ys_brillouin, color='gray', linestyle='dashed')  # , zorder=1

    if show_nb:
        xs_top = []
        ys_top = []

        # First hex
        xs_top.append(-Params.a_1[0])
        ys_top.append(-Params.a_1[1])

        xs_top.append(Params.a_1[0])
        ys_top.append(Params.a_1[1])

        xs_top.append(-Params.a_2[0])
        ys_top.append(-Params.a_2[1])

        xs_top.append(Params.a_2[0])
        ys_top.append(Params.a_2[1])

        xs_top.append(- Params.a_1[0] - Params.a_2[0])
        ys_top.append(- Params.a_1[1] - Params.a_2[1])

        xs_top.append(Params.a_1[0] + Params.a_2[0])
        ys_top.append(Params.a_1[1] + Params.a_2[1])

        # Second hex
        xs_top.append(2 * Params.a_1[0] + Params.a_2[0])
        ys_top.append(2 * Params.a_1[1] + Params.a_2[1])

        xs_top.append(Params.a_1[0] + 2 * Params.a_2[0])
        ys_top.append(Params.a_1[1] + 2 * Params.a_2[1])

        xs_top.append(-Params.a_1[0] + Params.a_2[0])
        ys_top.append(-Params.a_1[1] + Params.a_2[1])

        xs_top.append(Params.a_1[0] - Params.a_2[0])
        ys_top.append(Params.a_1[1] - Params.a_2[1])

        xs_top.append(-2 * Params.a_1[0] - Params.a_2[0])
        ys_top.append(-2 * Params.a_1[1] - Params.a_2[1])

        xs_top.append(-Params.a_1[0] - 2 * Params.a_2[0])
        ys_top.append(-Params.a_1[1] - 2 * Params.a_2[1])

        ax.scatter(xs_top, ys_top)

    if show_se:
        xs_top = []
        ys_top = []

        xs_bottom = []
        ys_bottom = []

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

        # Top first
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

        ax.scatter(xs_top, ys_top, color='#38cf5b') # 88c999
        ax.scatter(xs_bottom, ys_bottom, color='#a7c9af')

    # K = Params.K
    # M = Params.M

    # A = Params.K
    # B = (K[0], -K[1])
    # ax.plot([A[0] * 1 / 3, B[0] * 1 / 3], [A[1] * 1 / 3, B[1] * 1 / 3])

    fig.colorbar(c, ax=ax, label='')  # TODO: change/try a logarithmic color scale/bar .
    fig.gca().set_aspect('equal', adjustable='box')
    fig.tight_layout()
    fig.savefig(filename, rasterized=True, bbox_inches='tight')
    # fig.show()
    plt.close(fig)
    print('heat map plot: \n {} \n'.format(filename))
    copy2clip(filename)


def plot_scatter_heatmap(xs, ys, zs, filename, x_label, y_label, show_bz=False, show_nb=False, show_se=False):
    """

    :param xs: flat x-coords
    :param ys: flat y-coords
    :param zs: flat z-values
    :param filename:
    :param x_label:
    :param y_label:
    :param show_bz:
    :param show_nb:
    :param show_se:
    :return: None
    """

    fig, ax = plt.subplots()

    # New scatter plot
    c = ax.scatter(xs, ys, c=zs, s=500, cmap='hot', marker='s')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if show_bz:
        K = Params.K
        M = Params.M

        cs = [Params.K, (K[0], -K[1]), -2 * K + 2 * M, (-K[0], -K[1]), (-K[0], K[1]), 2 * K - 2 * M, Params.K]
        xs_brillouin = np.array([])
        ys_brillouin = np.array([])

        for coord in cs:
            xs_brillouin = np.append(xs_brillouin, coord[0])
            ys_brillouin = np.append(ys_brillouin, coord[1])

        ax.plot(xs_brillouin, ys_brillouin, color='gray', linestyle='dashed')  # , zorder=1

        xs_brillouin = np.array([])
        ys_brillouin = np.array([])

        for coord in cs:
            xs_brillouin = np.append(xs_brillouin, coord[0] * 2/3)
            ys_brillouin = np.append(ys_brillouin, coord[1] * 2/3)

        ax.plot(xs_brillouin, ys_brillouin, color='gray', linestyle='dashed')  # , zorder=1

        xs_brillouin = np.array([])
        ys_brillouin = np.array([])

        for coord in cs:
            xs_brillouin = np.append(xs_brillouin, coord[0] * 1 / 3)
            ys_brillouin = np.append(ys_brillouin, coord[1] * 1 / 3)

        ax.plot(xs_brillouin, ys_brillouin, color='gray', linestyle='dashed')  # , zorder=1

    if show_nb:
        xs_top = []
        ys_top = []

        # First hex
        xs_top.append(-Params.a_1[0])
        ys_top.append(-Params.a_1[1])

        xs_top.append(Params.a_1[0])
        ys_top.append(Params.a_1[1])

        xs_top.append(-Params.a_2[0])
        ys_top.append(-Params.a_2[1])

        xs_top.append(Params.a_2[0])
        ys_top.append(Params.a_2[1])

        xs_top.append(- Params.a_1[0] - Params.a_2[0])
        ys_top.append(- Params.a_1[1] - Params.a_2[1])

        xs_top.append(Params.a_1[0] + Params.a_2[0])
        ys_top.append(Params.a_1[1] + Params.a_2[1])

        # Second hex
        xs_top.append(2 * Params.a_1[0] + Params.a_2[0])
        ys_top.append(2 * Params.a_1[1] + Params.a_2[1])

        xs_top.append(Params.a_1[0] + 2 * Params.a_2[0])
        ys_top.append(Params.a_1[1] + 2 * Params.a_2[1])

        xs_top.append(-Params.a_1[0] + Params.a_2[0])
        ys_top.append(-Params.a_1[1] + Params.a_2[1])

        xs_top.append(Params.a_1[0] - Params.a_2[0])
        ys_top.append(Params.a_1[1] - Params.a_2[1])

        xs_top.append(-2 * Params.a_1[0] - Params.a_2[0])
        ys_top.append(-2 * Params.a_1[1] - Params.a_2[1])

        xs_top.append(-Params.a_1[0] - 2 * Params.a_2[0])
        ys_top.append(-Params.a_1[1] - 2 * Params.a_2[1])

        ax.scatter(xs_top, ys_top)

    if show_se:
        xs_top = []
        ys_top = []

        xs_bottom = []
        ys_bottom = []

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

        # Top first
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

        ax.scatter(xs_top, ys_top, color='#38cf5b')  # 88c999
        ax.scatter(xs_bottom, ys_bottom, color='#a7c9af')

    fig.colorbar(c, ax=ax, label='')  # TODO: change/try a logarithmic color scale/bar .
    fig.gca().set_aspect('equal', adjustable='box')
    fig.tight_layout()
    fig.savefig(filename, rasterized=True, bbox_inches='tight', dpi=300)
    # fig.show()
    plt.close(fig)
    print('scatter heatmap plot: \n {} \n'.format(filename))




def plot_points(points, ms, cs, filepath, show_bz=False):
    fig, ax = plt.subplots()
    # ax.scatter(xs_original, ys_original)

    if show_bz:
        K = Params.K
        M = Params.M

        bz_coords = [Params.K, (K[0], -K[1]), -2 * K + 2 * M, (-K[0], -K[1]), (-K[0], K[1]), 2 * K - 2 * M, Params.K]
        xs_brillouin = np.array([])
        ys_brillouin = np.array([])

        for coord in bz_coords:
            xs_brillouin = np.append(xs_brillouin, coord[0])
            ys_brillouin = np.append(ys_brillouin, coord[1])

        ax.plot(xs_brillouin, ys_brillouin, color='gray', linestyle='dashed')  # , zorder=1

        xs_brillouin = np.array([])
        ys_brillouin = np.array([])

        for coord in bz_coords:
            xs_brillouin = np.append(xs_brillouin, coord[0] * 2 / 3)
            ys_brillouin = np.append(ys_brillouin, coord[1] * 2 / 3)

        ax.plot(xs_brillouin, ys_brillouin, color='gray', linestyle='dashed')  # , zorder=1

        xs_brillouin = np.array([])
        ys_brillouin = np.array([])

        for coord in bz_coords:
            xs_brillouin = np.append(xs_brillouin, coord[0] * 1 / 3)
            ys_brillouin = np.append(ys_brillouin, coord[1] * 1 / 3)

        ax.plot(xs_brillouin, ys_brillouin, color='gray', linestyle='dashed')  # , zorder=1

    xs = []
    ys = []

    xs_original = []
    ys_original = []

    for p in points:
        xs.append(p.current_x)
        ys.append(p.current_y)

        xs_original.append(p.original_x)
        ys_original.append(p.original_y)


    ax.scatter(xs, ys, marker='x')

    for m, c in zip(ms, cs):
        ax.plot([m * min(ys) + c, m * max(ys) + c], [min(ys), max(ys)], color='red')

    fig.gca().set_aspect('equal', adjustable='box')
    fig.tight_layout()
    fig.savefig(filepath, rasterized=True, bbox_inches='tight')
    # fig.show()
    plt.close(fig)
    print('Points plot:\n{}\n'.format(filepath))

# def plot_heatmap(xs_top, ys_top, zs, filename, x_label, y_label, show_bz=False, show_nb=False, show_se=False):
#     fig, ax = plt.subplots()
#
#     c = ax.pcolormesh(xs_top, ys_top, zs, cmap='hot')
#     ax.set_xlabel(x_label)
#     ax.set_ylabel(y_label)
#
#     if show_bz:
#         K = Params.K
#         M = Params.M
#
#         cs = [Params.K, (K[0], -K[1]), -2 * K + 2 * M, (-K[0], -K[1]), (-K[0], K[1]), 2 * K - 2 * M, Params.K]
#         xs_brillouin = np.array([])
#         ys_brillouin = np.array([])
#
#         for coord in cs:
#             xs_brillouin = np.append(xs_brillouin, coord[0])
#             ys_brillouin = np.append(ys_brillouin, coord[1])
#
#         ax.plot(xs_brillouin, ys_brillouin, color='gray', linestyle='dashed')  # , zorder=1
#
#         xs_brillouin = np.array([])
#         ys_brillouin = np.array([])
#
#         for coord in cs:
#             xs_brillouin = np.append(xs_brillouin, coord[0]/3)
#             ys_brillouin = np.append(ys_brillouin, coord[1]/3)
#
#         ax.plot(xs_brillouin, ys_brillouin, color='gray', linestyle='dashed')  # , zorder=1
#
#     if show_nb:
#         xs_top = []
#         ys_top = []
#
#         # First hex
#         xs_top.append(-Params.a_1[0])
#         ys_top.append(-Params.a_1[1])
#
#         xs_top.append(Params.a_1[0])
#         ys_top.append(Params.a_1[1])
#
#         xs_top.append(-Params.a_2[0])
#         ys_top.append(-Params.a_2[1])
#
#         xs_top.append(Params.a_2[0])
#         ys_top.append(Params.a_2[1])
#
#         xs_top.append(- Params.a_1[0] - Params.a_2[0])
#         ys_top.append(- Params.a_1[1] - Params.a_2[1])
#
#         xs_top.append(Params.a_1[0] + Params.a_2[0])
#         ys_top.append(Params.a_1[1] + Params.a_2[1])
#
#         # Second hex
#         xs_top.append(2 * Params.a_1[0] + Params.a_2[0])
#         ys_top.append(2 * Params.a_1[1] + Params.a_2[1])
#
#         xs_top.append(Params.a_1[0] + 2 * Params.a_2[0])
#         ys_top.append(Params.a_1[1] + 2 * Params.a_2[1])
#
#         xs_top.append(-Params.a_1[0] + Params.a_2[0])
#         ys_top.append(-Params.a_1[1] + Params.a_2[1])
#
#         xs_top.append(Params.a_1[0] - Params.a_2[0])
#         ys_top.append(Params.a_1[1] - Params.a_2[1])
#
#         xs_top.append(-2 * Params.a_1[0] - Params.a_2[0])
#         ys_top.append(-2 * Params.a_1[1] - Params.a_2[1])
#
#         xs_top.append(-Params.a_1[0] - 2 * Params.a_2[0])
#         ys_top.append(-Params.a_1[1] - 2 * Params.a_2[1])
#
#         ax.scatter(xs_top, ys_top)
#
#     if show_se:
#         xs_top = []
#         ys_top = []
#
#         xs_bottom = []
#         ys_bottom = []
#
#         s1 = np.matrix((1 / 3) * Params.a_1 + (2 / 3) * Params.a_2)
#         s2 = np.matrix((-2 / 3) * Params.a_1 - (1 / 3) * Params.a_2)
#         s3 = np.matrix((1 / 3) * Params.a_1 + (-1 / 3) * Params.a_2)
#
#         # Rotate where Se atoms are
#         theta = np.pi / 6
#         theta = 0
#         R_z = np.matrix(((np.cos(theta), -np.sin(theta)),
#                          (np.sin(theta), np.cos(theta))))
#
#         s1 = R_z * s1.transpose()
#         s2 = R_z * s2.transpose()
#         s3 = R_z * s3.transpose()
#
#         # Top first
#         xs_top.append(s1[0])
#         ys_top.append(s1[1])
#
#         xs_top.append(s2[0])
#         ys_top.append(s2[1])
#
#         xs_top.append(s3[0])
#         ys_top.append(s3[1])
#
#         # Bottom first
#         xs_bottom.append(-s1[0])
#         ys_bottom.append(-s1[1])
#
#         xs_bottom.append(-s2[0])
#         ys_bottom.append(-s2[1])
#
#         xs_bottom.append(-s3[0])
#         ys_bottom.append(-s3[1])
#
#         # Bottom second
#         xs_bottom.append(2 * s1[0])
#         ys_bottom.append(2 * s1[1])
#
#         xs_bottom.append(2 * s2[0])
#         ys_bottom.append(2 * s2[1])
#
#         xs_bottom.append(2 * s3[0])
#         ys_bottom.append(2 * s3[1])
#
#         # Top second
#         xs_top.append(-2 * s1[0])
#         ys_top.append(-2 * s1[1])
#
#         xs_top.append(-2 * s2[0])
#         ys_top.append(-2 * s2[1])
#
#         xs_top.append(-2 * s3[0])  # xs.append(-Params.a_1[0] - 2 * Params.a_2[0] + s1[0])
#         ys_top.append(-2 * s3[1])  # ys.append(-Params.a_1[1] - 2 * Params.a_2[1] + s1[1])
#
#         ax.scatter(xs_top, ys_top, color='#38cf5b') # 88c999
#         ax.scatter(xs_bottom, ys_bottom, color='#a7c9af')
#
#
#     fig.colorbar(c, ax=ax, label='')  # TODO: change/try a logarithmic color scale/bar .
#     fig.gca().set_aspect('equal', adjustable='box')
#     fig.tight_layout()
#     fig.savefig(filename, rasterized=True, bbox_inches='tight')
#     # fig.show()
#     plt.close(fig)
#     print('heat map plot: \n {} \n'.format(filename))


def plot_combined(scattering_type, kxn, kx_length):
    print('Combined plot:')

    dkx = kx_length / kxn
    xn = kxn
    x_length = 2 * np.pi / dkx  # a/4 angstrom

    phi_k_plot = get_phi_k_plot_filepath(kxn=kxn, kx_length=kx_length)
    phi_r_plot = get_phi_r_plot_filepath(xn=xn, x_length=x_length)
    phi_r_sub_plot = get_phi_r_sub_plot_filepath(xn=Params.xn_sub, x_length=Params.x_length_sub)
    qpi_plot = get_qpi_filepath(scattering_type=scattering_type, kxn=kxn, kx_length=kx_length)
    greens_plot = get_bgtrace_plot_filepath(kxn=kxn, kx_length=kx_length)

    out_filepath = get_combined_plot_filepath()

    top_left_images = [Image.open(x) for x in [phi_r_sub_plot, phi_k_plot]]

    for i in range(len(top_left_images)):
        width, height = top_left_images[i].size
        ratio = height / width
        top_left_images[i] = top_left_images[i].resize((1000, int(1000 * ratio)))

    bottom_left_images = [Image.open(x) for x in [phi_r_plot, greens_plot]]

    for i in range(len(bottom_left_images)):
        width, height = bottom_left_images[i].size
        ratio = height / width
        bottom_left_images[i] = bottom_left_images[i].resize((1000, int(1000 * ratio)))

    right_image = Image.open(qpi_plot)

    right_image_width, right_image_height =right_image.size
    ratio = right_image_height / right_image_width
    right_image = right_image.resize((1500, int(1500 * ratio)))

    top_left_widths, top_left_heights = zip(*(i.size for i in top_left_images))
    bottom_left_widths, bottom_left_heights = zip(*(i.size for i in bottom_left_images))

    right_width, right_height = right_image.size

    top_left_total_width = sum(top_left_widths)
    top_left_max_height = max(top_left_heights)

    bottom_left_total_width = sum(bottom_left_widths)
    bottom_left_max_height = max(bottom_left_heights)

    left_total_width = max(top_left_total_width, bottom_left_total_width)
    left_total_height = top_left_max_height + bottom_left_max_height

    new_top_left_im = Image.new('RGB', (top_left_total_width, top_left_max_height), (255, 255, 255))
    new_bottom_left_im = Image.new('RGB', (bottom_left_total_width, bottom_left_max_height), (255, 255, 255))

    left_image = Image.new('RGB', (left_total_width, left_total_height), (255, 255, 255))

    x_offset = 0
    for im in top_left_images:
        new_top_left_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    x_offset = 0
    for im in bottom_left_images:
        new_bottom_left_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    left_image.paste(new_top_left_im, (0, 0))
    left_image.paste(new_bottom_left_im, (0, top_left_max_height))

    image = Image.new('RGB', (left_total_width + right_width, max(left_total_height, right_height)), (255, 255, 255))

    image.paste(left_image, (0, 0))
    image.paste(right_image, (left_total_width, 0))

    image.save(out_filepath)
    print(out_filepath)


def plot_combined_H_r_els(m, n):
    print('Combined plot:')

    out_filepath = '../plots/ham/H_r/combined_{}{}.png'.format(m, n)

    top_left_images = [Image.open(x) for x in ['../plots/ham/H_r/h_r_{}{}_real.png'.format(m, n), '../plots/ham/H_r/h_r_{}{}_imag.png'.format(m, n), '../plots/ham/H_r/h_r_{}{}_abs.png'.format(m, n)]]

    for i in range(len(top_left_images)):
        width, height = top_left_images[i].size
        ratio = height / width
        top_left_images[i] = top_left_images[i].resize((1000, int(1000 * ratio)))

    top_left_widths, top_left_heights = zip(*(i.size for i in top_left_images))

    top_left_total_width = sum(top_left_widths)
    top_left_max_height = max(top_left_heights)

    left_total_width = top_left_total_width
    left_total_height = top_left_max_height

    new_top_left_im = Image.new('RGB', (top_left_total_width, top_left_max_height), (255, 255, 255))

    left_image = Image.new('RGB', (left_total_width, left_total_height), (255, 255, 255))

    x_offset = 0
    for im in top_left_images:
        new_top_left_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    left_image.paste(new_top_left_im, (0, 0))

    image = Image.new('RGB', (left_total_width, left_total_height), (255, 255, 255))

    image.paste(left_image, (0, 0))

    image.save(out_filepath)
    print(out_filepath)


def plot_combined_H_k_els(m, n):
    print('Combined plot:')

    out_filepath = '../plots/ham/H_k/combined_H_k_{}{}.png'.format(m, n)

    top_left_images = [Image.open(x) for x in ['../plots/ham/H_k/h_k_{}{}_real.png'.format(m, n), '../plots/ham/H_k/h_k_{}{}_imag.png'.format(m, n), '../plots/ham/H_k/h_k_{}{}_abs.png'.format(m, n)]]

    for i in range(len(top_left_images)):
        width, height = top_left_images[i].size
        ratio = height / width
        top_left_images[i] = top_left_images[i].resize((1000, int(1000 * ratio)))

    top_left_widths, top_left_heights = zip(*(i.size for i in top_left_images))

    top_left_total_width = sum(top_left_widths)
    top_left_max_height = max(top_left_heights)

    left_total_width = top_left_total_width
    left_total_height = top_left_max_height

    new_top_left_im = Image.new('RGB', (top_left_total_width, top_left_max_height), (255, 255, 255))

    left_image = Image.new('RGB', (left_total_width, left_total_height), (255, 255, 255))

    x_offset = 0
    for im in top_left_images:
        new_top_left_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    left_image.paste(new_top_left_im, (0, 0))

    image = Image.new('RGB', (left_total_width, left_total_height), (255, 255, 255))

    image.paste(left_image, (0, 0))

    image.save(out_filepath)
    print(out_filepath)


def plot_combined_images(filepaths, out_filepath):
    print('Combined plot:')

    top_left_images = [Image.open(x) for x in filepaths]

    for i in range(len(top_left_images)):
        width, height = top_left_images[i].size
        ratio = height / width
        top_left_images[i] = top_left_images[i].resize((1000, int(1000 * ratio)))

    top_left_widths, top_left_heights = zip(*(i.size for i in top_left_images))

    top_left_total_width = sum(top_left_widths)
    top_left_max_height = max(top_left_heights)

    left_total_width = top_left_total_width
    left_total_height = top_left_max_height

    new_top_left_im = Image.new('RGB', (top_left_total_width, top_left_max_height), (255, 255, 255))

    left_image = Image.new('RGB', (left_total_width, left_total_height), (255, 255, 255))

    x_offset = 0
    for im in top_left_images:
        new_top_left_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    left_image.paste(new_top_left_im, (0, 0))

    image = Image.new('RGB', (left_total_width, left_total_height), (255, 255, 255))

    image.paste(left_image, (0, 0))

    image.save(out_filepath)
    print(out_filepath)


def plot_hstack_images(filepaths, out_filepath):
    print('H stack plot:')

    top_left_images = [Image.open(x) for x in filepaths]

    for i in range(len(top_left_images)):
        width, height = top_left_images[i].size
        ratio = height / width
        top_left_images[i] = top_left_images[i].resize((1000, int(1000 * ratio)))

    top_left_widths, top_left_heights = zip(*(i.size for i in top_left_images))

    top_left_total_width = sum(top_left_widths)
    top_left_max_height = max(top_left_heights)

    left_total_width = top_left_total_width
    left_total_height = top_left_max_height

    new_top_left_im = Image.new('RGB', (top_left_total_width, top_left_max_height), (255, 255, 255))

    left_image = Image.new('RGB', (left_total_width, left_total_height), (255, 255, 255))

    x_offset = 0
    for im in top_left_images:
        new_top_left_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    left_image.paste(new_top_left_im, (0, 0))

    image = Image.new('RGB', (left_total_width, left_total_height), (255, 255, 255))

    image.paste(left_image, (0, 0))

    image.save(out_filepath)
    print(out_filepath)


def plot_vstack_images(filepaths, out_filepath):
    print('V stack plot:')

    top_left_images = [Image.open(x) for x in filepaths]

    for i in range(len(top_left_images)):
        width, height = top_left_images[i].size
        ratio = height / width
        top_left_images[i] = top_left_images[i].resize((1000, int(1000 * ratio)))

    top_left_widths, top_left_heights = zip(*(i.size for i in top_left_images))

    top_left_total_width = max(top_left_widths)
    top_left_max_height = sum(top_left_heights)

    left_total_width = top_left_total_width
    left_total_height = top_left_max_height

    new_top_left_im = Image.new('RGB', (top_left_total_width, top_left_max_height), (255, 255, 255))

    left_image = Image.new('RGB', (left_total_width, left_total_height), (255, 255, 255))

    y_offset = 0
    for im in top_left_images:
        new_top_left_im.paste(im, (0, y_offset))
        y_offset += im.size[1]

    left_image.paste(new_top_left_im, (0, 0))

    image = Image.new('RGB', (left_total_width, left_total_height), (255, 255, 255))

    image.paste(left_image, (0, 0))

    image.save(out_filepath)
    print(out_filepath)

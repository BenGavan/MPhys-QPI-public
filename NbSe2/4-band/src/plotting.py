import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from params import *
from utils import *
from files import *


def plot_heatmap(xs, ys, zs, filename, x_label, y_label, show_bz=False, show_nb=False, show_se=False, show_4_band_fermi=False, colormap='hot'):
    fig, ax = plt.subplots()

    # translate coordinates into form matplotlib understands
    # {x0, x1, ..., xn} -> {{x00, x01, ..., x0n}, {x10, x11, ..., x1n}, ...,{xm0, xm1, ..., xmn}}
    # {y0, y1, ..., yn} -> {{y00, y01, ..., y0n}, {y10, y11, ..., y1n}, ...,{ym0, ym1, ..., ymn}}
    xn = len(xs)
    yn = len(ys)

    x_matrix = np.zeros((xn, yn))
    y_matrix = np.zeros((xn, yn))

    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]

    for i in range(xn):
        for j in range(yn):
            x_matrix[i, j] = xs[i] - dx / 2
            y_matrix[i, j] = ys[j] - dy / 2

    # plot heatmap
    c = ax.pcolormesh(x_matrix, y_matrix, zs, cmap=colormap)  # TODO: ax.imshow()
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if show_4_band_fermi:
        plot_4_band_fermi_surface_on_axes(ax, xs, ys)

    # centre point (used for Debug)
    # ax.scatter([0], [0], s=0.5)

    # Additional annotations #
    if show_bz:
        K = Params.K
        M = Params.M

        cs = [K, (K[0], -K[1]), -2 * K + 2 * M, (-K[0], -K[1]), (-K[0], K[1]), 2 * K - 2 * M, K]
        xs_brillouin = np.array([])
        ys_brillouin = np.array([])

        for coord in cs:
            xs_brillouin = np.append(xs_brillouin, coord[0])
            ys_brillouin = np.append(ys_brillouin, coord[1])

        ax.plot(xs_brillouin, ys_brillouin, color='gray', linestyle='dashed')  # , zorder=1

        xs_brillouin = np.array([])
        ys_brillouin = np.array([])

        for coord in cs:
            xs_brillouin = np.append(xs_brillouin, coord[0] * 2 / 3)
            ys_brillouin = np.append(ys_brillouin, coord[1] * 2 / 3)

        ax.plot(xs_brillouin, ys_brillouin, color='gray', linestyle='dashed')  # , zorder=1

        xs_brillouin = np.array([])
        ys_brillouin = np.array([])

        for coord in cs:
            xs_brillouin = np.append(xs_brillouin, coord[0] * 1 / 3)
            ys_brillouin = np.append(ys_brillouin, coord[1] * 1 / 3)

        ax.plot(xs_brillouin, ys_brillouin, color='gray', linestyle='dashed')  # , zorder=1

    if show_nb:
        xs_nb = []
        ys_nb = []

        rs_nb = []

        a1 = Params.a_1
        a2 = Params.a_2

        # Central Nb
        # xs_nb.append(0)
        # ys_nb.append(0)
        #
        # # First hex
        # xs_nb.append(-Params.a_1[0])
        # ys_nb.append(-Params.a_1[1])
        #
        # xs_nb.append(Params.a_1[0])
        # ys_nb.append(Params.a_1[1])
        #
        # xs_nb.append(-Params.a_2[0])
        # ys_nb.append(-Params.a_2[1])
        #
        # xs_nb.append(Params.a_2[0])
        # ys_nb.append(Params.a_2[1])
        #
        # xs_nb.append(- Params.a_1[0] - Params.a_2[0])
        # ys_nb.append(- Params.a_1[1] - Params.a_2[1])
        #
        # xs_nb.append(Params.a_1[0] + Params.a_2[0])
        # ys_nb.append(Params.a_1[1] + Params.a_2[1])
        #
        # # Second hex
        # xs_nb.append(2 * Params.a_1[0] + Params.a_2[0])
        # ys_nb.append(2 * Params.a_1[1] + Params.a_2[1])
        #
        # xs_nb.append(Params.a_1[0] + 2 * Params.a_2[0])
        # ys_nb.append(Params.a_1[1] + 2 * Params.a_2[1])
        #
        # xs_nb.append(-Params.a_1[0] + Params.a_2[0])
        # ys_nb.append(-Params.a_1[1] + Params.a_2[1])
        #
        # xs_nb.append(Params.a_1[0] - Params.a_2[0])
        # ys_nb.append(Params.a_1[1] - Params.a_2[1])
        #
        # xs_nb.append(-2 * Params.a_1[0] - Params.a_2[0])
        # ys_nb.append(-2 * Params.a_1[1] - Params.a_2[1])
        #
        # xs_nb.append(-Params.a_1[0] - 2 * Params.a_2[0])
        # ys_nb.append(-Params.a_1[1] - 2 * Params.a_2[1])

        # Third Hex
        # rs_nb.append(2 * a2)
        # rs_nb.append(a1 + 2 * a2)
        # rs_nb.append()

        for i in range(-2, 3):
            for j in range(-2, 3):
                rs_nb.append(i * a1 + j * a2)

        for r in rs_nb:
            xs_nb.append(r[0])
            ys_nb.append(r[1])

        ax.scatter(xs_nb, ys_nb, color='#2685eb')
    #     , s=0.5

    if show_se:
        xs_se_uppper = []
        ys_se_uppper = []

        xs_se_lower = []
        ys_se_lower = []

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
        xs_se_uppper.append(s1[0])
        ys_se_uppper.append(s1[1])

        xs_se_uppper.append(s2[0])
        ys_se_uppper.append(s2[1])

        xs_se_uppper.append(s3[0])
        ys_se_uppper.append(s3[1])

        # Bottom first
        xs_se_lower.append(-s1[0])
        ys_se_lower.append(-s1[1])

        xs_se_lower.append(-s2[0])
        ys_se_lower.append(-s2[1])

        xs_se_lower.append(-s3[0])
        ys_se_lower.append(-s3[1])

        # Bottom second
        xs_se_lower.append(2 * s1[0])
        ys_se_lower.append(2 * s1[1])

        xs_se_lower.append(2 * s2[0])
        ys_se_lower.append(2 * s2[1])

        xs_se_lower.append(2 * s3[0])
        ys_se_lower.append(2 * s3[1])

        # Top second
        xs_se_uppper.append(-2 * s1[0])
        ys_se_uppper.append(-2 * s1[1])

        xs_se_uppper.append(-2 * s2[0])
        ys_se_uppper.append(-2 * s2[1])

        xs_se_uppper.append(-2 * s3[0])  # xs.append(-Params.a_1[0] - 2 * Params.a_2[0] + s1[0])
        ys_se_uppper.append(-2 * s3[1])  # ys.append(-Params.a_1[1] - 2 * Params.a_2[1] + s1[1])

        ax.scatter(xs_se_uppper, ys_se_uppper, color='#38cf5b')  # 88c999
        ax.scatter(xs_se_lower, ys_se_lower, color='#a7c9af')

    # K = Params.K
    # M = Params.M

    # A = Params.K
    # B = (K[0], -K[1])
    # ax.plot([A[0] * 1 / 3, B[0] * 1 / 3], [A[1] * 1 / 3, B[1] * 1 / 3])

    # Finish plot & save
    fig.colorbar(c, ax=ax, label='')  # TODO: change/try a logarithmic color scale/bar .
    fig.gca().set_aspect('equal', adjustable='box')
    fig.tight_layout()
    fig.savefig(filename, bbox_inches='tight', dpi=300)
    # fig.show()
    plt.close(fig)
    print('heat map plot: \n {} \n'.format(filename))


def plot_heatmap_on_axex(ax, xs_top, ys_top, zs, colormap='hot'):
    c = ax.pcolormesh(xs_top, ys_top, zs, cmap=colormap)
    # K = Params.K
    # M = Params.M

    # A = Params.K
    # B = (K[0], -K[1])
    # ax.plot([A[0] * 1 / 3, B[0] * 1 / 3], [A[1] * 1 / 3, B[1] * 1 / 3])


def plot_4_band_fermi_surface_on_axes(ax, kxs, kys):
    from hamiltonian_4_band import get_4_band_hamiltonian
    from hamiltonian_4_band import calculate_4_band_H_k
    from hamiltonian_4_band import extract_eigenstates_from_hamiltonian

    H_arr_R, rs, weights = get_4_band_hamiltonian(FilePaths.hamiltonian_NbSe2_4band_data)
    # fig, ax = plt.subplots()
    # Fermi-energy
    E_f = Params.E_f  # eV
    # dE_f = 110 * pow(10, -3)
    dE_f = Params.delta_E_f

    E_f = E_f + dE_f

    # make k-space (to be used in constructing H(k))
    kxn = len(kxs)
    kyn = len(kys)

    # make mesh to hold hamiltonians(k) (mesh of 2*2 matrices)

    eigenvalues_k = np.zeros((kxn, kyn), dtype=type([1.]))

    for i in range(kxn):
        for j in range(kyn):
            kx = kxs[i]
            ky = kys[j]

            h_k = calculate_4_band_H_k(H_arr_R, rs, weights, kx, ky)
            _, evs = extract_eigenstates_from_hamiltonian(h_k)
            eigenvalues_k[i, j] = evs - E_f

        if i % 20 == 0:
            print('finished row: {}'.format(i))

    # plot fermi-surface
    for i in range(4):
        zs = np.zeros((kxn, kyn), dtype=float)

        for j in range(kxn):
            for k in range(kyn):
                zs[j, k] = eigenvalues_k[j, k][i].real

        ax.contour(kxs, kys, zs.transpose(), levels=[0], linewidths=[0.3], linestyles=['solid'], colors=['#03fcfc'])

    # return ax
    # fig.gca().set_aspect('equal', adjustable='box')
    # fig.tight_layout()
    # fig.savefig('test.png', bbox_inches='tight', dpi=300)
    # fig.show()
    # plt.close(fig)
    # , colors=['black']
    # ax.set_xlabel('$k_x$ $(\AA^{-1})$')
    # ax.set_ylabel('$k_y$ $(\AA^{-1})$')


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
    fig.savefig(filepath, bbox_inches='tight')
    # fig.show()
    plt.close(fig)
    print('Points plot:\n{}\n'.format(filepath))


def plot_scatter(xs, ys, filepath):
    plt.scatter(xs, ys)
    plt.xlabel = '$k_y$ at $k_x = 0$'
    plt.ylabel = '$|Tr(G_0(k))|$'

    plt.savefig(filepath, dpi=300)


def plot_line(xs, ys, filepath):
    plt.plot(xs, ys)
    plt.xlabel = '$k_y$ at $k_x = 0$'
    plt.ylabel = '$|Tr(G_0(k))|$'

    plt.savefig(filepath, dpi=300)

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
#     fig.savefig(filename, bbox_inches='tight')
#     # fig.show()
#     plt.close(fig)
#     print('heat map plot: \n {} \n'.format(filename))


def plot_combined_18_band_and_2_band(scattering_type, kxn, kx_length, version_number):
    print('Combined plot:')

    dkx = kx_length / kxn
    xn = kxn
    x_length = 2 * np.pi / dkx  # a/4 angstrom

    phi_k_plot = get_phi_k_plot_filepath(kxn=kxn, kx_length=kx_length, version_number=version_number)
    phi_r_plot = get_phi_r_plot_filepath(xn=xn, x_length=x_length, version_number=version_number)
    phi_r_sub_plot = get_phi_r_sub_plot_filepath(xn=Params.xn_sub, x_length=Params.x_length_sub,
                                                 version_number=version_number)
    qpi_18_bands_plot_path = get_qpi_18_band_filepath(scattering_type=scattering_type, kxn=kxn, kx_length=kx_length,
                                                      version_number=version_number)
    qpi_2_bands_plot_path = get_qpi_2_band_filepath(scattering_type=scattering_type, kxn=kxn, kx_length=kx_length,
                                                    version_number=version_number)
    greens_2_bands_plot_path = get_bgtrace_2_band_plot_filepath(kxn=kxn, kx_length=kx_length,
                                                                version_number=version_number)
    greens_18_bands_plot_path = get_bgtrace_18_band_plot_filepath(kxn=kxn, kx_length=kx_length,
                                                                  version_number=version_number)

    out_filepath = get_combined_18_and_2_plot_filepath(version_number=version_number)

    top_images = [Image.open(x) for x in [phi_r_sub_plot, greens_18_bands_plot_path, qpi_18_bands_plot_path]]

    for i in range(len(top_images)):
        width, height = top_images[i].size
        ratio = height / width
        top_images[i] = top_images[i].resize((1000, int(1000 * ratio)))

    bottom_images = [Image.open(x) for x in [phi_k_plot, greens_2_bands_plot_path, qpi_2_bands_plot_path]]

    for i in range(len(bottom_images)):
        width, height = bottom_images[i].size
        ratio = height / width
        bottom_images[i] = bottom_images[i].resize((1000, int(1000 * ratio)))

    top_left_widths, top_left_heights = zip(*(i.size for i in top_images))
    bottom_left_widths, bottom_left_heights = zip(*(i.size for i in bottom_images))

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
    for im in top_images:
        new_top_left_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    x_offset = 0
    for im in bottom_images:
        new_bottom_left_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    left_image.paste(new_top_left_im, (0, 0))
    left_image.paste(new_bottom_left_im, (0, top_left_max_height))

    image = Image.new('RGB', (left_total_width, left_total_height), (255, 255, 255))

    image.paste(left_image, (0, 0))

    image.save(out_filepath)
    print(out_filepath)


def plot_combined_select_18_band_and_2_band(scattering_type, kxn, kx_length, bands, version_number):
    print('Combined plot:')

    dkx = kx_length / kxn
    xn = kxn
    x_length = 2 * np.pi / dkx  # a/4 angstrom

    phi_k_plot = get_phi_k_plot_filepath(kxn=kxn, kx_length=kx_length, version_number=version_number)
    phi_r_plot = get_phi_r_plot_filepath(xn=xn, x_length=x_length, version_number=version_number)
    phi_r_sub_plot = get_phi_r_sub_plot_filepath(xn=Params.xn_sub, x_length=Params.x_length_sub,
                                                 version_number=version_number)
    qpi_18_bands_plot_path = get_qpi_select_18_band_filepath(scattering_type=scattering_type, kxn=kxn,
                                                             kx_length=kx_length, bands=bands,
                                                             version_number=version_number)
    qpi_2_bands_plot_path = get_qpi_2_band_filepath(scattering_type=scattering_type, kxn=kxn, kx_length=kx_length,
                                                    version_number=version_number)
    greens_2_bands_plot_path = get_bgtrace_2_band_plot_filepath(kxn=kxn, kx_length=kx_length,
                                                                version_number=version_number)
    greens_18_bands_plot_path = get_bgtrace_select_18_band_plot_filepath(kxn=kxn, kx_length=kx_length, bands=bands,
                                                                         version_number=version_number)

    out_filepath = get_combined_select_18_and_2_plot_filepath(bands=bands, version_number=version_number)

    top_images = [Image.open(x) for x in [phi_r_sub_plot, greens_18_bands_plot_path, qpi_18_bands_plot_path]]

    for i in range(len(top_images)):
        width, height = top_images[i].size
        ratio = height / width
        top_images[i] = top_images[i].resize((1000, int(1000 * ratio)))

    bottom_images = [Image.open(x) for x in [phi_k_plot, greens_2_bands_plot_path, qpi_2_bands_plot_path]]

    for i in range(len(bottom_images)):
        width, height = bottom_images[i].size
        ratio = height / width
        bottom_images[i] = bottom_images[i].resize((1000, int(1000 * ratio)))

    top_left_widths, top_left_heights = zip(*(i.size for i in top_images))
    bottom_left_widths, bottom_left_heights = zip(*(i.size for i in bottom_images))

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
    for im in top_images:
        new_top_left_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    x_offset = 0
    for im in bottom_images:
        new_bottom_left_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    left_image.paste(new_top_left_im, (0, 0))
    left_image.paste(new_bottom_left_im, (0, top_left_max_height))

    image = Image.new('RGB', (left_total_width, left_total_height), (255, 255, 255))

    image.paste(left_image, (0, 0))

    image.save(out_filepath)
    print(out_filepath)


def plot_combined_18_band(scattering_type, kxn, kx_length, version_number):
    print('Combined plot:')

    dkx = kx_length / kxn
    xn = kxn
    x_length = 2 * np.pi / dkx  # a/4 angstrom

    phi_k_plot = get_phi_k_plot_filepath(kxn=kxn, kx_length=kx_length, version_number=version_number)
    phi_r_plot = get_phi_r_plot_filepath(xn=xn, x_length=x_length, version_number=version_number)
    phi_r_sub_plot = get_phi_r_sub_plot_filepath(xn=Params.xn_sub, x_length=Params.x_length_sub,
                                                 version_number=version_number)
    qpi_plot = get_qpi_18_band_filepath(scattering_type=scattering_type, kxn=kxn, kx_length=kx_length,
                                        version_number=version_number)
    greens_plot = get_bgtrace_18_band_plot_filepath(kxn=kxn, kx_length=kx_length, version_number=version_number)

    out_filepath = get_combined_18_band_plot_filepath(version_number=version_number)

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

    right_image_width, right_image_height = right_image.size
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


def plot_combined_4_band(scattering_type, kxn, kx_length, version_number):
    print('Combined plot:')

    dkx = kx_length / kxn
    xn = kxn
    x_length = 2 * np.pi / dkx  # a/4 angstrom

    phi_k_plot = get_phi_k_plot_filepath(kxn=kxn, kx_length=kx_length, version_number=version_number)
    phi_r_plot = get_phi_r_plot_filepath(xn=xn, x_length=x_length, version_number=version_number)
    phi_r_sub_plot = get_phi_r_sub_plot_filepath(xn=Params.xn_sub, x_length=Params.x_length_sub,
                                                 version_number=version_number)
    qpi_plot = get_qpi_4_band_filepath(scattering_type=scattering_type, kxn=kxn, kx_length=kx_length,
                                       version_number=version_number)
    greens_plot = get_bgtrace_4_band_plot_filepath(kxn=kxn, kx_length=kx_length, version_number=version_number)

    out_filepath = get_combined_4_band_plot_filepath(version_number=version_number)

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

    right_image_width, right_image_height = right_image.size
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


def plot_combined_2_band(scattering_type, kxn, kx_length, version_number):
    print('Combined plot:')

    dkx = kx_length / kxn
    xn = kxn
    x_length = 2 * np.pi / dkx  # a/4 angstrom

    phi_k_plot = get_phi_k_plot_filepath(kxn=kxn, kx_length=kx_length, version_number=version_number)
    phi_r_plot = get_phi_r_plot_filepath(xn=xn, x_length=x_length, version_number=version_number)
    phi_r_sub_plot = get_phi_r_sub_plot_filepath(xn=Params.xn_sub, x_length=Params.x_length_sub,
                                                 version_number=version_number)
    qpi_plot = get_qpi_2_band_filepath(scattering_type=scattering_type, kxn=kxn, kx_length=kx_length,
                                       version_number=version_number)
    greens_plot = get_bgtrace_2_band_plot_filepath(kxn=kxn, kx_length=kx_length, version_number=version_number)

    out_filepath = get_combined_2_band_plot_filepath(version_number=version_number)

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

    right_image_width, right_image_height = right_image.size
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


def plot_combined_18_band_split_into_9_2by2(version_string):
    print('Combined plot:')

    out_filepath = 'temp/plots/combined_{}_.png'.format(version_string)

    top_images = [Image.open(x) for x in
                  ['temp/plots/greens_i={}_j={}_{}.png'.format(i, i, version_string) for i in range(9)]]

    for i in range(len(top_images)):
        width, height = top_images[i].size
        ratio = height / width
        top_images[i] = top_images[i].resize((1000, int(1000 * ratio)))

    bottom_images = [Image.open(x) for x in
                     ['temp/plots/qpi_{}_{}__{}.png'.format(i, i, version_string) for i in range(9)]]

    for i in range(len(bottom_images)):
        width, height = bottom_images[i].size
        ratio = height / width
        bottom_images[i] = bottom_images[i].resize((1000, int(1000 * ratio)))

    top_left_widths, top_left_heights = zip(*(i.size for i in top_images))
    bottom_left_widths, bottom_left_heights = zip(*(i.size for i in bottom_images))

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
    for im in top_images:
        new_top_left_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    x_offset = 0
    for im in bottom_images:
        new_bottom_left_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    left_image.paste(new_top_left_im, (0, 0))
    left_image.paste(new_bottom_left_im, (0, top_left_max_height))

    image = Image.new('RGB', (left_total_width, left_total_height), (255, 255, 255))

    image.paste(left_image, (0, 0))

    image.save(out_filepath)
    print(out_filepath)


def plot_combined_greens_and_qpi_for_ijs(i_s, j_s, version_string):
    print('i_s = {}'.format(i_s))
    print('j_s = {}'.format(j_s))

    i_s_str = ''
    for i in i_s:
        if not i_s_str == '':
            i_s_str += ','
        i_s_str += str(i)

    j_s_str = ''
    for j in j_s:
        if not j_s_str == '':
            j_s_str += ','
        j_s_str += str(j)

    out_filepath = 'temp/plots/combined_is={}_js={}__{}_.png'.format(i_s_str, j_s_str, version_string)

    # Get Top images
    top_images = [Image.open(x) for x in
                  ['temp/plots/greens_i={}_j={}_{}.png'.format(i, j, version_string) for i, j in zip(i_s, j_s)]]

    for i in range(len(top_images)):
        width, height = top_images[i].size
        ratio = height / width
        top_images[i] = top_images[i].resize((1000, int(1000 * ratio)))

    # Get bottom images
    bottom_images = [Image.open(x) for x in
                     ['temp/plots/qpi_i={}_j={}__{}.png'.format(i, j, version_string) for i, j in zip(i_s, j_s)]]

    for i in range(len(bottom_images)):
        width, height = bottom_images[i].size
        ratio = height / width
        bottom_images[i] = bottom_images[i].resize((1000, int(1000 * ratio)))

    # get widths of images
    top_images_widths, top_images_heights = zip(*(i.size for i in top_images))
    bottom_images_widths, bottom_images_heights = zip(*(i.size for i in bottom_images))

    # Make text labels
    text_labels = []
    font = ImageFont.truetype("LEMONMILK-Regular.otf", 40)  # load font
    text_label_height = 50

    for i in range(len(top_images_widths)):
        img = Image.new('RGB', (top_images_widths[i], text_label_height), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        draw.text((0, 0), "i={}, j={}".format(i_s[i], j_s[i]), (0, 0, 0), font=font)  # draw text

        text_labels.append(img)

    top_left_total_width = sum(top_images_widths)
    top_left_max_height = max(top_images_heights)

    bottom_left_total_width = sum(bottom_images_widths)
    bottom_left_max_height = max(bottom_images_heights)

    left_total_width = max(top_left_total_width, bottom_left_total_width)
    left_total_height = top_left_max_height + bottom_left_max_height + text_label_height

    all_text_label_im = Image.new('RGB', (top_left_total_width, text_label_height), (255, 255, 255))
    new_top_left_im = Image.new('RGB', (top_left_total_width, top_left_max_height), (255, 255, 255))
    new_bottom_left_im = Image.new('RGB', (bottom_left_total_width, bottom_left_max_height), (255, 255, 255))

    left_image = Image.new('RGB', (left_total_width, left_total_height), (255, 255, 255))

    x_offset = 0
    for txt_lbl in text_labels:
        all_text_label_im.paste(txt_lbl, (x_offset, 0))
        x_offset += txt_lbl.size[0]

    x_offset = 0
    for im in top_images:
        new_top_left_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    x_offset = 0
    for im in bottom_images:
        new_bottom_left_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    left_image.paste(all_text_label_im, (0, 0))
    left_image.paste(new_top_left_im, (0, text_label_height))
    left_image.paste(new_bottom_left_im, (0, top_left_max_height + text_label_height))

    image = Image.new('RGB', (left_total_width, left_total_height), (255, 255, 255))

    image.paste(left_image, (0, 0))

    image.save(out_filepath)
    print(out_filepath)


def plot_combined_greens_and_qpi_contributions(version_str):
    counter = 0
    i_s = []
    j_s = []
    print('here')
    for i in range(0, 18, 2):
        for j in range(1, 18, 2):
            if j - i == 1:
                continue
            print(i, j)
            counter += 1

            i_s.append(i)
            j_s.append(j)

            if counter % 3 == 0 and counter > 0:
                # plot the past three
                plot_combined_greens_and_qpi_for_ijs(i_s, j_s, version_string=version_str)

                i_s = []
                j_s = []


def plot_combined_green_bands(version_number):
    out_filepath = 'temp/plots/combined_greens_band_investigation__{}eV_{}_.png'.format(Params.delta_E_f,
                                                                                        version_number)

    # Get Greens images
    top_images = [Image.open(x) for x in
                  ['temp/plots/band_invest_greens_band={}__{}eV_{}.png'.format(i, Params.delta_E_f, version_number) for
                   i in range(18)]]

    for i in range(len(top_images)):
        width, height = top_images[i].size
        ratio = height / width
        top_images[i] = top_images[i].resize((1000, int(1000 * ratio)))

    # get widths of images
    top_images_widths, top_images_heights = zip(*(i.size for i in top_images))

    # Make text labels
    text_labels = []
    font = ImageFont.truetype("LEMONMILK-Regular.otf", 40)  # load font
    text_label_height = 50

    for i in range(len(top_images_widths)):
        img = Image.new('RGB', (top_images_widths[i], text_label_height), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        draw.text((0, 0), "band={}".format(i), (0, 0, 0), font=font)  # draw text

        text_labels.append(img)

    top_left_total_width = sum(top_images_widths)
    top_left_max_height = max(top_images_heights)

    left_total_width = top_left_total_width
    left_total_height = top_left_max_height + text_label_height

    all_text_label_im = Image.new('RGB', (top_left_total_width, text_label_height), (255, 255, 255))
    new_top_left_im = Image.new('RGB', (top_left_total_width, top_left_max_height), (255, 255, 255))

    left_image = Image.new('RGB', (left_total_width, left_total_height), (255, 255, 255))

    x_offset = 0
    for txt_lbl in text_labels:
        all_text_label_im.paste(txt_lbl, (x_offset, 0))
        x_offset += txt_lbl.size[0]

    x_offset = 0
    for im in top_images:
        new_top_left_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    left_image.paste(all_text_label_im, (0, 0))
    left_image.paste(new_top_left_im, (0, text_label_height))

    image = Image.new('RGB', (left_total_width, left_total_height), (255, 255, 255))

    image.paste(left_image, (0, 0))

    image.save(out_filepath)
    print(out_filepath)


def plot_combined_green_threshold():
    out_filepath = '../plots/combined_greens_threshold_investigation__-50eV_.png'

    # Get Greens images
    top_images = [Image.open(x) for x in
                  [FilePaths.greens_plots + '/G_trace_4_bands_kxn=300_kxlen=3.65___1.91.17_Scalar_4_band_just-diag_eigenstate-1,2,3,4_threshold={}__non-norm_central-alpha=5_dEf=-0.05_kxn=300_.png'.format(i*5) for
                   i in range(20)]]

    for i in range(len(top_images)):
        width, height = top_images[i].size
        ratio = height / width
        top_images[i] = top_images[i].resize((1000, int(1000 * ratio)))

    # get widths of images
    top_images_widths, top_images_heights = zip(*(i.size for i in top_images))

    # Make text labels
    text_labels = []
    font = ImageFont.truetype(FilePaths.font_lemonmilk, 40)  # load font
    text_label_height = 50

    for i in range(len(top_images_widths)):
        img = Image.new('RGB', (top_images_widths[i], text_label_height), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        draw.text((0, 0), "threshold={}".format(i*5), (0, 0, 0), font=font)  # draw text

        text_labels.append(img)

    top_left_total_width = sum(top_images_widths)
    top_left_max_height = max(top_images_heights)

    left_total_width = top_left_total_width
    left_total_height = top_left_max_height + text_label_height

    all_text_label_im = Image.new('RGB', (top_left_total_width, text_label_height), (255, 255, 255))
    new_top_left_im = Image.new('RGB', (top_left_total_width, top_left_max_height), (255, 255, 255))

    left_image = Image.new('RGB', (left_total_width, left_total_height), (255, 255, 255))

    x_offset = 0
    for txt_lbl in text_labels:
        all_text_label_im.paste(txt_lbl, (x_offset, 0))
        x_offset += txt_lbl.size[0]

    x_offset = 0
    for im in top_images:
        new_top_left_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    left_image.paste(all_text_label_im, (0, 0))
    left_image.paste(new_top_left_im, (0, text_label_height))

    image = Image.new('RGB', (left_total_width, left_total_height), (255, 255, 255))

    image.paste(left_image, (0, 0))

    image.save(out_filepath)
    print(out_filepath)


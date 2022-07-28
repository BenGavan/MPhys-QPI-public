from params import *
from matrix_utils import *


class FilePaths:
    orbital_data = '../data/ld1.wfc'
    hamiltonian_NbSe2_4band_data = '../../data/NbSe2_hr_(4-band).dat'
    hamiltonian_NbSe2_2band_data = '../../data/Hamiltonian_2band.dat'
    fonts = '../../fonts/'
    font_lemonmilk = '/Users/ben/Desktop/MPhys-QPI/project/QPI/NbSe2/data/fonts/LEMONMILK-Regular.otf'
    greens_cache = '/Users/ben/Desktop/MPhys-QPI/project/QPI/NbSe2/data/greens/'
    rho_cache = '/Users/ben/Desktop/MPhys-QPI/project/QPI/NbSe2/4-band/greens/rho_q/'
    phi_k_plots = '/Users/ben/Desktop/MPhys-QPI/project/QPI/NbSe2/4-band/plots/phi_k'
    phi_r_plots = '/Users/ben/Desktop/MPhys-QPI/project/QPI/NbSe2/4-band/plots/phi_r'
    greens_plots = '/Users/ben/Desktop/MPhys-QPI/project/QPI/NbSe2/4-band/plots/greens'
    qpi_plots = '/Users/ben/Desktop/MPhys-QPI/project/QPI/NbSe2/4-band/plots/QPI'
    combined_plots = '/Users/ben/Desktop/MPhys-QPI/project/QPI/NbSe2/4-band/plots/combined'
    run_history = '/Users/ben/Desktop/MPhys-QPI/project/QPI/NbSe2/4-band/run_history.txt'


def get_qpi_2_band_filepath(scattering_type, kxn, kx_length, version_number):
    filename = ''
    if scattering_type == Scattering.Scalar_2_bands:
        filename = FilePaths.qpi_plots + '/scalar/scalar_2_band_qpi_kx_len={}__{}.png'
    # elif scattering_type == Scattering.Magnetic_2_bands:
    #     filename = FilePaths.qpi_plots + '/magnetic/magnetic_omega={}eV_kxn={}_kxlen={:.2f}__{}.pdf'
    # elif scattering_type == Scattering.Scalar_and_Spin_orbit_2_bands:
    #     filename = FilePaths.qpi_plots + '/scalar+SO/spin-orbit_scalar_omega={}eV_c=' + str(
    #         Params.c) + '_n=101_kxn={}_kxlen={:.2f}__{}.png'
    elif scattering_type == Scattering.Scalar_and_Spin_orbit_2_bands:
        filename = FilePaths.qpi_plots + '/scalar+SO/spin-orbit_scalar_2-band_c=' + str(
            Params.c) + 'qpi_kx-len={}_{}.png'

    return filename.format(kx_length, version_number)


def get_qpi_4_band_filepath(scattering_type, kxn, kx_length, version_number):
    filename = ''
    if scattering_type == Scattering.Scalar_4_band:
        filename = FilePaths.qpi_plots + '/scalar/scalar_4_band_qpi_kx_len={}__{}.png'
    elif scattering_type == Scattering.Scalar_and_Spin_orbit_4_bands:
        filename = FilePaths.qpi_plots + '/scalar+SO/spin-orbit_scalar_4-band_c=' + str(
            Params.c) + 'qpi_kx-len={}_{}.png'

    return filename.format(kx_length, version_number)


def get_qpi_18_band_filepath(scattering_type, kxn, kx_length, version_number):
    filename = ''
    if scattering_type == Scattering.Scalar_2_bands:
        filename = FilePaths.qpi_plots + '/scalar/scalar_18_band_qpi_kx-len={}__{}.png'
    elif scattering_type == Scattering.Scalar_and_Spin_orbit_18_bands:
        filename = FilePaths.qpi_plots + '/scalar+SO/spin-orbit_scalar_18-bands_at_once__qpi_kx-len={}_c=' + str(
            Params.c) + '__{}.png'
    elif scattering_type == Scattering.Scalar_and_Spin_orbit_2_bands or scattering_type == Scattering.Scalar_and_Spin_orbit_6_bands:
        filename = FilePaths.qpi_plots + '/scalar+SO/spin-orbit_scalar_18-bands_qpi_kx-len={}_c=' + str(
            Params.c) + '__{}.png'

    return filename.format(kx_length, version_number)


def get_qpi_select_18_band_filepath(scattering_type, kxn, kx_length, bands, version_number):
    bands_str = ''
    for i in bands:
        if bands_str != '':
            bands_str += ','
        bands_str += str(i)

    filename = ''
    if scattering_type == Scattering.Scalar_2_bands:
        filename = FilePaths.qpi_plots + '/scalar/scalar_select_18_band_qpi_kx-len={}_{}__{}.png'
    # elif scattering_type == Scattering.Magnetic_2_bands:
    #     filename = FilePaths.qpi_plots + '/magnetic/magnetic_omega={}eV_kxn={}_kxlen={:.2f}__{}.pdf'
    elif scattering_type == Scattering.Scalar_and_Spin_orbit_18_bands:
        filename = FilePaths.qpi_plots + '/scalar+SO/spin-orbit_scalar_select_18-bands_at_once__qpi_kx-len={}_c=' + str(
            Params.c) + '_{}__{}.png'
    elif scattering_type == Scattering.Scalar_and_Spin_orbit_2_bands or scattering_type == Scattering.Scalar_and_Spin_orbit_6_bands:
        filename = FilePaths.qpi_plots + '/scalar+SO/spin-orbit_scalar_select_18-bands_qpi_kx-len={}_c=' + str(
            Params.c) + '_{}__{}.png'

    return filename.format(kx_length, bands_str, version_number)


def get_phi_r_plot_filepath(xn, x_length, version_number):
    return FilePaths.phi_r_plots + "/phi_r_xn={}_xlen={:.2f}__{}.png".format(xn, x_length, version_number)


def get_phi_r_sub_plot_filepath(xn, x_length, version_number):
    return FilePaths.phi_r_plots + "/phi_r_xn={}_xlen={:.2f}__{}.png".format(xn, x_length, version_number)


def get_phi_k_plot_filepath(kxn, kx_length, version_number):
    return FilePaths.phi_k_plots + "/phi_k_kxn={}_kxlen={:.2f}__{}.png".format(kxn, kx_length, version_number)


def get_bgtrace_2_band_plot_filepath(kxn, kx_length, version_number):
    return FilePaths.greens_plots+'/G_trace_2_bands_kxn={}_kxlen={:.2f}__{}.png'.format(kxn, kx_length, version_number)


def get_bgtrace_4_band_plot_filepath(kxn, kx_length, version_number):
    return FilePaths.greens_plots+'/G_trace_4_bands_kxn={}_kxlen={:.2f}__{}.png'.format(kxn, kx_length, version_number)


def get_bgtrace_18_band_plot_filepath(kxn, kx_length, version_number):
    return FilePaths.greens_plots+'/G_trace_18_bands_kxn={}_kxlen={:.2f}__{}.png'.format(kxn, kx_length, version_number)


def get_bgtrace_select_18_band_plot_filepath(kxn, kx_length, bands, version_number):
    bands_str = ''
    for i in bands:
        if bands_str != '':
            bands_str += ','
        bands_str += str(i)
    return FilePaths.greens_plots+'/G_trace_18_bands_kxn={}_kxlen={:.2f}_bands={}_{}.png'.format(kxn, kx_length, bands_str, version_number)


def get_combined_18_and_2_plot_filepath(version_number):
    return FilePaths.combined_plots + '/combined_18_w_2_{}.png'.format(version_number)


def get_combined_select_18_and_2_plot_filepath(bands, version_number):
    bands_str = ''
    for i in bands:
        if bands_str != '':
            bands_str += ','
        bands_str += str(i)
    return FilePaths.combined_plots + '/combined_18_w_2_{}_{}.png'.format(bands_str, version_number)


def get_combined_18_band_plot_filepath(version_number):
    return FilePaths.combined_plots + '/combined_18_band_{}.png'.format(version_number)


def get_combined_4_band_plot_filepath(version_number):
    return FilePaths.combined_plots + '/combined_4_band_{}.png'.format(version_number)


def get_combined_2_band_plot_filepath(version_number):
    return FilePaths.combined_plots + '/combined_2_band_{}.png'.format(version_number)


def appends_val_to_file(filepath, x):
    f = open(filepath, 'a')
    f.write(str(x))
    f.write('\n')
    f.close()


def append_arr_to_file(filename, vs):
    f = open(filename, 'a')

    for v in vs:
        f.write(str(v))
        f.write(' ')
    f.write('\n')

    f.close()


def append_matrix_to_file(filename, m):
    f = open(filename, 'a')

    xn = m.shape[0]
    yn = m.shape[1]

    for i in range(xn):
        for j in range(yn):
            f.write(str(m[i, j]))
            f.write(' ')
        f.write('\n')

    f.close()


def write_phi_r_data(filename, xn, yn, x_length, y_length, kxn, kyn, kx_length, ky_length, phi_r):
    f = open(filename, 'w')

    f.write(str(xn))
    f.write('\n')
    f.write(str(yn))
    f.write('\n')
    f.write(str(x_length))
    f.write('\n')
    f.write(str(y_length))
    f.write('\n')
    f.write(str(kxn))
    f.write('\n')
    f.write(str(kyn))
    f.write('\n')
    f.write(str(kx_length))
    f.write('\n')
    f.write(str(ky_length))
    f.write('\n')

    for i in range(xn):
        for j in range(yn):
            f.write(str(phi_r[i, j]))
            f.write(' ')
        f.write('\n')

    f.close()


def read_phi_k_data(filename):
    kxn = None
    kyn = None

    kx_length = None
    ky_length = None

    f = open(filename, 'r')

    phi_k = None

    line_counter = -1
    for line in f:
        line_counter += 1
        if line_counter <= 3:
            vs = extract_floats_from_line(line)
            if len(vs) == 1:
                if line_counter == 0:
                    kxn = int(vs[0])
                elif line_counter == 1:
                    kyn = int(vs[0])
                    phi_k = np.zeros((kxn, kyn), dtype=complex)
                elif line_counter == 2:
                    kx_length = float(vs[0])
                elif line_counter == 3:
                    ky_length = float(vs[0])
                continue

        vs = extract_complex_from_line(line)

        for i in range(len(vs)):
            phi_k[line_counter - 4][i] = vs[i]

    return kxn, kyn, kx_length, ky_length, phi_k


# appends to  'filename' xs, ys, zs, space seperated with new line between data
#  xs = kx sample values
#  ys = ky sample values
#  zs = Hamiltonian eigenvalues (singular (selected one only from pair (either min or max)))
def write_heatmap_data(filename, xs, ys, zs):
    print('writing data to: ', filename)
    append_arr_to_file(filename, xs)
    append_arr_to_file(filename, ys)
    append_matrix_to_file(filename, zs)


def get_heatmap_data(filename):
    xs = np.array([])
    ys = np.array([])
    zs = None

    f = open(filename, 'r')
    line_counter = 0
    for line in f:
        els = line.split(' ')[:-1]
        for i in range(len(els)):
            if line_counter == 0:
                xs = np.append(xs, float(els[i]))
            elif line_counter == 1:
                ys = np.append(ys, float(els[i]))
            else:
                zs[line_counter - 2][i] = float(els[i])

        if line_counter == 1:
            zs = np.zeros((len(xs), len(ys)))

        line_counter += 1

    f.close()

    return xs, ys, zs


def write_complex_data(filename, xs, ys, zs):
    print('writing data to: ', filename)
    append_arr_to_file(filename, xs)
    append_arr_to_file(filename, ys)
    append_matrix_to_file(filename, zs)


def get_complex_data(filename):
    xs = np.array([])
    ys = np.array([])
    zs = None

    f = open(filename, 'r')
    line_counter = 0
    for line in f:
        els = line.split(' ')[:-1]
        for i in range(len(els)):
            if line_counter == 0:
                xs = np.append(xs, float(els[i]))
            elif line_counter == 1:
                ys = np.append(ys, float(els[i]))
            else:
                zs[line_counter - 2][i] = complex(els[i])

        if line_counter == 1:
            zs = np.zeros((len(xs), len(ys)), dtype=complex)

        line_counter += 1

    f.close()

    return xs, ys, zs


def write_2_bands_greens_data(filename, xs, ys, gs):
    print('writing data to: ', filename)
    append_arr_to_file(filename, xs)
    append_arr_to_file(filename, ys)
    gs_flat = flatten_mesh_of_matrices(gs)
    append_matrix_to_file(filename, gs_flat[0, 0])
    append_matrix_to_file(filename, gs_flat[0, 1])
    append_matrix_to_file(filename, gs_flat[1, 0])
    append_matrix_to_file(filename, gs_flat[1, 1])


def get_2_bands_greens_data(filename):
    kxs = np.array([])
    kys = np.array([])
    gs_flat = np.matrix(np.zeros((2, 2), dtype=np.matrix))

    f = open(filename, 'r')
    line_counter = 0
    ky_i = 0
    for line in f:
        els = line.split(' ')[:-1]
        # print(line_counter, els)
        for i in range(len(els)):
            if line_counter == 0:
                kxs = np.append(kxs, float(els[i]))
            elif line_counter == 1:
                kys = np.append(kys, float(els[i]))
            else:
                g_i = int((line_counter - 2) / len(kxs))

                x_i = line_counter - 2 - (len(kxs)*g_i)
                gs_flat[g_i % 2, int((g_i - (g_i % 2)) / 2)][x_i, i] = complex(els[i])

        if line_counter > 1:
            ky_i += 1
            if (line_counter - 2) % len(kys) == 0:
                ky_i = 0

        if line_counter == 1:
            for i in range(2):
                for j in range(2):
                    gs_flat[i, j] = np.matrix(np.zeros((len(kxs), len(kys)), dtype=complex))

        line_counter += 1

    f.close()

    return kxs, kys, unflatten_to_mesh_of_matrices(gs_flat)



def write_4_bands_greens_data(filename, xn, yn, gs):
    print('writing data to: ', filename)
    appends_val_to_file(filename, xn)
    appends_val_to_file(filename, yn)
    for i in range(xn):
        for j in range(yn):
            append_matrix_to_file(filename, gs[i, j])


def get_4_bands_greens_data(filename):
    f = open(filename, 'r')

    kxn = 0
    kyn = 0

    gs = None

    line_counter = -1

    for line in f:
        line_counter += 1
        if line_counter == 0:
            kxn = int(line)
        elif line_counter == 1:
            kyn = int(line)
            gs = np.zeros((kxn, kyn), dtype=np.ndarray)
            for i in range(kxn):
                for j in range(kyn):
                    gs[i, j] = np.zeros((4, 4), dtype=complex)

        els = line.split(' ')[:-1]  # empty if 1st or 2nd line (only contain one int per line)

        for j in range(len(els)):
            i = int((line_counter-2) % 4)
            n = int((line_counter-2) / 4)  # number of matrices read so for (matrix number)
            x = int(n / kyn)
            y = int(n % kyn)
            gs[x, y][i, j] = complex(els[j])

    f.close()

    return gs


def write_18_bands_greens_data(filename, xn, yn, gs):
    print('writing data to: ', filename)
    appends_val_to_file(filename, xn)
    appends_val_to_file(filename, yn)
    for i in range(xn):
        for j in range(yn):
            append_matrix_to_file(filename, gs[i, j])

    # gs_flat = flatten_mesh_of_matrices(gs)
    # for i in range(18):
    #     for j in range(18):
    #         append_matrix_to_file(filename, gs_flat[i, j])


    # append_matrix_to_file(filename, gs_flat[0, 1])
    # append_matrix_to_file(filename, gs_flat[1, 0])
    # append_matrix_to_file(filename, gs_flat[1, 1])


def get_18_bands_greens_data(filename):
    f = open(filename, 'r')

    kxn = 0
    kyn = 0

    gs = None

    line_counter = -1

    for line in f:
        line_counter += 1
        if line_counter == 0:
            kxn = int(line)
        elif line_counter == 1:
            kyn = int(line)
            gs = np.zeros((kxn, kyn), dtype=np.ndarray)
            for i in range(kxn):
                for j in range(kyn):
                    gs[i, j] = np.zeros((18, 18), dtype=complex)

        els = line.split(' ')[:-1]  # empty if 1st or 2nd line (only contain one int per line)

        for j in range(len(els)):
            i = int((line_counter-2) % 18)
            n = int((line_counter-2) / 18)  # number of matrices read so for (matrix number)
            x = int(n / kyn)
            y = int(n % kyn)
            gs[x, y][i, j] = complex(els[j])

    f.close()

    return kxn, kyn, gs


def extract_floats_from_line(line):
    els = line.split(' ')

    vs = np.array([])

    for e in els:
        if e == '' or e == '\n':  # check for '\n' might not be needed (but added just in case)
            continue

        vs = np.append(vs, float(e))

    return vs


def extract_complex_from_line(line):
    els = line.split(' ')

    vs = np.array([], dtype=complex)

    for e in els:
        if e == '' or e == '\n':  # check for '\n' might not be needed (but added just in case)
            continue

        c_str = e.split(',')

        r = float(c_str[0].split('(')[1])
        i = float(c_str[1].split(')')[0])

        vs = np.append(vs, r + 1j*i)
    return vs


def get_2_band_hamiltonian(filepath):
    print('*** get hamiltonian: {}'.format(filepath))

    f = open(filepath)

    number_bands = -1
    number_hopping_vectors = -1  # = number of weights

    weights = np.array([])
    H_arr_R = np.array([], dtype=np.matrix)
    hopping_vectors = np.array([])

    line_num = -1
    H_line = -1
    for line in f:
        line_num += 1

        if line_num == 0:
            continue

        vs = extract_floats_from_line(line)

        if line_num == 1 and len(vs) == 1:
            number_bands = vs[0]
            continue

        if line_num == 2 and len(vs) == 1:
            number_hopping_vectors = int(vs[0])

            H_arr_R = np.array([np.zeros((2, 2), dtype=complex) for _ in range(number_hopping_vectors)])
            hopping_vectors = np.array([np.zeros(2, dtype=float) for _ in range(number_hopping_vectors)])
            continue

        if number_hopping_vectors == -1:
            continue  # still need to read number_hopping_vectors on line 2

        if len(weights) != number_hopping_vectors:
            weights = np.append(weights, vs)
            continue

        H_line += 1
        H_arr_index = int(H_line/4)

        if H_line % 4 == 0:  # only need to assign hopping vector at the start of each possition (repeasted 4 times correspinding to the 4 elements of the hamilitonian at a given point)
            hopping_vectors[H_arr_index] = vs[:2]
        H_arr_R[H_arr_index][int(vs[3])-1, int(vs[4])-1] = vs[5] + 1j * vs[6]

    f.close()

    rs = hopping_vectors

    return H_arr_R, rs, weights


def get_orbital_data(filepath):
    f = open(filepath)

    rs = np.array([])
    Rs = np.array([])

    line_counter = -1

    R_n = 0

    for line in f:
        line_counter += 1
        if line_counter == 0:
            continue

        vs = extract_floats_from_line(line)

        r = vs[0]
        rR = vs[1:]

         # if you only want to plot upto a certain r value - modify here
        # if r > 10 or r < 3:
        #     continue

        R_n = len(rR)

        #  Data in the format of r, r*R(r) (we want r, R(r))
        R = rR / r

        rs = np.append(rs, r)
        Rs = np.append(Rs, R)

    return rs, Rs, R_n


def append_matrix_as_column(filepath, rho_q):
    kxn = rho_q.shape[0]
    kyn = rho_q.shape[1]

    for i in range(kxn):
        for j in range(kyn):
            appends_val_to_file(filepath, rho_q[i, j].real)


def get_5s_orbital(filepath=FilePaths.orbital_data):
    rs, Rs, R_n = get_orbital_data(filepath)

    R_5s = [Rs[j] for j in range(2, len(Rs), R_n)]

    return rs, R_5s


def get_orbital(orbital, filepath=FilePaths.orbital_data):
    rs, Rs, R_n = get_orbital_data(filepath)

    R_5s = [Rs[j] for j in range(orbital.value, len(Rs), R_n)]

    return rs, R_5s


def write_18_band_rho_q(kxn, kyn, rho_q, version_number):
    # filepath = FilePaths.rho_cache + 'rho_q_18-band_kxn={}_kyn={}_{}_.dat'.format(kxn, kyn, version_number)
    # appends_val_to_file(filepath, kxn)
    # appends_val_to_file(filepath, kyn)
    filepath = 'temp/Saeed/rho_q_18-band_kxn={}_kyn={}_{}eV_{}_.dat'.format(kxn, kyn, Params.delta_E_f, version_number)
    append_matrix_as_column(filepath, rho_q)
    # append_matrix_to_file(filepath, rho_q)


def read_18_band_rho_q(kxn, kyn, version_number):
    filepath = FilePaths.rho_cache + 'rho_q_18-band_kxn={}_kyn={}_{}_.dat'.format(kxn, kyn, version_number)
    # filepath = FilePaths.rho_cache + 'rho_q_18-band_kxn=300_kyn=300__1.88.61_Scalar+SO_18_bands_just-diag_c=60_18-from-2_just-inner-BZ_k-pocket-radius=0.2__dEf=0.05_kxn=300_a=2__.dat'


    kxn_file = None
    kyn_file = None

    rho_q = None

    f = open(filepath, 'r')

    line_counter = -1
    for line in f:
        line_counter += 1
        if line_counter == 0:
            kxn_file = int(line)
        if line_counter == 1:
            kyn_file = int(line)
            rho_q = np.zeros((kxn, kyn), dtype=complex)

        if line_counter > 1:
            els = line.split(' ')[:-1]
            for j in range(len(els)):
                rho_q[line_counter-2, j] = complex(els[j])

    f.close()

    return rho_q


def write_2_band_rho_q(kxn, kyn, rho_q, version_number):
    # filepath = FilePaths.rho_cache + 'rho_q_2-band_kxn={}_kyn={}_{}_.dat'.format(kxn, kyn, version_number)
    # filepath = FilePaths.rho_cache + 'dEf={}.dat'.format(Params.delta_E_f)
    #
    # appends_val_to_file(filepath, kxn)
    # appends_val_to_file(filepath, kyn)
    # append_matrix_to_file(filepath, rho_q)
    filepath = 'temp/Saeed/rho_q_2-band_kxn={}_kyn={}_{}eV_{}_.dat'.format(kxn, kyn, Params.delta_E_f, version_number)
    append_matrix_as_column(filepath, rho_q)


def read_2_band_rho_q(kxn, kyn, version_number):
    # filepath = FilePaths.rho_cache + 'rho_q_2-band_kxn={}_kyn={}_{}_.dat'.format(kxn, kyn, version_number)
    filepath = FilePaths.rho_cache + 'dEf={}.dat'.format(Params.delta_E_f)

    kxn_file = None
    kyn_file = None

    rho_q = None

    f = open(filepath, 'r')

    line_counter = -1
    for line in f:
        line_counter += 1
        if line_counter == 0:
            kxn_file = int(line)
        if line_counter == 1:
            kyn_file = int(line)
            rho_q = np.zeros((kxn, kyn), dtype=complex)

        if line_counter > 1:
            els = line.split(' ')[:-1]
            for j in range(len(els)):
                rho_q[line_counter-2, j] = complex(els[j])

    f.close()

    return rho_q


def append_run_to_history_file(version_number):
    from datetime import datetime
    f = open(FilePaths.run_history, 'a')
    now = datetime.now()
    time = now.strftime("%H:%M:%S")
    date = now.strftime('%Y/%m/%d')
    f.write('{} {} {}\n'.format(time, date, version_number))
    f.close()

import numpy as np
from scipy.fftpack import fft2
from scipy.fftpack import ifft2


def flatten_mesh_of_matrices(m):
    '''
    transforms a matrix (mesh) of matrices into matrix of matrices (meshes)
    :param m: matrix of matrices
    :return:
    '''
    kx_n = m.shape[0]
    ky_n = m.shape[1]

    element_shape_x = m[0, 0].shape[0]
    element_shape_y = m[0, 0].shape[1]

    # make empty matrix of meshes
    g = np.matrix(np.zeros((element_shape_x, element_shape_y), dtype=np.matrix))

    # invert mesh of matrices to matrix of meshes
    for i in range(element_shape_x):  # Number of "Pixels" along the kx axis
        for j in range(element_shape_y):  # Number of "Pixels" along the ky axis
            g[i, j] = np.matrix(np.zeros((kx_n, ky_n), dtype=complex))  # new blank mesh
            # populate new blank mesh
            for k in range(kx_n):
                for l in range(ky_n):
                    g[i, j][k, l] = m[k, l][i, j]

    return g


def unflatten_to_mesh_of_matrices(m):
    '''
    "unflattens" matrix of meshes to mesh of matrices (at each mesh point there is a matrix)
    :param m: matrix of meshes
    :return:
    '''

    m = np.matrix(m)

    kx_n = m[0, 0].shape[0]
    ky_n = m[0, 0].shape[1]

    element_shape_x = m.shape[0]
    element_shape_y = m.shape[1]

    gs = np.matrix(np.zeros((kx_n, ky_n), dtype=np.matrix))

    for i in range(kx_n):  # Number of "Pixels" along the kx axis
        for j in range(ky_n):  # Number of "Pixels" along the ky axis
            gs[i, j] = np.matrix(
                np.zeros((element_shape_x, element_shape_y), dtype=complex))  # new blank matrix at (kx,ky)
            #  populate new () blank matrix at (kx, ky)
            for k in range(element_shape_x):
                for l in range(element_shape_y):
                    gs[i, j][k, l] = m[k, l][i, j]

    return gs


def fftransform(x):
    return fft2(x)


def ifftransform(x):
    return ifft2(x)


def fft_of_mesh_of_matrices(xs):
    '''
    g_ij = FFT [x_ij]
    where x and g are matrix elements of the the matrices at each mesh point
    don't know the dimensions of the matrices at each mesh point, but have to assume there are of the same dimensions at
    each mesh point
    :param xs:
    :return:
    '''
    xs_flat = flatten_mesh_of_matrices(xs)

    element_shape_x = xs_flat.shape[0]
    element_shape_y = xs_flat.shape[1]

    xs_flat_fft = np.zeros((element_shape_x, element_shape_y), dtype=np.matrix)

    for i in range(element_shape_x):
        for j in range(element_shape_y):
            xs_flat_fft[i, j] = fftransform(xs_flat[i, j])

    xs_fft = unflatten_to_mesh_of_matrices(xs_flat_fft)
    return xs_fft


def ifft_of_mesh_of_matrices(xs):
    '''
    g_ij = IFFT [x_ij]
    where x and g are matrix elements of the the matrices at each mesh point
    don't know the dimensions of the matrices at each mesh point, but have to assume there are of the same dimensions at
    each mesh point
    :param xs:
    :return:
    '''
    xs_flat = flatten_mesh_of_matrices(xs)

    element_shape_x = xs_flat.shape[0]
    element_shape_y = xs_flat.shape[1]

    xs_flat_fft = np.zeros((element_shape_x, element_shape_y), dtype=np.matrix)

    for i in range(element_shape_x):
        for j in range(element_shape_y):
            xs_flat_fft[i, j] = ifftransform(xs_flat[i, j])

    xs_fft = unflatten_to_mesh_of_matrices(xs_flat_fft)
    return xs_fft

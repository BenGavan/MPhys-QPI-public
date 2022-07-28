# import subprocess
import numpy as np


# def copy2clip(txt):
#     cmd = 'echo '+txt.strip()+'|pbcopy'
#     return subprocess.check_call(cmd, shell=True)


def generate_axes(x_n, y_n, x_length, y_length):
    '''

    :param x_n: # pixes along x axis
    :param y_n: # pixes along y axis
    :param x_length:
    :param y_length:
    :return:
    '''
    xs = np.linspace(-x_length / 2, x_length / 2, x_n)
    ys = np.linspace(-y_length / 2, y_length / 2, y_n)

    return xs, ys

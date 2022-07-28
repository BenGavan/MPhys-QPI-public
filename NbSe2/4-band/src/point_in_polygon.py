import matplotlib.pyplot as plt
import numpy as np

# Crossing number test for a point in a polygon


class Point:
    x = None
    y = None

    def __init__(self, x, y):
        self.x = x
        self.y = y


# ps = ps[n+1] with ps[n]=p[0]
def cn_poly(p: Point, ps: [Point]):
    cn = 0  # crossing number counter

    for i in range(len(ps)-1):
        if ((ps[i].y <= p.y) and (ps[i+1].y > p.y)) or ((ps[i].y > p.y) and (ps[i+1].y <= p.y)):  # 1) upward crossing, 2) downward crossing
            vt = (p.y - ps[i].y) / (ps[i+1].y - ps[i].y)   # the actual edge-ray intersect x-coordinate
            if p.x < ps[i].x + vt * (ps[i+1].x - ps[i].x):
                cn += 1
    return cn


def is_point_in_poly(p: Point, ps: [Point]):
    return (cn_poly(p, ps) % 2) == 1


def plot(p: Point,  ps: [Point], filepath: str):
    is_in_poly = is_point_in_poly(p, ps)

    print('is point in poly: ', is_in_poly)

    xs = []
    ys = []
    for point in ps:
        xs.append(point.x)
        ys.append(point.y)

    print(xs, ys)

    plt.plot(xs, ys)
    plt.xlabel('x')
    plt.ylabel('y')

    p_fmt = 'rx'
    if is_in_poly:
        p_fmt = 'gx'

    plt.plot([p.x], [p.y], p_fmt)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(filepath)
    print('Point in Polygon plot:\n{}'.format(filepath))
    # plt.show()


class Params:
    a_0 = 3.44  # \AA (Angstrom)
    a_1 = a_0 * np.array([np.sqrt(3) / 2, -1 / 2])
    a_2 = a_0 * np.array([0, 1])

    b_1 = (2 * np.pi / a_0) * np.array([2 * np.sqrt(3) / 3, 0])
    b_2 = (2 * np.pi / a_0) * np.array([np.sqrt(3) / 3, 1])


def run():
    poly_points = [Point(0, 0), Point(1, 2), Point(2, 0), Point(2, 2), Point(1, 3), Point(1, 1), Point(0.7, 1), Point(0.7, 2)]

    poly_points.append(poly_points[0])

    test_point = Point(0.5, 0)

    print(poly_points)

    x = cn_poly(test_point, poly_points)
    print(x)

    plot(test_point, poly_points)



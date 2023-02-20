import numpy as np
from scipy.spatial import ConvexHull

r_max = 24
d_max = 255 * 255


def convex_hull(rd):
    r_min = np.min(rd[:, 1])
    d_min = np.min(rd[:, 2])

    hull_bounds = np.array([
        [-1, r_min, d_max],
        [-1, r_max, d_max],
        [-1, r_max, d_min]
    ])
    rd = np.vstack((rd, hull_bounds))

    hull = ConvexHull(rd[:, 1:3])

    rd = rd[hull.vertices]
    rd = rd[rd[:, 0] >= 0]

    return rd


def choose_best_point_in_rd_hull(rd_hull, lambda_):
    cost = rd_hull[:, 2] + lambda_ * rd_hull[:, 1]  # J = D + lambda * R
    pos = cost.argmin()
    return rd_hull[pos, :]


def calculate_bpp(rd_hulls, lambda_):
    chosen_points = [choose_best_point_in_rd_hull(rd_hull, lambda_) for rd_hull in rd_hulls]
    rate = np.array(chosen_points)[:, 1].mean()

    return chosen_points, rate


def bisection(rd_hulls, target_bpp, lambda_a=0.01, lambda_b=1000.0, eps=0.01):
    best_rd_points = []

    while lambda_b - lambda_a > eps:
        lambda_ = (lambda_a + lambda_b) / 2
        chosen_rd_points, bpp = calculate_bpp(rd_hulls, lambda_)

        if bpp > target_bpp:
            lambda_a = lambda_
        else:
            best_rd_points = chosen_rd_points
            lambda_b = lambda_

    mode_ids = [int(point[0]) for point in best_rd_points]

    return mode_ids

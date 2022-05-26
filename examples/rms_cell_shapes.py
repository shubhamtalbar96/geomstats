import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import geomstats.backend as gs
import geomstats.datasets.utils as data_utils
from geomstats.geometry.discrete_curves import R2, DiscreteCurves, SRVMetric
from geomstats.geometry.pre_shape import PreShapeSpace
from geomstats.learning.frechet_mean import FrechetMean
from geomstats.learning.geometric_median import WeiszfeldAlgorithm

gs.random.seed(2021)

cells, lines, treatments = data_utils.load_cells()
print(f"Total number of cells : {len(cells)}")

TREATMENTS = gs.unique(treatments)
print(TREATMENTS)

LINES = gs.unique(lines)
print(LINES)

# normal_cell_idx = 1
# plt.plot(cells[normal_cell_idx][:, 0], cells[normal_cell_idx][:, 1], "blue")
# plt.plot(cells[normal_cell_idx][0, 0], cells[normal_cell_idx][0, 1], "blue", marker="o")

ds = {}

n_cells_arr = gs.zeros((3, 2))

for i, treatment in enumerate(TREATMENTS):
    print(f"{treatment} :")
    ds[treatment] = {}
    for j, line in enumerate(LINES):
        to_keep = gs.array(
            [
                one_treatment == treatment and one_line == line
                for one_treatment, one_line in zip(treatments, lines)
            ]
        )
        ds[treatment][line] = [
            cell_i for cell_i, to_keep_i in zip(cells, to_keep) if to_keep_i
        ]
        nb = len(ds[treatment][line])
        print(f"\t {nb} {line}")
        n_cells_arr[i, j] = nb

n_cells_df = pd.DataFrame({"dlm8": n_cells_arr[:, 0], "dunn": n_cells_arr[:, 1]})
n_cells_df = n_cells_df.set_index(TREATMENTS)

len(ds["cytd"]["dlm8"])


def apply_func_to_ds(input_ds, func):
    """Apply the input function func to the input dictionnary input_ds.

    This function goes through the dictionnary structure and applies
    func to every cell in input_ds[treatment][line].

    It stores the result in a dictionnary output_ds that is returned
    to the user.

    Parameters
    ----------
    input_ds : dict
        Input dictionnary, with keys treatment-line.
    func : callable
        Function to be applied to the values of the dictionnary, i.e.
        the cells.

    Returns
    -------
    output_ds : dict
        Output dictionnary, with the same keys as input_ds.
    """
    output_ds = {}
    for treatment in TREATMENTS:
        output_ds[treatment] = {}
        for line in LINES:
            output_list = []
            for one_cell in input_ds[treatment][line]:
                output_list.append(func(one_cell))
            output_ds[treatment][line] = gs.array(output_list)
    return output_ds


def interpolate(curve, nb_points):
    """Interpolate a discrete curve with nb_points from a discrete curve.

    Returns
    -------
    interpolation : discrete curve with nb_points points
    """
    old_length = curve.shape[0]
    interpolation = gs.zeros((nb_points, 2))
    incr = old_length / nb_points
    pos = 0
    for i in range(nb_points):
        index = int(gs.floor(pos))
        interpolation[i] = curve[index] + (pos - index) * (
            curve[(index + 1) % old_length] - curve[index]
        )
        pos += incr
    return interpolation


N_SAMPLING_POINTS = 200

cell_rand = cells[62]
cell_interpolation = interpolate(cell_rand, N_SAMPLING_POINTS)

# fig = plt.figure(figsize=(15, 5))
#
# fig.add_subplot(121)
# plt.plot(cell_rand[:, 0], cell_rand[:, 1])
# plt.axis("equal")
# plt.title(f"Original curve ({len(cell_rand)} points)")
# plt.axis("off")
#
# fig.add_subplot(122)
# plt.plot(cell_interpolation[:, 0], cell_interpolation[:, 1])
# plt.axis("equal")
# plt.title(f"Interpolated curve ({N_SAMPLING_POINTS} points)")
# plt.axis("off")

ds_interp = apply_func_to_ds(
    input_ds=ds, func=lambda x: interpolate(x, N_SAMPLING_POINTS)
)

print(ds_interp["control"]["dunn"].shape)

# n_cells_to_plot = 10
# fig = plt.figure(figsize=(16, 6))
# count = 1
# for treatment in TREATMENTS:
#     for line in LINES:
#         cell_data = ds_interp[treatment][line]
#         for i_to_plot in range(n_cells_to_plot):
#             cell = gs.random.choice(cell_data)
#             fig.add_subplot(3, 2 * n_cells_to_plot, count)
#             count += 1
#             plt.plot(cell[:, 0], cell[:, 1], color="C" + str(int((line == "dunn"))))
#             plt.axis("equal")
#             plt.axis("off")
#             if i_to_plot == n_cells_to_plot // 2:
#                 plt.title(f"{treatment}   -   {line}", fontsize=20)

M_AMBIENT = 2

PRESHAPE_SPACE = PreShapeSpace(m_ambient=M_AMBIENT, k_landmarks=N_SAMPLING_POINTS)
PRESHAPE_METRIC = PRESHAPE_SPACE.embedding_metric


def preprocess(curve, tol):
    """Preprocess curve to ensure that there are no consecutive duplicate points.

    Returns
    -------
    curve : discrete curve
    """

    dist = curve[1:] - curve[:-1]
    dist_norm = np.sqrt(np.sum(np.square(dist), axis=1))

    if np.any(dist_norm < tol):
        for i in range(len(curve) - 1):
            if np.sqrt(np.sum(np.square(curve[i + 1] - curve[i]), axis=0)) < tol:
                curve[i + 1] = (curve[i] + curve[i + 2]) / 2

    return curve


def exhaustive_align(curve, base_curve):
    """Align curve to base_curve to minimize the L² distance.

    Returns
    -------
    aligned_curve : discrete curve
    """
    nb_sampling = len(curve)
    distances = gs.zeros(nb_sampling)
    for shift in range(nb_sampling):
        reparametrized = [curve[(i + shift) % nb_sampling] for i in range(nb_sampling)]
        aligned = PRESHAPE_SPACE.align(point=reparametrized, base_point=base_curve)
        distances[shift] = PRESHAPE_METRIC.norm(
            gs.array(aligned) - gs.array(base_curve)
        )
    shift_min = gs.argmin(distances)
    reparametrized_min = [
        curve[(i + shift_min) % nb_sampling] for i in range(nb_sampling)
    ]
    aligned_curve = PRESHAPE_SPACE.align(
        point=reparametrized_min, base_point=base_curve
    )
    return aligned_curve


TOL = 1e-10


ds_proc = apply_func_to_ds(ds_interp, func=lambda x: preprocess(x, TOL))
print(ds_proc["control"]["dunn"].shape)

ds_proj = apply_func_to_ds(ds_proc, func=PRESHAPE_SPACE.projection)
print(ds_proj["control"]["dunn"].shape)

BASE_CURVE = ds_proj["control"]["dunn"][0]
print("Shape of BASE_CURVE:", BASE_CURVE.shape)

ds_align = apply_func_to_ds(ds_proj, func=lambda x: exhaustive_align(x, BASE_CURVE))
print(ds_align["control"]["dunn"].shape)

# i_rand = gs.random.randint(n_cells_df.loc["control"]["dunn"])
# unaligned_cell = ds_proj["control"]["dunn"][i_rand]
# aligned_cell = ds_align["control"]["dunn"][i_rand]

# fig = plt.figure(figsize=(15, 5))
#
# fig.add_subplot(131)
# plt.plot(BASE_CURVE[:, 0], BASE_CURVE[:, 1])
# plt.plot(BASE_CURVE[0, 0], BASE_CURVE[0, 1], "ro")
# plt.axis("equal")
# plt.title("Reference curve")
#
# fig.add_subplot(132)
# plt.plot(unaligned_cell[:, 0], unaligned_cell[:, 1])
# plt.plot(unaligned_cell[0, 0], unaligned_cell[0, 1], "ro")
# plt.axis("equal")
# plt.title("Unaligned curve")
#
# fig.add_subplot(133)
# plt.plot(aligned_cell[:, 0], aligned_cell[:, 1])
# plt.plot(aligned_cell[0, 0], aligned_cell[0, 1], "ro")
# plt.axis("equal")
# plt.title("Aligned curve")

cell_shapes_list = []
for treatment in TREATMENTS:
    for line in LINES:
        cell_shapes_list.extend(ds_align[treatment][line])

cell_shapes = gs.array(cell_shapes_list)
print(cell_shapes.shape)

CURVES_SPACE = DiscreteCurves(R2)
SRV_METRIC = CURVES_SPACE.square_root_velocity_metric

# mean = FrechetMean(metric=SRV_METRIC, point_type="matrix", method="default")
# mean.fit(cell_shapes[:600])
#
# mean_estimate = mean.estimate_
# plt.plot(mean_estimate[:, 0], mean_estimate[:, 1], "black")

num_of_cells = 50

# median = WeiszfeldAlgorithm(metric=SRV_METRIC)
# median.fit(cell_shapes[:num_of_cells])
# median_estimate = median.estimate_

median = WeiszfeldAlgorithm(metric=SRV_METRIC)
median_estimate = median.brute_force(cell_shapes[:num_of_cells])

plt.plot(median_estimate[:, 0], median_estimate[:, 1], "blue")
plt.plot(median_estimate[:, 0], median_estimate[:, 1], "blue")

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.linear_model import RANSACRegressor
from sympy.physics.units import length


def extract_centroid_3d(record_3ds,
                        delta_t=1e-9,
                        N_bin=4000,
                        pulse_of_hist=1000,
                        window_size=5,
                        count_threshold=10):
    """
    record_3ds: (N_row, N_col, N_total)
                N_total = N_pulse * hist_pulse
    """

    record_3ds = np.asarray(record_3ds)
    N_row, N_col, N_pulse = record_3ds.shape

    N_frame = N_pulse // pulse_of_hist
    points_output = []

    for n in range(N_frame):

        # -------- Grouping: Retrieve data from the nth frame --------
        start = n * pulse_of_hist
        end = (n + 1) * pulse_of_hist
        pulse_data = record_3ds[:, :, start:end]  # (row, col, hist_pulse)

        for i in range(N_row):
            for j in range(N_col):

                data = pulse_data[i, j, :]   # bin index sequence

                # Only retain valid bins: 1 to N_bin
                valid = (data > 0) & (data <= N_bin)
                data_valid = data[valid]

                if data_valid.size == 0:
                    continue

                hist = np.bincount(data_valid, minlength=N_bin + 1)[1:]

                window_sum = np.convolve(
                    hist, np.ones(window_size, dtype=int), mode='valid'
                )

                candidates = np.where(window_sum >= count_threshold)[0]
                if len(candidates) == 0:
                    continue

                start_bin = candidates[0]
                signal_bins = np.arange(start_bin, start_bin + window_size)
                signal_counts = hist[signal_bins]

                centroid_bin = np.sum(signal_bins * signal_counts) / np.sum(signal_counts)
                depth = centroid_bin * delta_t * 3e8 / 2
                points_output.append([n, j, i, depth])

    points_output = np.array(points_output)
    groups = np.unique(points_output[:, 0]).astype(int)
    k = 3
    # Subsequent handling
    point_range = []
    point_range = []

    for g in groups:
        mask = points_output[:, 0] == g
        pts = points_output[mask]

        # ---------- j, i take the centre ----------
        j_s = pts[:, 1]
        i_s = pts[:, 2]

        j = int(round((j_s.max() + j_s.min()) / 2))
        i = int(round((i_s.max() + i_s.min()) / 2))

        # ---------- depth processing ----------
        depth_s = pts[:, 3]
        depth = max(depth_s)


        point_range.append([g, j, i, depth])

    return np.array(point_range)

import numpy as np

def calculate_RMSE(points, re_points, num_points, pulse_of_hist):
    """
    If missing frames exist in re_points, perform linear interpolation on the (u, v, d) coordinates of the missing frames;
    If no missing frames exist, calculate the RMSE directly (without interpolation).
    """

    # ========= 1. Construct a point_frame =========
    N_frame = num_points // pulse_of_hist
    point_frame = []

    for n in range(N_frame):
        start = n * pulse_of_hist
        end = (n + 1) * pulse_of_hist
        data = points[start:end, :]

        u = np.mean(data[:, 1])
        v = np.mean(data[:, 2])
        d = np.mean(data[:, 3])

        point_frame.append([n, u, v, d])

    point_frame = np.asarray(point_frame)
    re_points = np.asarray(re_points)

    full_frames = point_frame[:, 0].astype(int)
    re_frames = re_points[:, 0].astype(int)

    # ========= 2. Determine whether frames are missing =========
    missing_frames = np.setdiff1d(full_frames, re_frames)

    if len(missing_frames) == 0:
        # ---------- No gaps: Direct alignment ----------
        # Assuming the frame order of re_points matches that of point_frame
        re_points_used = re_points

    else:
        # ---------- Missing: Interpolation only for missing frames ----------
        u_interp = np.interp(full_frames, re_frames, re_points[:, 1])
        v_interp = np.interp(full_frames, re_frames, re_points[:, 2])
        d_interp = np.interp(full_frames, re_frames, re_points[:, 3])

        re_points_used = np.stack(
            [full_frames, u_interp, v_interp, d_interp],
            axis=1
        )

    # ========= 3. Calculate the Root Mean Square Error =========
    error_u = point_frame[:, 1] - re_points_used[:, 1]
    error_v = point_frame[:, 2] - re_points_used[:, 2]
    error_d = point_frame[:, 3] - re_points_used[:, 3]

    RMSE_u = np.sqrt(np.mean(error_u ** 2))
    RMSE_v = np.sqrt(np.mean(error_v ** 2))
    RMSE_d = np.sqrt(np.mean(error_d ** 2))

    return RMSE_u, RMSE_v, RMSE_d, point_frame, re_points_used

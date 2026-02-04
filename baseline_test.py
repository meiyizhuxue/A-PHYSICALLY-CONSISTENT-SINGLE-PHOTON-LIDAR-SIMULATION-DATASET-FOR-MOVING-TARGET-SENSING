import os
import re

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from baseline_algorithm import extract_centroid_3d, calculate_RMSE

import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings(
    "ignore",
    category=UndefinedMetricWarning
)

# ---------------- Folders and parameters ----------------
folder_path = "D:/cu_file/20260101/rbg_400k_data"
points = np.loadtxt(os.path.join(folder_path, "pixels.txt"), skiprows=2)
num_points = 20000
points[:, 0] = np.linspace(1, num_points, num_points)

# Select only batch_*.bin files and sort them naturally by the numbers in their filenames.
def natural_sort_key(s):
    match = re.search(r'(\d+)', s)
    return int(match.group(1)) if match else -1

bin_files = sorted(
    [f for f in os.listdir(folder_path) if f.startswith("batch_") and f.endswith(".bin")],
    key=natural_sort_key
)

N_row = 128
N_col = 128
pulse_of_batch = 1000

# ---------------- Read each file and calculate the depth ----------------
N_total_pulse = len(bin_files) * pulse_of_batch

record_3ds = np.empty(
    (N_row, N_col, N_total_pulse),
    dtype=np.int16
)

pulse_offset = 0

for idx, filename in enumerate(bin_files):
    file_path = os.path.join(folder_path, filename)

    with open(file_path, "rb") as f:
        data = f.read()
        record_data = np.frombuffer(data, dtype=np.int16)
        record_3d = record_data.reshape(N_row, N_col, pulse_of_batch)

        record_3ds[:, :, pulse_offset:pulse_offset + pulse_of_batch] = record_3d
        pulse_offset += pulse_of_batch

print("All files have finished loading!")

delta_t = 1e-9
N_bin = 4000
pulse_of_hist = 100
window_size = 5
count_threshold = 6

# Multi-pulse Histogram Centroid Method
points_centroid = extract_centroid_3d(
    record_3ds,
    delta_t=delta_t,
    N_bin=N_bin,
    pulse_of_hist=pulse_of_hist,
    window_size=window_size,
    count_threshold=count_threshold)

[RMSE_u_centroid, RMSE_v_centroid, RMSE_d_centroid, point_frame, re_points_used] = calculate_RMSE(points, points_centroid, num_points, pulse_of_hist)
print("RMSE_u_centroid:", RMSE_u_centroid)
print("RMSE_v_centroid:", RMSE_v_centroid)
print("RMSE_d_centroid:", RMSE_d_centroid)

# Set Times New Roman as the font
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['mathtext.fontset'] = 'stix'


# Create a graph with 1 row and 3 columns of subgraphs
fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=150)

# Set a uniform font size
fontsize_label = 18
fontsize_legend = 14
fontsize_ticks = 14

color_reconstruct = "#E41A1C"
color_true = "#377EB8"

marker_size = 8
alpha_reconstruct = 0.7
alpha_true = 0.3

# ---------- First image: X-coordinate ----------
axes[0].scatter(
    re_points_used[:, 0],
    re_points_used[:, 1],
    s=marker_size,
    alpha=alpha_reconstruct,
    color=color_reconstruct,
    edgecolors='none',
    label="Reconstructed X"
)

axes[0].scatter(
    point_frame[:, 0],
    point_frame[:, 1],
    s=marker_size,
    alpha=alpha_true,
    color=color_true,
    edgecolors='none',
    label="Ground truth"
)

axes[0].set_ylim(0, 127)
axes[0].set_xlabel("Frame number", fontsize=fontsize_label)
axes[0].set_ylabel("X coordinate(pixel)", fontsize=fontsize_label)
axes[0].tick_params(axis='both', which='major', labelsize=fontsize_ticks)

legend0 = axes[0].legend(loc="upper right", fontsize=fontsize_legend,
                        frameon=True, framealpha=0.9, edgecolor='black')
legend0.get_frame().set_linewidth(0.5)

axes[0].grid(True, linestyle=':', linewidth=0.3, alpha=0.5)

# ---------- Second image: Y-coordinate ----------
axes[1].scatter(
    re_points_used[:, 0],
    re_points_used[:, 2],
    s=marker_size,
    alpha=alpha_reconstruct,
    color=color_reconstruct,
    edgecolors='none',
    label="Reconstructed Y"
)

axes[1].scatter(
    point_frame[:, 0],
    point_frame[:, 2],
    s=marker_size,
    alpha=alpha_true,
    color=color_true,
    edgecolors='none',
    label="Ground truth"
)

axes[1].set_ylim(0, 127)
axes[1].set_xlabel("Frame number", fontsize=fontsize_label)
axes[1].set_ylabel("Y coordinate(pixel)", fontsize=fontsize_label)
axes[1].tick_params(axis='both', which='major', labelsize=fontsize_ticks)

axes[1].legend(loc="upper right", fontsize=fontsize_legend,
               frameon=True, framealpha=0.9, edgecolor='black')
axes[1].grid(True, linestyle=':', linewidth=0.3, alpha=0.5)

# ---------- Third image: Depth ----------
axes[2].scatter(
    re_points_used[:, 0],
    re_points_used[:, 3],
    s=marker_size,
    alpha=alpha_reconstruct,
    color=color_reconstruct,
    edgecolors='none',
    label="Reconstructed Depth"
)

axes[2].scatter(
    point_frame[:, 0],
    point_frame[:, 3],
    s=marker_size,
    alpha=alpha_true,
    color=color_true,
    edgecolors='none',
    label="Ground truth"
)

axes[2].set_ylim(0, 600)
axes[2].set_xlabel("Frame number", fontsize=fontsize_label)
axes[2].set_ylabel("Depth(m)", fontsize=fontsize_label)
axes[2].tick_params(axis='both', which='major', labelsize=fontsize_ticks)

axes[2].legend(loc="upper right", fontsize=fontsize_legend,
               frameon=True, framealpha=0.9, edgecolor='black')
axes[2].grid(True, linestyle=':', linewidth=0.3, alpha=0.5)

for i, label in enumerate(['(a)', '(b)', '(c)']):
    axes[i].text(0.02, 0.98, label, transform=axes[i].transAxes,
                fontsize=12, fontweight='bold',
                verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                         alpha=0.8, edgecolor='none'))

plt.tight_layout(pad=2.0)

# Save
plt.savefig('rbg_400k.png', dpi=300, bbox_inches='tight')
plt.savefig("rbg_400k.pdf", format='pdf', dpi=300, bbox_inches='tight')

plt.show()
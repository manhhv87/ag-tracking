import numpy as np
import matplotlib.pyplot as plt

# Precomputed per-sequence metrics for 4 trackers on (presumably) 8 sequences.
# Keys inside each tracker dictionary are metric names, each mapping to a list
# of per-sequence values. These are used for quick aggregation (mean ± std).
metrics = {
    "Tracktor": {
        "MOTA_values": [93.006, 93.051, 95.622, 94.777, 93.825, 92.863, 93.811, 94.597],
        "HOTA_values": [69.550, 69.779, 68.985, 70.660, 94.299, 91.845, 93.815, 94.768],
        "DetA_values": [99.627, 99.321, 99.145, 98.967, 99.538, 99.438, 99.435, 99.164],
        "IDF1_values": [57.171, 52.513, 55.913, 55.950, 93.496, 90.155, 93.729, 94.570],
    },
    "Tracktor++": {
        "MOTA_values": [96.855, 96.893, 98.249, 97.831, 99.009, 98.125, 97.937, 98.535],
        "HOTA_values": [52.306, 43.609, 65.203, 59.103, 58.887, 49.548, 50.227, 55.688],
        "DetA_values": [99.583, 99.355, 99.114, 99.009, 99.538, 99.438, 99.435, 99.212],
        "IDF1_values": [43.719, 29.000, 58.368, 49.651, 36.687, 26.757, 27.433, 33.010],
    },
    "PlantTracktor": {
        "MOTA_values": [94.722, 95.271, 94.793, 93.365, 96.249, 95.867, 96.036, 95.989],
        "HOTA_values": [72.775, 71.313, 71.298, 70.829, 96.273, 95.891, 96.092, 96.037],
        "DetA_values": [95.676, 96.525, 95.310, 94.281, 96.253, 95.874, 96.045, 96.028],
        "IDF1_values": [60.415, 60.577, 60.008, 60.352, 95.069, 95.893, 95.983, 95.949],
    },  # 60, 54, 60, 57, 98, 97, 97, 97
    "PlantTracktor+": {
        "MOTA_values": [98.120, 98.078, 97.341, 95.377, 98.536, 98.173, 98.173, 98.087],
        "HOTA_values": [98.138, 98.940, 74.112, 72.608, 98.524, 98.199, 98.199, 97.104],
        "DetA_values": [99.123, 99.117, 98.888, 96.363, 99.559, 99.357, 99.357, 99.165],
        "IDF1_values": [98.560, 98.420, 61.295, 58.055, 98.716, 98.496, 98.496, 98.509],
    },
}


def plot_metric(t, data, metric, title, xlabel, ylabel, fontsize=20):
    """
    Plot one metric against a parameter grid with multiple series overlaid.

    This function draws line plots for the provided series, sharing the same x-grid,
    and applies consistent labels, title, legend, and axes formatting. It is intended
    to be used inside pre-configured subplots (i.e., caller handles plt.subplot).

    Args:
        t (array-like): 1D array of x-values (e.g., thresholds: [0.0, 0.2, ..., 1.0]).
        data (list[list]): Each element is [y_values, style, label], where:
            - y_values (array-like): 1D values for the metric to be plotted vs t.
            - style (str): Matplotlib line/marker style (e.g., "s-", "*-g", ".-r").
            - label (str): Legend label for this series (LaTeX ok, e.g., "$\\lambda_{nms}$").
        metric (str): Metric name (not used inside, kept for readability when calling).
        title (str): Title for the subplot.
        xlabel (str): Label for the x-axis (can include LaTeX math).
        ylabel (str): Label for the y-axis.
        fontsize (int, optional): Base font size for labels/ticks/title. Defaults to 20.

    Effects:
        - Plots each series as a line on the current axes.
        - Sets labels, title, legend, grid, and y-limits to [-5, 105].

    Notes:
        - y-limits assume metric values are percentages (0–100). The buffer (-5, 105)
          avoids clipping markers at extremes.
        - Caller should invoke plt.tight_layout() and plt.savefig(...) as appropriate.
    """
    for dat in data:    
        plt.plot(t, dat[0], dat[1], label=dat[2])   # plot y vs. t with given style and label
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize * 0.8)
    plt.xticks(fontsize=fontsize * 0.8)
    plt.yticks(fontsize=fontsize * 0.8)
    plt.ylim(-5, 105)   # fixed y-range for percentage-like metrics
    plt.legend(fontsize=fontsize * 0.8)
    plt.title(title, fontsize=fontsize)
    plt.grid()  # add background grid for readability


if __name__ == "__main__":
    # ---------------------------
    # First part: summary stats
    # ---------------------------
    # For each tracker and metric list, compute mean ± std across sequences and print.
    for k, v in metrics.items():
        for metric, values in v.items():
            vals = np.array(values)     # convert to ndarray for numeric ops
            print(f"{k} {metric} {vals.mean():.3f} +- {vals.std():.3f}")

    # ---------------------------
    # Second part: parameter sweeps
    # ---------------------------
    # Parameter grid (shared x-axis):
    #   t = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
    # Interpreted as candidate thresholds for NMS, detection, and regression gating.
    t = np.array([0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0])

    # NMS threshold (λ_nms) curves for HOTA/MOTA/IDF1
    hota_nms = np.array([87.719, 91.978, 90.272, 90.778, 91.322, 87.463, 21.741])
    mota_nms = np.array([97.845, 97.736, 97.318, 97.459, 97.655, 97.611, 0.000])
    idf1_nms = np.array([86.530, 88.818, 84.410, 86.579, 86.877, 85.204, 19.677])

    # Detection score threshold (s_new) curves
    hota_dth = np.array([66.048, 88.406, 89.509, 91.978, 87.859, 86.135, 0])
    mota_dth = np.array([81.172, 97.305, 97.434, 97.736, 97.298, 97.374, 0])
    idf1_dth = np.array([73.149, 85.777, 86.403, 88.818, 86.390, 83.888, 0])

    # Regression/association threshold (s_active) curves
    hota_rth = np.array([67.546, 87.362, 89.234, 91.978, 88.508, 87.047, 0])
    mota_rth = np.array([81.545, 97.414, 97.051, 97.736, 96.017, 95.607, 0])
    idf1_rth = np.array([74.297, 86.454, 86.161, 88.818, 86.097, 85.155, 0])

    # ---------------------------
    # Views "by metric": three subplots, each shows one metric vs. all parameters
    # ---------------------------
    plt.figure(figsize=(13, 4))

    # MOTA vs. (λ_nms, s_new, s_active)
    plt.subplot(1, 3, 1)
    plot_metric(
        t,
        [
            [mota_nms, "s-", "$\lambda_{nms}$"],
            [mota_dth, "*-", "$s_{new}$"],
            [mota_rth, ".-", "$s_{active}$"],
        ],
        "MOTA",
        "",
        "Parameter",
        "MOTA",
    )

    # HOTA vs. (λ_nms, s_new, s_active)
    plt.subplot(1, 3, 2)
    plot_metric(
        t,
        [
            [hota_nms, "s-", "$\lambda_{nms}$"],
            [hota_dth, "*-", "$s_{new}$"],
            [hota_rth, ".-", "$s_{active}$"],
        ],
        "HOTA",
        "",
        "Parameter",
        "HOTA",
    )

    # IDF1 vs. (λ_nms, s_new, s_active)
    plt.subplot(1, 3, 3)
    plot_metric(
        t,
        [
            [idf1_nms, "s-", "$\lambda_{nms}$"],
            [idf1_dth, "*-", "$s_{new}$"],
            [idf1_rth, ".-", "$s_{active}$"],
        ],
        "IDF1",
        "",
        "Parameter",
        "IDF1",
    )

    plt.tight_layout()
    plt.savefig("output/LettuceMOT/parameters/parameter_by_metric.jpg")
    plt.savefig("output/LettuceMOT/parameters/parameter_by_metric.eps")

    # ---------------------------
    # Views "by parameter": three subplots, each shows all metrics vs. one parameter
    # ---------------------------
    plt.figure(figsize=(13, 4))

    plt.subplot(1, 3, 1)
    plot_metric(
        t,
        [[mota_nms, "s-", "MOTA"], [hota_nms, "*-", "HOTA"], [idf1_nms, ".-", "IDF1"]],
        "NMS Threshold",
        "",
        "$\lambda_{nms}$",
        "Metric",
    )

    # s_new: compare MOTA/HOTA/IDF1 curves
    plt.subplot(1, 3, 2)
    plot_metric(
        t,
        [[mota_dth, "s-", "MOTA"], [hota_dth, "*-", "HOTA"], [idf1_dth, ".-", "IDF1"]],
        "Detection Threshold",
        "",
        "$s_{new}$",
        "Metric",
    )

    # s_active: compare MOTA/HOTA/IDF1 curves
    plt.subplot(1, 3, 3)
    plot_metric(
        t,
        [[mota_rth, "s-", "MOTA"], [hota_rth, "*-", "HOTA"], [idf1_rth, ".-", "IDF1"]],
        "Regression Threshold",
        "",
        "$s_{active}$",
        "Metric",
    )
    plt.tight_layout()
    plt.savefig("output/LettuceMOT/parameters/metric_by_parameter.jpg")
    plt.savefig("output/LettuceMOT/parameters/metric_by_parameter.eps")

    # ---------------------------
    # Full grid (3×3): each subplot is a single (metric, parameter) pairing
    # ---------------------------
    plt.figure(figsize=(13, 10))

    # Row 1: MOTA vs. each parameter (λ_nms, s_new, s_active)
    plt.subplot(3, 3, 1)
    plot_metric(
        t,
        [[mota_nms, "s-b", "MOTA"]],
        "MOTA",
        "MOTA vs. NMS Threshold",
        "$\lambda_{nms}$",
        "",
    )
    plt.subplot(3, 3, 2)
    plot_metric(
        t,
        [[mota_dth, "*-g", "MOTA"]],
        "MOTA",
        "MOTA vs. Detection Threshold",
        "$s_{new}$",
        "",
    )
    plt.subplot(3, 3, 3)
    plot_metric(
        t,
        [[mota_rth, ".-r", "MOTA"]],
        "MOTA",
        "MOTA vs. Regression Threshold",
        "$s_{active}$",
        "",
    )

    # Row 2: HOTA vs. each parameter
    plt.subplot(3, 3, 4)
    plot_metric(
        t,
        [[hota_nms, "s-b", "HOTA"]],
        "HOTA",
        "HOTA vs. NMS Threshold",
        "$\lambda_{nms}$",
        "",
    )
    plt.subplot(3, 3, 5)
    plot_metric(
        t,
        [[hota_dth, "*-g", "HOTA"]],
        "HOTA",
        "HOTA vs. Detection Threshold",
        "$s_{new}$",
        "",
    )
    plt.subplot(3, 3, 6)
    plot_metric(
        t,
        [[hota_rth, ".-r", "HOTA"]],
        "HOTA",
        "HOTA vs. Regression Threshold",
        "$s_{active}$",
        "",
    )

    # Row 3: IDF1 vs. each parameter
    plt.subplot(3, 3, 7)
    plot_metric(
        t,
        [[idf1_nms, "s-b", "IDF1"]],
        "IDF1",
        "IDF1 vs. NMS Threshold",
        "$\lambda_{nms}$",
        "",
    )
    plt.subplot(3, 3, 8)
    plot_metric(
        t,
        [[idf1_dth, "*-g", "IDF1"]],
        "IDF1",
        "IDF1 vs. Detection Threshold",
        "$s_{new}$",
        "",
    )
    plt.subplot(3, 3, 9)
    plot_metric(
        t,
        [[idf1_rth, ".-r", "IDF1"]],
        "IDF1",
        "IDF1 vs. Regression Threshold",
        "$s_{active}$",
        "",
    )
    plt.tight_layout()
    plt.savefig("output/LettuceMOT/parameters/all.jpg")
    plt.savefig("output/LettuceMOT/parameters/all.eps")
    plt.show()  # display figures interactively (blocks until windows are closed)

import os
import sys
import numpy as np
from tabulate import tabulate
from multiprocessing import freeze_support

# Add project root to sys.path so `import trackeval` resolves correctly when running this file directly.
# This prepends the parent directory of the current file (i.e., repo root) to Python's import path.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import trackeval  # external TrackEval package (module layout assumed by this script)


def populate_seq_track(
    dict_by_tracker, dict_by_seq, tracker, seq, metricname, metricvalue
):
    """
    Insert a (metric -> value) entry into two nested lookup tables:
    (1) table keyed first by tracker then by sequence, and
    (2) table keyed first by sequence then by tracker.

    The structure built is:
        dict_by_tracker[tracker][seq][metricname] = metricvalue
        dict_by_seq[seq][tracker][metricname] = metricvalue

    If intermediate dictionaries do not exist, they are created.

    Args:
        dict_by_tracker (dict): Nested mapping of the form {tracker: {seq: {metric: val}}}.
        dict_by_seq (dict): Nested mapping of the form {seq: {tracker: {metric: val}}}.
        tracker (str): Tracker identifier/name.
        seq (str): Sequence identifier/name.
        metricname (str): Name of the metric (e.g., "MOTA", "HOTA", "IDF1", ...).
        metricvalue (float or ndarray): Numeric value for the metric. In this code path,
                                        the upstream caller passes a scalar (mean) value.

    Returns:
        tuple(dict, dict): Updated (dict_by_tracker, dict_by_seq) for chaining.
    """
    # Ensure first-level and second-level dicts exist for the tracker→sequence indexing
    if tracker not in dict_by_tracker:
        dict_by_tracker[tracker] = {}
    if seq not in dict_by_tracker[tracker]:
        dict_by_tracker[tracker][seq] = {}
    if metricname not in dict_by_tracker[tracker][seq]:
        dict_by_tracker[tracker][seq][metricname] = metricvalue

    # Ensure first-level and second-level dicts exist for the sequence→tracker indexing
    if seq not in dict_by_seq:
        dict_by_seq[seq] = {}
    if tracker not in dict_by_seq[seq]:
        dict_by_seq[seq][tracker] = {}
    if metricname not in dict_by_seq[seq][tracker]:
        dict_by_seq[seq][tracker][metricname] = metricvalue

    return dict_by_tracker, dict_by_seq


def sequence_data_table(res):
    """
    Traverse TrackEval's nested results structure and aggregate per-sequence metrics.

    Expected `res` layout (high-level):
        res[0]["MotChallenge2DBox"][tracker][seq][class][metric_type][metric_name] = array_like

    For each (tracker, seq, metric_name), this function takes the mean of the metric array
    (e.g., mean across thresholds or frames depending on TrackEval output), then stores it
    in two cross-indexed tables:
        - table_by_sequence: {seq: {tracker: {metric: value}}}
        - table_by_tracker : {tracker: {seq: {metric: value}}}

    Args:
        res (list/dict): Output of `evaluator.evaluate(...)` from TrackEval, containing nested dicts/arrays.

    Returns:
        tuple(dict, dict):
            - table_by_sequence (dict): Aggregated results keyed by sequence first.
            - table_by_tracker  (dict): Aggregated results keyed by tracker first.
    """
    table_by_tracker = {}
    table_by_sequence = {}

    # Navigate the hierarchical results for the MotChallenge2DBox dataset
    for tracker, trackervalues in res[0]["MotChallenge2DBox"].items():
        for seq, seqvalues in trackervalues.items():
            # The next level indexes by object class (e.g., "pedestrian"); values are dicts
            for _, classvalues in seqvalues.items():  # obj-class
                # Next level splits metrics into categories (e.g., "HOTA", "CLEAR", "Identity")
                for _, metrictypevalues in classvalues.items():
                    # Finally, iterate individual metric arrays (e.g., "MOTA", "HOTA", "IDF1", ...)
                    for metricname, metricvalue in metrictypevalues.items():
                        value = np.mean(
                            metricvalue
                        )  # collapse to a single scalar for table display
                        table_by_tracker, table_by_sequence = populate_seq_track(
                            table_by_tracker,
                            table_by_sequence,
                            tracker,
                            seq,
                            metricname,
                            value,
                        )
    return table_by_sequence, table_by_tracker


def render_table(table_results, metrics, sequences, trackers, group_by="trackers"):
    """
    Pretty-print a tabular view of results using `tabulate`.

    You can display either:
        - one row per (Tracker, Sequence) if group_by="trackers", or
        - one row per (Sequence, Tracker) if group_by="sequences".

    The function multiplies metric values by 100 and formats them with two decimals.

    Args:
        table_results (dict): Nested mapping generated by `sequence_data_table`, either
                              {tracker: {seq: {metric: value}}} or {seq: {tracker: {metric: value}}}
                              depending on how it's passed in.
        metrics (list[str]): List of metric names to render as columns.
        sequences (list[str]): Ordered list of sequence names to display.
        trackers (list[str]): Ordered list of tracker names to display.
        group_by (str): Either "trackers" (default) or "sequences" to select the leading column.

    Side effects:
        - Prints a formatted table to stdout.
    """
    table_data = []
    if group_by == "trackers":
        # Leading column is the tracker, then we iterate over sequences
        groups = [trackers, sequences]
        headers = ["Tracker", "Sequences"] + metrics

    else:  # group_by == "sequences":
        # Leading column is the sequence, then we iterate over trackers
        groups = [sequences, trackers]
        headers = ["Sequence", "Tracker"] + metrics

    pre_el1 = (
        None  # remembers the last value of the leading column to draw delimiter rows
    )
    for element1 in groups[0]:
        for element2 in groups[1]:
            # Insert an empty value for the repeated leading key to visually group rows
            if element1 == pre_el1:
                col = ["", element2]
            else:
                # Insert a separator row between different leading groups (except before the first)
                col = [element1, element2]
                if pre_el1 is not None:
                    table_data.append(["-----"] * len(headers))
            pre_el1 = element1

            # Append metric values (converted to percentages, 2 decimals); fallback to "" if missing
            for metric in metrics:
                col.append(
                    "%.2f"
                    % (
                        table_results.get(element1, {})
                        .get(element2, {})
                        .get(metric, "")
                        * 100
                    )
                )
            table_data.append(col)

    # Print the final table with headers
    print("\n\n" + tabulate(table_data, headers=headers) + "\n")


def analyze_custom_track_eval(params, metrics, sequences, trackers):
    """
    Configure TrackEval, run evaluation, collect results, and render summary tables.

    Steps:
      1) Pull default configs from TrackEval for evaluator, dataset, and metrics.
      2) Filter the `params` dict to the keys expected by each config type.
      3) Instantiate an Evaluator and the MotChallenge2DBox dataset wrapper.
      4) Build the metrics list according to `params["METRICS"]` (supports HOTA/CLEAR/Identity/VACE).
      5) Run evaluation, producing a nested results structure.
      6) Aggregate results by (sequence, tracker) and (tracker, sequence).
      7) Print two formatted tables (grouped by sequences, then by trackers).

    Args:
        params (dict): Mixed configuration dictionary; only keys recognized by TrackEval
                       defaults are forwarded (others are ignored). Expected keys include:
                       - GT_FOLDER, GT_LOC_FORMAT, SEQMAP_FOLDER/FILE, TRACKERS_FOLDER, ...
                       - OUTPUT_FOLDER, BENCHMARK, SPLIT_TO_EVAL, CLASSES_TO_EVAL, ...
                       - METRICS, THRESHOLD, USE_PARALLEL, NUM_PARALLEL_CORES, etc.
        metrics (list[str]): Metric names for table columns (e.g., ["MOTA", "HOTA", "IDF1", ...]).
        sequences (list[str]): Sequence names to print in the tables.
        trackers (list[str]): Tracker names to print in the tables.

    Raises:
        Exception: If no metrics are selected (i.e., `metrics_list` ends up empty).
    """
    # Get default configurations from TrackEval
    default_eval_config = trackeval.Evaluator.get_default_eval_config()
    default_dataset_config = (
        trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    )
    default_metrics_config = {
        "METRICS": ["HOTA", "CLEAR", "Identity"],
        "THRESHOLD": 0.5,
    }

    # Filter `params` to only those keys that each config expects
    eval_config = {k: v for k, v in params.items() if k in default_eval_config.keys()}
    dataset_config = {
        k: v for k, v in params.items() if k in default_dataset_config.keys()
    }
    metrics_config = {
        k: v for k, v in params.items() if k in default_metrics_config.keys()
    }

    # Create evaluator and dataset instances
    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]

    # Choose metrics to compute based on the METRICS list in `metrics_config`
    metrics_list = []
    for metric in [
        trackeval.metrics.HOTA,
        trackeval.metrics.CLEAR,
        trackeval.metrics.Identity,
        trackeval.metrics.VACE,  # available in TrackEval; will be included if requested
    ]:
        if metric.get_name() in metrics_config["METRICS"]:
            metrics_list.append(metric(metrics_config))

    if len(metrics_list) == 0:
        raise Exception("No metrics selected for evaluation")

    # Run evaluation across the dataset(s) and metric(s)
    res = evaluator.evaluate(dataset_list, metrics_list)

    # Aggregate results into cross-indexed dicts for table rendering
    dict_by_sequence, dict_by_tracker = sequence_data_table(res)

    # Render the two complementary tables
    render_table(dict_by_sequence, metrics, sequences, trackers, "sequences")
    render_table(dict_by_tracker, metrics, sequences, trackers, "trackers")


def track_eval(cfg, benchmark):
    """
    Convenience wrapper to evaluate a benchmark using settings from a runtime config.

    It builds a `params` dict from `cfg`, lists trackers from `<outdir>/trackers`,
    prepares the sequence list, and redirects stdout to a log file:
        `<outdir>/track_eval_{benchmark}.log`

    Finally, it calls `analyze_custom_track_eval` and closes the log.

    Args:
        cfg (dict): Runtime configuration with at least:
            - "datadir" (str): Root containing MOT-style ground-truth (`{seq}/gt/gt.txt`).
            - "outdir"  (str): Where tracker outputs and evaluation results live.
            - "track_metrics_eval" (list[str]): Metrics for table columns.
            - "sets" (list[str]): Sequence names (without "COMBINED_SEQ").
        benchmark (str): Benchmark name (e.g., "AppleMOT", "LettuceMOT") for logging and params.
    """
    freeze_support()  # important for safe multiprocessing startup on Windows

    # Construct a parameter dictionary. Only keys that match TrackEval defaults will be used.
    params = {
        "GT_FOLDER": cfg["datadir"],
        "GT_LOC_FORMAT": "{gt_folder}/{seq}/gt/gt.txt",
        "SEQMAP_FOLDER": ["data/seqmaps"],
        "SEQMAP_FILE": None,
        "SEQ_INFO": None,
        "TRACKERS_FOLDER": cfg["outdir"] + "trackers",
        "OUTPUT_FOLDER": cfg["outdir"] + "trackers-results",
        "OUTPUT_SUB_FOLDER": "",
        "LOG_ON_ERROR": cfg["outdir"] + "trackers-results/error.log",
        "BENCHMARK": benchmark,
        "SPLIT_TO_EVAL": "all",
        "CLASSES_TO_EVAL": ["pedestrian"],
        "TRACKERS_TO_EVAL": None,
        "TRACKER_SUB_FOLDER": "",
        "TRACKER_DISPLAY_NAMES": None,
        "METRICS": ["HOTA", "CLEAR", "Identity"],
        "THRESHOLD": 0.5,
        "USE_PARALLEL": True,
        "NUM_PARALLEL_CORES": 16,
        "OUTPUT_SUMMARY": True,
        "OUTPUT_DETAILED": True,
        "OUTPUT_EMPTY_CLASSES": True,
        "PRINT_CONFIG": False,
        "PRINT_RESULTS": False,
        "PRINT_ONLY_COMBINED": False,
        "DISPLAY_LESS_PROGRESS": True,
        "PLOT_CURVES": True,
        "TIME_PROGRESS": True,
        "INPUT_AS_ZIP": False,
        "DO_PREPROC": True,
        "SKIP_SPLIT_FOL": True,
        "BREAK_ON_ERROR": True,
        "RETURN_ON_ERROR": False,
    }

    # Columns to display in the printed tables
    metrics = cfg["track_metrics_eval"]
    # Sequence list plus the TrackEval-computed combined row
    sequences = cfg["sets"] + ["COMBINED_SEQ"]
    # Discover tracker result folders under <outdir>/trackers (one subdir per tracker)
    trackers = sorted([tracker for tracker in os.listdir(cfg["outdir"] + "trackers")])

    # Redirect stdout to a persistent log file; subsequent prints go into this file
    print("Saving output to:", cfg["outdir"] + "track_eval_%s.log" % benchmark)
    sys.stdout = open(cfg["outdir"] + "track_eval_%s.log" % benchmark, "w")
    analyze_custom_track_eval(params, metrics, sequences, trackers)
    sys.stdout.close()  # restore/close file handle (stdout will be detached after this)


if __name__ == "__main__":
    # Guard for Windows multiprocessing; ensures child processes import this module safely.
    freeze_support()

    # Standalone parameters when running this file directly (example for AppleMOT layout)
    params = {
        "GT_FOLDER": "data/AppleMOTS/MOT",
        "GT_LOC_FORMAT": "{gt_folder}/{seq}/gt/gt.txt",
        "SEQMAP_FOLDER": ["data/seqmaps"],
        "SEQMAP_FILE": None,
        "SEQ_INFO": None,
        "TRACKERS_FOLDER": "output/AppleMOTS/MOT/trackers",
        "OUTPUT_FOLDER": ["output/AppleMOTS/MOT/trackers-results"],
        "OUTPUT_SUB_FOLDER": "",
        "LOG_ON_ERROR": "output/AppleMOTS/MOT/trackers-results/error.log",
        "BENCHMARK": "AppleMOT",
        "SPLIT_TO_EVAL": "all",
        "CLASSES_TO_EVAL": ["pedestrian"],
        "TRACKERS_TO_EVAL": None,
        "TRACKER_SUB_FOLDER": "",
        "TRACKER_DISPLAY_NAMES": None,
        "METRICS": ["HOTA", "CLEAR", "Identity"],
        "THRESHOLD": 0.5,
        "USE_PARALLEL": True,
        "NUM_PARALLEL_CORES": 16,
        "OUTPUT_SUMMARY": True,
        "OUTPUT_DETAILED": True,
        "OUTPUT_EMPTY_CLASSES": True,
        "PRINT_CONFIG": False,
        "PRINT_RESULTS": False,
        "PRINT_ONLY_COMBINED": False,
        "DISPLAY_LESS_PROGRESS": True,
        "PLOT_CURVES": True,
        "TIME_PROGRESS": True,
        "INPUT_AS_ZIP": False,
        "DO_PREPROC": True,
        "SKIP_SPLIT_FOL": True,
        "BREAK_ON_ERROR": True,
        "RETURN_ON_ERROR": False,
    }

    # Columns to print in the tables (these should match keys TrackEval emits)
    metrics = ["MOTA", "HOTA", "DetA", "AssA", "AssRe", "AssPr", "IDF1", "IDR", "IDP"]

    # Sequence set to display (individual sequences plus the auto-computed combined result)
    sequences = [
        "0001",
        "0002",
        "0003",
        "0004",
        "0005",
        "0006",
        "0007",
        "0008",
        "0010",
        "0011",
        "0012",
        "COMBINED_SEQ",
    ]

    # Which trackers to include in the printed tables (must match folder names in TRACKERS_FOLDER)
    trackers = ["ag", "bytetrack", "clean"]

    # Run the full pipeline using the parameters above and print/format the results
    analyze_custom_track_eval(params, metrics, sequences, trackers)

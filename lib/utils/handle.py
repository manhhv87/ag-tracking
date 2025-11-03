import os
import numpy as np
import matplotlib.pyplot as plt


def create_histograms_from_path(p, bins, range):
    """
    Create per-file histograms and aggregate plots from a directory of numeric text files.

    This function expects a directory `p` that contains multiple plain-text files,
    each of which can be loaded via `np.loadtxt` into a 1D numeric array. For every
    file it:
      1) loads the array,
      2) appends its values to a global accumulator array,
      3) draws a histogram of the file's values and saves it as a JPG named after that file,
      4) clears the matplotlib state.

    After processing all files, it:
      - draws a histogram of *all* aggregated values and saves it as `all.jpg`,
      - clears the figure,
      - draws a simple line plot (`plt.plot(all)`) of the aggregated values and saves it as `plot.jpg`.

    Args:
        p (str): Directory path *ending with a path separator* (e.g., "/path/to/dir/").
                 The code concatenates paths using `p + file`, so if `p` does not end
                 with "/" (or "\\" on Windows), the resulting path will be malformed.
        bins (int or sequence): Number of histogram bins or explicit bin edges (passed to `plt.hist`).
        range (tuple): Lower and upper range of the bins for the histogram (passed to `plt.hist`).

    Caveats:
        - Files are accessed as `p + file`; this assumes `p` already contains the trailing separator.
        - `np.loadtxt` expects each file to be a whitespace- or delimiter-separated numeric text file.
        - No error handling is performed for empty files or load failures.
        - The global accumulator starts as an empty float array (`np.array([])`), so dtypes
          will follow NumPy's upcasting rules when concatenating.
    """
    files = os.listdir(p)           # list all directory entries (files/subdirs) in `p`
    all = np.array([])              # accumulator for all values across files (1D float array)
    for file in files:
        arr = np.loadtxt(p + file)          # load numeric data from text file into a 1D/2D ndarray (expects numeric content)
        all = np.concatenate([all, arr])    # append the new values to the accumulator (flattens if needed)

        # Draw and save a histogram for the current file
        plt.hist(arr, bins, range)              # histogram of this file's values
        plt.savefig(p + file[:-4] + ".jpg")     # save figure, replacing last 4 chars (e.g., ".txt") by ".jpg"
        plt.clf()                               # clear current figure to avoid overlaying next plots
    
    # Draw and save a histogram for the aggregated values across all files
    plt.hist(all, bins, range)
    plt.savefig(p + "all.jpg")
    plt.clf()

    # Draw and save a simple line plot of the aggregated values (value vs. index)
    plt.plot(all)
    plt.savefig(p + "plot.jpg")


def obtain_summary_dict(p):
    """
    Parse a two-line summary file into a metric->value dictionary.

    Expected file layout at `<p>/pedestrian_summary.txt`:
        Line 1: space-separated metric names, e.g., "MOTA HOTA IDF1 ..."
        Line 2: space-separated numeric values, aligned by index with Line 1.

    The function:
      - reads the first two lines,
      - splits them by spaces,
      - casts the values from line 2 to float,
      - returns a dict mapping each metric name (line 1) to its float value (line 2).

    Args:
        p (str): Directory path containing "pedestrian_summary.txt". This function
                 joins using `p + "/pedestrian_summary.txt"` (so it does not require
                 `p` to have a trailing slash).

    Returns:
        dict[str, float]: Dictionary of metric name -> value.

    Caveats:
        - No validation is done if there are fewer than 2 lines or misaligned counts.
        - Trailing newline characters are removed using `[:-1]`, which assumes
          each line ends with exactly one newline.
    """
    file = open(p + "/pedestrian_summary.txt")  # open the summary text file
    lines = file.readlines()                    # read all lines into a list
    line1, line2 = lines[0], lines[1]           # the first two lines hold names and values

    line1 = line1[:-1].split(" ")               # drop trailing newline, split names by spaces
    line2 = line2[:-1].split(" ")               # drop trailing newline, split values by spaces
    
    summary = {}                        # output dict: metric -> float value
    for i, met in enumerate(line1):
        summary[met] = float(line2[i])  # align by index; cast value to float
    return summary


def create_summary_from_csv(p):
    """
    Parse a per-sequence metrics CSV and build a nested dictionary, also printing a LaTeX-friendly row.

    Input CSV path: `<p>/pedestrian_detailed.csv`

    Expected CSV structure (simplified):
        Row 0 (header): "seq,HOTA___50,DetA___50,AssA___50,AssRe___50,AssPr___50,MOTA,MOTP,IDSW,IDF1,IDR,IDP,..."
        Row i>0:        "<SEQ_NAME>,<values...>"

    Behavior:
      * Selects the 50th-percentile (alpha=50) slice for HOTA-family metrics (HOTA, DetA, AssA, AssRe, AssPr)
        by constructing column keys like "HOTA___50".
      * Also includes a set of "countable" metrics that do not depend on alpha: MOTA, MOTP, IDSW, IDF1, IDR, IDP.
      * Reads each subsequent CSV row, collects `seq` and a float array of metric values (scaled by 100).
      * For each sequence, builds a dict of metric->value using the header names to find column positions.
      * Prints (to stdout) each metric per sequence and a compact line of key metrics separated by " & " (useful for tables).
      * Returns a nested dict: {seq_name: {metric_name: value, ...}, ...}.

    Args:
        p (str): Directory path containing "pedestrian_detailed.csv".

    Returns:
        dict[str, dict[str, float]]: Nested mapping from sequence name to a dict of metric values.

    Notes & Caveats:
        - The function multiplies parsed values by 100, assuming the CSV stores them in [0, 1] fractions.
        - It searches column indices by header name using `names.index(...)`; if headers are missing
          or spelled differently, a `ValueError` will be raised.
        - No CSV quoting/parsing library is used; splitting is done by a simple comma `,`,
          so embedded commas in quoted fields would not be handled.
        - `extract_perc = 50` fixes the alpha percentile used for HOTA-family metrics.
    """
    # read file
    file = open(p + "/pedestrian_detailed.csv")  # open CSV with detailed metrics

    # metrics to extract (HOTA family at a fixed alpha-percentile)
    metrics = ["HOTA", "DetA", "AssA", "AssRe", "AssPr"]

    # metrics that do not depend on alpha; add as-is from the CSV headers
    countable_metrics = [
        "MOTA",
        "MOTP",
        "IDSW",
        "IDF1",
        "IDR",
        "IDP",
    ]  # not dependent of alpha
    
    extract_perc = 50  # alpha/100 (we will look for header names suffixed with "___50")

    # Build the expected header names for the alpha-dependent metrics, e.g., "HOTA___50"
    extract_name = [met + "___%d" % extract_perc for met in metrics]  # names
    
    # add countable ones:
    metrics += countable_metrics        # overall metric keys we will expose
    extract_name += countable_metrics   # the corresponding header names to look up

    # initialize sequences and per-sequence values
    seqs = []     # list of sequence names from column 0
    values = []   # list of float arrays (metric values per sequence)

    # iterate over CSV lines
    for i, line in enumerate(file.readlines()):
        fields = line[:-1].split(",")   # strip final newline, split by comma
        if i:
            # data row: first field is sequence name; subsequent fields are numeric
            seqs.append(fields[0])
            values.append([float(val) * 100 for val in fields[1:]]) # scale to percentage space
        else:
            # header row: capture column names for later index lookups
            names = fields[1:]

    values = np.array(values)   # make it a NumPy array for convenient indexing

    # Separate metrics into a nested dict keyed by sequence and metric name
    all = {}    # {seq: {metric: value}}
    for i, seq in enumerate(seqs):
        all[seq] = {}
        for j, met in enumerate(extract_name):
            # Find the column index for the metric header (e.g., "HOTA___50" or "MOTA")
            # and pull the corresponding value for the current sequence.
            all[seq][metrics[j]] = values[i, names.index(met)]
            print(seq, "|", metrics[j], "= %3.3f" % values[i, names.index(met)])

        # Build a compact printable row for common metrics (useful for LaTeX tables)
        pr = ""
        for met in [
            "MOTA",
            "HOTA",
            "DetA",
            "AssA",
            "AssRe",
            "AssPr",
            "IDF1",
            "IDR",
            "IDP",
            "IDSW",
        ]:
            pr += " & %3.3f" % all[seq][met]
        print(pr)
        print("-------------------------------------")
    return all


if __name__ == "__main__":
    pass
    # path = os.getcwd() + "/output/reids/"
    # create_histograms_from_path(path, 25, (0, 2.5))

    path = "TrackEval/data/results_lettuce/exp06"
    create_summary_from_csv(path)

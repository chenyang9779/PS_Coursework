#!/usr/bin/env python3

"""
plot_hotspots.py

Scans the 'results/' folder for .prof files (cProfile output), extracts the
top hotspot functions from each profile, and visualizes them as bar charts.

- For each .prof file, we:
  1. Parse it with pstats.
  2. Sort functions by cumulative time (the total time spent in each function + subcalls).
  3. Extract the top N hotspot functions.
  4. Save a bar chart showing function name vs. cumulative time in seconds.

Usage:
    python plot_hotspots.py
"""

import os
import glob
import pstats
import matplotlib.pyplot as plt


def parse_profile(prof_file, top_n=5):
    """
    Parse a .prof file using pstats, extract the top N hotspot functions
    by cumulative time, and return them as a list of (func_name, cumulative_time).

    pstats structure: p.stats is a dict like:
      (filename, line, funcname) -> (cc, nc, tt, ct, callers)
      where:
        cc = call count
        nc = primitive calls
        tt = total time in function only
        ct = cumulative time including subcalls

    Parameters:
        prof_file (str): Path to the .prof file (cProfile output).
        top_n (int): Number of hotspot functions to extract.

    Returns:
        list of (str, float): [(function_name, cumulative_time), ...] sorted descending.
    """
    p = pstats.Stats(prof_file)
    # Remove long path prefixes & sort by cumulative time descending
    p.strip_dirs().sort_stats("cumulative")

    # p.stats is a dict; each key is (filename, line, funcname); value is (cc, nc, tt, ct, callers)
    calls_data = list(p.stats.items())
    # Sort by ct (cumulative time) in descending order
    calls_data.sort(key=lambda item: item[1][3], reverse=True)

    # Take the top N
    top_calls = calls_data[:20]

    hotspots = []
    for func_info, stats_tuple in top_calls:
        # func_info is a tuple: (filename, line, funcname)
        # stats_tuple is: (cc, nc, tt, ct, callers)
        if "<" not in func_info[2]:
            cumulative_time = stats_tuple[3]
            funcname = f"{func_info[2]}"  # e.g., update_densities

            # Build something like: ("update_densities", 3.45)
            hotspots.append((funcname, cumulative_time))

    return hotspots[:5]


def plot_hotspots(hotspots, out_file, title="Hotspot Functions"):
    """
    Create a bar chart of function name vs. cumulative time.

    Parameters:
        hotspots (list of (str, float)): e.g. [("funcA", 2.3), ("funcB", 1.9), ...]
        out_file (str): Path (including filename) to save the plot.
        title (str): Title for the chart.
    """
    if not hotspots:
        print("[WARN] No hotspots data to plot.")
        return

    # Separate function names & cumulative times
    func_names = [item[0] for item in hotspots]
    cum_times = [item[1] for item in hotspots]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(func_names, cum_times, color="tab:red")
    plt.xlabel("Function Name")
    plt.ylabel("Cumulative Time (s)")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    # Add values on top of each bar
    for bar in bars:
        height = bar.get_height()  # Get the height of the bar
        plt.text(
            bar.get_x() + bar.get_width() / 2,  # X-coordinate (center of bar)
            height,  # Y-coordinate (top of bar)
            f"{height:.2f}s",  # Value to display (formatted to 2 decimals)
            ha="center",  # Horizontal alignment
            va="bottom",  # Vertical alignment
            fontsize=10,  # Font size
        )

    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"[INFO] Saved hotspot bar chart: {out_file}")


def main():
    # Where your .prof files are stored
    results_dir = "results"
    prof_files = glob.glob(os.path.join(results_dir, "*.prof"))
    if not prof_files:
        print("[WARN] No .prof files found in 'results/' directory.")
        return

    # For each .prof file, parse & plot
    for prof_file in prof_files:
        base_name = os.path.splitext(os.path.basename(prof_file))[0]
        # e.g., base_name = "profile_50x50land_ts50"

        # Extract top 5 hotspots
        hotspots = parse_profile(prof_file, top_n=5)

        # Output image name
        out_plot = os.path.join(
            results_dir, f"../plot_hotspots/hotspots_{base_name}.png"
        )

        # Plot & save the chart
        plot_hotspots(hotspots, out_plot, title=f"Hotspot Functions: {base_name}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

"""
plot_usage.py

Reads all usage_* CSV files in the 'results/' directory and generates
line plots showing CPU and memory usage over time.

For each CSV file (e.g. 'usage_50x50land_ts50.csv'), this script:
  1. Parses 'Elapsed Time (s)', 'CPU Usage (%)', and 'Memory (MB)'
  2. Plots CPU and memory usage vs. elapsed time
  3. Saves the plot as a .png in 'results/' named plot_usage_50x50land_ts50.png

Usage:
  python plot_usage.py
"""

import os
import re
import glob
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_usage(usage_csv):
    """
    Reads a usage CSV file (with columns: Elapsed Time (s), CPU Usage (%), Memory (MB))
    and generates a plot of CPU & memory over time, saving it as a PNG.

    Parameters:
        usage_csv (str): Path to the usage CSV file.
    """
    times = []
    cpu_vals = []
    mem_vals = []

    # Read data from CSV
    with open(usage_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert strings to floating-point
            times.append(float(row["Elapsed Time (s)"]))
            cpu_vals.append(float(row["CPU Usage (%)"]))
            mem_vals.append(float(row["Memory (MB)"]))

    # Create the plot
    base_name = os.path.splitext(os.path.basename(usage_csv))[0]
    plt.figure(figsize=(10, 6))
    plt.plot(times, cpu_vals, label="CPU Usage (%)", color="tab:blue")
    plt.plot(times, mem_vals, label="Memory (MB)", color="tab:orange")
    plt.xlabel("Time (s)")
    plt.ylabel("Usage")
    plt.title(f"Resource Usage Over Time\n{base_name}")
    plt.legend(loc="best")

    # Save plot as a PNG in the same directory
    plot_filename = os.path.join(
        os.path.dirname(usage_csv), f"../plot/plot_{base_name}.png"
    )
    plt.savefig(plot_filename, dpi=150)
    plt.close()
    print(f"[INFO] Saved plot to: {plot_filename}")

    return (base_name, max(mem_vals))


def plot_memory(name_mem_tuples, out_filename):
    """
    Plots a bar chart of max memory usage for each run.

    Parameters:
        name_mem_tuples (list of tuples): A list of (run_name, max_memory) pairs.
        out_filename (str): Path (including filename) for saving the bar plot.
    """
    # Separate names (X labels) and memory values (Y data) from the tuples
    names, mem_values = zip(*name_mem_tuples)

    plt.figure(figsize=(10, 6))
    plt.bar(names, mem_values, color="tab:purple")
    plt.xlabel("Run (usage file)")
    plt.ylabel("Max Memory (MB)")
    plt.title("Maximum Memory Usage per Run")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    plt.savefig(out_filename, dpi=150)
    plt.close()
    print(f"[INFO] Saved max memory bar plot to: {out_filename}")


def extract_key(item):
    """
    Return (width, height, time_step) for each item like:
      ("usage_100x100land_ts50", 47.92)
    If it doesn't match the pattern, push it to the end of the sort.
    """
    filename = item[0]
    pattern = r"usage_(\d+)x(\d+)land_ts(\d+)"
    match = re.match(pattern, filename)
    if match:
        width_str, height_str, ts_str = match.groups()
        width, height, ts = int(width_str), int(height_str), int(ts_str)
        return (width, height, ts)
    else:
        return (999999, 999999, 999999)


def extract_key(item):
    """
    Return (width, height, time_step) for each item like:
      ("100x100land.dat", 47.92)
    If it doesn't match the pattern, push it to the end of the sort.
    """
    filename = item[0]
    pattern = r"(\d+)x(\d+)land.dat"
    match = re.match(pattern, filename)
    if match:
        width_str, height_str = match.groups()
        width, height = int(width_str), int(height_str)
        return (width, height)
    else:
        return (999999, 999999)


def plot_landscape_vs_runtime(csv_file, output_file="landscape_runtime_plot.png"):
    """
    Reads a CSV file with columns: Landscape File, Time Step, Runtime (s), Profile File.
    Plots Landscape File (x-axis) vs. Runtime (s) (y-axis), with Time Step as a grouping factor.

    Parameters:
        csv_file (str): Path to the CSV file containing the data.
        output_file (str): Path to save the output plot (e.g., "landscape_runtime_plot.png").
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Ensure Landscape File is treated as a categorical variable for consistent sorting
    df["Landscape File"] = pd.Categorical(df["Landscape File"], ordered=True)

    landscape_runtime = []
    tmp_landscape_file = ""
    tmp_runtime = 0
    for landscape_file, run_time in zip(df["Landscape File"], df["Runtime (s)"]):
        if landscape_file != tmp_landscape_file:
            landscape_runtime.append((landscape_file, run_time))
            tmp_landscape_file = landscape_file
            tmp_runtime = run_time
        elif run_time > tmp_runtime:
            landscape_runtime.pop()
            landscape_runtime.append((landscape_file, run_time))
            tmp_runtime = run_time
        else:
            continue

    sorted_landscpae_runtime = sorted(landscape_runtime, key=extract_key)

    plt.figure(figsize=(12, 6))

    landscape_filename, run_time = zip(*sorted_landscpae_runtime)

    # Plot Runtime for each Time Step as a separate line

    plt.plot(
        landscape_filename,
        run_time,
        marker="o",
    )

    plt.xlabel("Landscape File")
    plt.ylabel("Runtime (s)")
    plt.title("Landscape File vs. Runtime")
    plt.xticks(rotation=45, ha="right")  # Rotate x-axis labels for readability
    plt.grid(True)
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"[INFO] Saved plot to: {output_file}")

    plt.plot(
        landscape_filename,
        np.log(run_time),
        marker="o",
    )

    plt.xlabel("Landscape File")
    plt.ylabel("Runtime (s)")
    plt.title("Landscape File vs. Runtime(log)")
    plt.xticks(rotation=45, ha="right")  # Rotate x-axis labels for readability
    plt.grid(True)
    plt.tight_layout()

    # Save the plot
    plt.savefig("log_" + output_file, dpi=150)
    plt.close()
    print(f"[INFO] Saved plot to: log_{output_file}")


def main():
    # Look for usage_*.csv files in the 'results' folder
    results_dir = "results"
    usage_files = glob.glob(os.path.join(results_dir, "usage_*.csv"))

    if not usage_files:
        print("No usage_*.csv files found in 'results/' directory.")
        return

    name_memories = []
    for usage_csv in usage_files:
        name_memory = plot_usage(usage_csv)
        name_memories.append(name_memory)

    sorted_name_memories = sorted(name_memories, key=extract_key)

    max_mem_plot_path = os.path.join(results_dir, "../plot/plot_max_memory_usage.png")
    plot_memory(sorted_name_memories, max_mem_plot_path)

    plot_landscape_vs_runtime("results/time_step_experiment.csv")


if __name__ == "__main__":
    main()

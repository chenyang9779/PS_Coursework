#!/usr/bin/env python3

"""
run_experiments.py

Automates a performance experiment for the Predator-Prey simulation by:
- Iterating through multiple .dat files in a 'testsuit' folder (e.g., 50x50land.dat).
- Varying only the 'time_step' parameter, keeping other parameters fixed.
- Profiling each run using cProfile, saving the results to a .prof file for analysis.
- Measuring total execution time, storing it in a CSV for reference.

Usage:
    PYTHONPATH=.. python run_experiments.py
"""

import os
import csv
import time
import glob
import psutil


def monitor_resources(process, usage_csv_path, poll_interval=0.5):
    """
    Monitors the CPU and memory usage of the given psutil.Popen process
    at fixed intervals, writing usage data to a separate CSV file.

    Parameters:
        process (psutil.Popen): The running process for the simulation.
        usage_csv_path (str): Path to a CSV file for storing resource usage data.
        poll_interval (float): Frequency (in seconds) to poll resource usage.
    """
    start_time = time.time()

    # Prepare CSV for real-time usage data
    with open(usage_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Elapsed Time (s)", "CPU Usage (%)", "Memory (MB)"])

        # Continuously poll until the process finishes
        while True:
            # Sleep before measuring to get more stable deltas
            time.sleep(poll_interval)

            # Check if process is still running
            if process.poll() is not None:
                break
            # CPU usage since last call to cpu_percent()
            # Dividing by the number of CPU cores for a per-core percentage if desired
            cpu_percent = (
                process.cpu_percent()
            )  # This is overall CPU usage for the process
            mem_info = process.memory_info()
            rss_mb = mem_info.rss / (1024**2)  # Convert bytes to MB

            elapsed = time.time() - start_time
            writer.writerow(
                [f"{elapsed:.2f}", f"{cpu_percent:.2f}", f"{rss_mb:.2f}"]
            )


def run_simulation(landscape_file, time_step, out_csv):
    """
    Runs the Predator-Prey simulation with fixed defaults, varying only 'time_step'.
    - Uses cProfile to produce a .prof file for each run.
    - Collects CPU & memory usage in real time.
    - Measures runtime and appends results to 'out_csv'.

    Parameters:
        landscape_file (str): Path to the .dat file (e.g., 50x50land.dat).
        time_step (int): Interval at which data (averages, maps) is output.
        out_csv (str): Path to the CSV file where results will be stored.

    Returns:
        float: Execution time (seconds) for this run.
    """

    # Construct a unique profile filename. Example:
    #   profile_50x50land_ts100.prof
    landscape_basename = os.path.splitext(os.path.basename(landscape_file))[0]
    results_dir = os.path.dirname(out_csv)
    profile_filename = os.path.join(
        os.path.dirname(out_csv), f"profile_{landscape_basename}_ts{time_step}.prof"
    )

    # Construct a usage CSV for this run's CPU/memory data
    usage_csv_filename = f"usage_{landscape_basename}_ts{time_step}.csv"
    usage_csv_path = os.path.join(results_dir, usage_csv_filename)

    # Build the command to run the simulation under cProfile
    cmd = [
        "python",
        "-m",
        "cProfile",
        "-o",
        profile_filename,  # Output .prof file
        os.path.join("..", "predator_prey", "simulate_predator_prey.py"),
        "--landscape-file",
        landscape_file,
        "--time_step",
        str(time_step),
    ]

    # Measure wall-clock time (exclude cProfile overhead from total if desired).
    start_time = time.time()
    
    # Using psutil to spawn the subprocess for resource monitoring
    process = psutil.Popen(cmd)
    # Start monitoring in parallel
    monitor_resources(process, usage_csv_path, poll_interval=0.1)

    end_time = time.time()
    runtime = end_time - start_time

    # Append results to CSV
    with open(out_csv, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                os.path.basename(landscape_file),
                time_step,
                f"{runtime:.3f}",
                profile_filename,  # Record where the profile data was saved
            ]
        )

    return runtime


def main():
    # Determine the directory containing this script
    current_dir = os.path.dirname(__file__)

    # Set up paths to test suit and results
    testsuit_dir = os.path.join(current_dir, "testsuit")
    results_dir = os.path.join(current_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # The .dat files you want to test
    landscape_files = glob.glob(os.path.join(testsuit_dir, "*.dat"))

    # CSV output file for storing the results
    out_csv = os.path.join(results_dir, "time_step_experiment.csv")

    # Create a fresh CSV with header
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Landscape File",
                "Time Step",
                "Runtime (s)",
                "Profile File",
            ]
        )

    # Define the range of 'time_step' values to test
    time_step_values = [50, 75, 100]

    # Run the simulation for each combination
    for landscape_file in landscape_files:
        for t_step in time_step_values:
            runtime = run_simulation(landscape_file, t_step, out_csv)
            print(
                f"[INFO] {os.path.basename(landscape_file)} | time_step={t_step} => {runtime:.3f} s"
            )


if __name__ == "__main__":
    main()

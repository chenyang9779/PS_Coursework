#!/usr/bin/env python3

"""
generate_landscapes.py

Generates .dat files for all-land predator-prey landscapes of various sizes.
Each file has a width, a height, and all cells set to '1'.

Default output files:
  - 50x50land.dat
  - 100x100land.dat
  - 150x150land.dat
  - 200x200land.dat

Usage:
    python generate_landscapes.py <sizes of landscape (square)>
"""
import os
import argparse

def generate_landscape(width, height, filename):
    """
    Creates a .dat file with the specified width, height, and all cells = 1.

    Parameters:
        width (int): Number of columns (inner cells).
        height (int): Number of rows (inner cells).
        filename (str): Output file name (e.g., '50x50land.dat').
    """
    path = os.path.join("testsuit", filename)
    with open(path, "w") as f:
        # First line: "<width> <height>"
        f.write(f"{width} {height}\n")
        # Build one row of 1s, repeated 'width' times
        row_data = " ".join(["1"] * width)
        # Write that row 'height' times
        for _ in range(height):
            f.write(row_data + "\n")

def main():
    parser = argparse.ArgumentParser(description="Generate .dat files for predator-prey landscapes.")
    parser.add_argument(
        "--sizes",
        default="50,100,150,200",
        help="Comma-separated list of grid sizes (e.g., '50,100,150,200')."
    )
    args = parser.parse_args()

    # Parse the comma-separated sizes into a list of integers
    sizes = [int(s) for s in args.sizes.split(",")]

    for size in sizes:
        filename = f"{size}x{size}land.dat"
        generate_landscape(size, size, filename)
        print(f"Created {filename}")

    # # Generate a 50x50 landscape (all land)
    # generate_landscape(50, 50, "50x50land.dat")
    # print("Created 50x50land.dat")

    # # Generate a 100x100 landscape (all land)
    # generate_landscape(100, 100, "100x100land.dat")
    # print("Created 100x100land.dat")

    # # Generate a 150x150 landscape (all land)
    # generate_landscape(150, 150, "150x150land.dat")
    # print("Created 150x150land.dat")

    # # Generate a 200x200 landscape (all land)
    # generate_landscape(200, 200, "200x200land.dat")
    # print("Created 200x200land.dat")

if __name__ == "__main__":
    main()

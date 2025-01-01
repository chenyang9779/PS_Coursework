"""Predator-prey simulation. Foxes and mice.

Version 3.0, last updated in September 2023.
"""

from predator_prey.arg_parser import (
    parse_command_line_args,
)  # Local module handling command-line arguments
import numpy as np  # Numerical Python for array operations
import random  # Random module for seeding and generating random numbers


def getVersion():
    """
    Print and return the current version of the predator-prey simulation.

    Returns:
        float: The version number (e.g., 3.0).
    """
    print("Predator-prey simulation 3.0")
    return 3.0


def sum(array):
    """
    Sum the elements in the provided array.

    Parameters:
        array (np.ndarray): A NumPy array of numeric values.

    Returns:
        float or int: The sum of all elements in the array.
    """
    return np.sum(array)


def zeros(height, width):
    """
    Return a 2D NumPy array of zeros (dtype=int) of size (height, width).

    Parameters:
        height (int): Number of rows for the array.
        width (int): Number of columns for the array.

    Returns:
        np.ndarray: A 2D array of zeros with integer dtype.
    """
    return np.zeros((height, width), int)


def max(array):
    """
    Return the maximum value from the provided array.

    Parameters:
        array (np.ndarray): A NumPy array of numeric values.

    Returns:
        float or int: The maximum element in the array.
    """
    return np.max(array)


def load_landscape(landscape_file):
    """
    Load the landscape from a file and pad it with a halo of width 1.

    The landscape file should start with two integers (width, height)
    on the first line, followed by the landscape data. Land cells are
    typically represented by 1, water cells by 0.

    Parameters:
        landscape_file (str): Path to the landscape file.

    Returns:
        tuple:
            - landscape (np.ndarray): 2D array of shape (height+2, width+2)
              containing land/water values (with halo).
            - width (int): The original width of the landscape (without halo).
            - height (int): The original height of the landscape (without halo).
    """
    with open(landscape_file, "r") as f:
        width, height = [int(i) for i in f.readline().split()]  # Read width and height
        halo_width = width + 2  # Adjust width for halo cells
        halo_height = height + 2  # Adjust height for halo cells
        landscape = zeros(halo_height, halo_width)  # Initialize array with zeros
        for row, line in enumerate(f.readlines(), start=1):
            # Each line corresponds to a row, pad with zeros on left/right
            landscape[row] = [0] + [int(i) for i in line.split()] + [0]
    return landscape, width, height


def get_num_lands(landscape):
    """
    Count the total number of land cells (value != 0) in the provided 2D array.

    Parameters:
        landscape (np.ndarray): 2D array representing the landscape with halo.

    Returns:
        int: Number of land cells in the landscape.
    """
    num_lands = np.count_nonzero(landscape)
    print("Number of land-only squares: {}".format(num_lands))
    return num_lands


def initialize_densities(landscape, height, width, seed, max_density=5.0):
    """
    Initialize densities for a species (mice or foxes) over a given landscape.

    If `seed` is 0, all densities will be set to 0. Otherwise, for each land cell,
    a random density in [0, max_density] is assigned.

    Parameters:
        landscape (np.ndarray): The landscape array (with halo).
        height (int): The original (inner) height of the landscape (without halo).
        width (int): The original (inner) width of the landscape (without halo).
        seed (int): Random seed to control reproducibility of density assignment.
        max_density (float, optional): Upper limit for random density initialization.

    Returns:
        np.ndarray: A 2D array of float densities matching the size of `landscape`.
    """
    random.seed(seed)  # Set the random seed
    density = landscape.astype(float).copy()  # Make a float copy of the landsscape
    for x in range(1, height + 1):
        for y in range(1, width + 1):
            if seed == 0:
                # If seed is zero, set density to zerp
                density[x, y] = 0
            else:
                # otherwise, assign a random density in land cells
                if landscape[x, y]:
                    density[x, y] = random.uniform(0, max_density)
                else:
                    density[x, y] = 0
    return density


def calculate_neighbours(landscape, height, width):
    """
    Calculate, for each cell, how many of its four direct neighbors (N, S, E, W) are land cells.

    Parameters:
        landscape (np.ndarray): 2D array with halo indicating land/water.
        height (int): The original (inner) height of the landscape (without halo).
        width (int): The original (inner) width of the landscape (without halo).

    Returns:
        np.ndarray: A 2D array of the same shape as `landscape`, where each cell
                    contains the count of neighboring land cells (up to 4).
    """
    num_neighbours = zeros(height + 2, width + 2)
    for x in range(1, height + 1):
        for y in range(1, width + 1):
            num_neighbours[x, y] = (
                landscape[x - 1, y]  # North
                + landscape[x + 1, y]  # South
                + landscape[x, y - 1]  # West
                + landscape[x, y + 1]  # East
            )
    return num_neighbours


def write_averages_to_csv(timestep, time, avg_mice, avg_foxes, filename="averages.csv"):
    """
    Append simulation averages of mice and fox densities to a CSV file.

    If the file doesn't exist, it will be created automatically.

    Parameters:
        timestep (int): The current simulation timestep (integer).
        time (float): The current simulation time in arbitrary units (e.g., seconds).
        avg_mice (float): Average density of mice over land cells.
        avg_foxes (float): Average density of foxes over land cells.
        filename (str, optional): Path to the CSV file for writing.
    """
    with open(filename, "a") as f:
        # Write data in CSV format: Timestep,Time,AvgMice,AvgFoxes
        f.write(f"{timestep},{time},{avg_mice},{avg_foxes}\n")


def write_map_file(filename, landscape, mice_colours, foxes_colours, width, height):
    """
    Write a color map representation (`.ppm` format) of the simulation state.

    For each land cell, the RGB color is determined by the fox density (R channel),
    the mice density (G channel), and a fixed zero for the B channel. For water cells,
    a default color is used (e.g., light blue).

    Parameters:
        filename (str): Path to the output file (should end with `.ppm`).
        landscape (np.ndarray): 2D array (with halo) indicating land/water.
        mice_colours (np.ndarray): 2D array holding computed mouse-based color intensities.
        foxes_colours (np.ndarray): 2D array holding computed fox-based color intensities.
        width (int): The original (inner) width of the landscape (without halo).
        height (int): The original (inner) height of the landscape (without halo).
    """
    with open(filename, "w") as f:
        # PPM header: P3 (plain PPM), width, height, and max color value (255)
        hdr = f"P3\n{width} {height}\n255\n"
        f.write(hdr)
        for x in range(height):
            for y in range(width):
                # If cell is land, write fox/mouse color. If not, write water color.
                if landscape[x + 1, y + 1]:
                    f.write(f"{foxes_colours[x, y]} {mice_colours[x, y]} 0\n")
                else:
                    f.write("0 200 255\n")  # Color for water


def print_info(timestep, time, average_mice_density, average_foxes_density):
    """
    Print simulation status information (timestep, simulation time, and average densities).

    Parameters:
        timestep (int): The current simulation timestep.
        time (float): The current simulation time in arbitrary units.
        average_mice_density (float): Average mice density over land cells.
        average_foxes_density (float): Average fox density over land cells.
    """
    print(
        "Averages. Timestep: {} Time (s): {:.1f} Mice: {:.17f} Foxes: {:.17f}".format(
            timestep, time, average_mice_density, average_foxes_density
        )
    )


def initialize_csv():
    """
    Create or overwrite a CSV file named 'averages.csv' and write the header line.

    The header line includes the column names: Timestep,Time,Mice,Foxes.
    """
    with open("averages.csv", "w") as f:
        hdr = "Timestep,Time,Mice,Foxes\n"
        f.write(hdr)


def update_densities_vec(
    mice_density,
    foxes_density,
    mice_birth_rate,
    foxes_birth_rate,
    mice_death_rate,
    foxes_death_rate,
    mice_diffusion_rate,
    foxes_diffusion_rate,
    num_neighbours,
    landscape,
    delta_t,
):
    """
    Update the population densities of mice and foxes using a reaction-diffusion model.

    Parameters:
    - mice_density: 2D array representing the current density of mice.
    - foxes_density: 2D array representing the current density of foxes.
    - mice_birth_rate: Scalar, birth rate of mice.
    - foxes_birth_rate: Scalar, birth rate of foxes (dependent on mice population).
    - mice_death_rate: Scalar, death rate of mice due to predation.
    - foxes_death_rate: Scalar, natural death rate of foxes.
    - mice_diffusion_rate: Scalar, diffusion rate of mice across the landscape.
    - foxes_diffusion_rate: Scalar, diffusion rate of foxes across the landscape.
    - num_neighbours: 2D array representing the number of neighbors for each cell.
    - landscape: 2D array indicating valid areas (non-zero values indicate valid regions).
    - delta_t: Scalar, time step for the update.

    Returns:
    - new_mice_density: Updated 2D array of mice densities.
    - new_foxes_density: Updated 2D array of fox densities.
    """

    # Create copies of input denties, avoid modifying the original arrays
    new_mice_density = mice_density.copy()  # Copy current mice density
    new_foxes_density = foxes_density.copy()  # Copy current fox density

    # Mask for valid landscape areas(non-zero entries int he landscape array)
    mask_land = landscape != 0

    # Extract the interior regoin (ignore boundary)
    mice_in = mice_density[1:-1, 1:-1]
    fox_in = foxes_density[1:-1, 1:-1]
    land_in = mask_land[1:-1, 1:-1]
    neighbours_in = num_neighbours[1:-1, 1:-1]

    # Compute the neighbour values for mice
    up_m = mice_density[0:-2, 1:-1]
    down_m = mice_density[2:, 1:-1]
    left_m = mice_density[1:-1, 0:-2]
    right_m = mice_density[1:-1, 2:]

    # Laplacian mice population
    lap_mice = (up_m + down_m + left_m + right_m) - (neighbours_in * mice_in)

    # Compute the neighbour values for foxes
    up_f = foxes_density[0:-2, 1:-1]
    down_f = foxes_density[2:, 1:-1]
    left_f = foxes_density[1:-1, 0:-2]
    right_f = foxes_density[1:-1, 2:]

    # Laplacian foxes population
    lap_foxes = (up_f + down_f + left_f + right_f) - (neighbours_in * fox_in)

    # Growth of mice and foxes
    growth_mice = mice_birth_rate * mice_in - (mice_death_rate * mice_in * fox_in)
    growth_fox = (foxes_birth_rate * mice_in * fox_in) - foxes_death_rate * fox_in

    # Change in density for mice and foxes regarding growth and diffusion
    dt_mice = growth_mice + mice_diffusion_rate * lap_mice
    dt_fox = growth_fox + foxes_diffusion_rate * lap_foxes

    # update densities using the computed values
    new_in_mice = mice_in + delta_t * dt_mice
    new_in_fox = fox_in + delta_t * dt_fox

    # Make sure all values are non-negative
    new_in_mice = np.clip(new_in_mice, 0, None)
    new_in_fox = np.clip(new_in_fox, 0, None)

    # Enforce valid areas
    new_in_mice[~land_in] = 0
    new_in_fox[~land_in] = 0

    # Update new densities for the inner region and return the updated values
    new_mice_density[1:-1, 1:-1] = new_in_mice
    new_foxes_density[1:-1, 1:-1] = new_in_fox
    return new_mice_density, new_foxes_density


def get_colour(
    height,
    width,
    landscape,
    max_mice_density,
    mice_density,
    max_foxes_density,
    foxes_density,
    mice_colours,
    foxes_colours,
):
    """
    Compute the mouse and fox color intensities for each land cell,
    scaling by each species' maximum density observed in the grid.

    Parameters:
        height (int): The original (inner) height of the landscape (without halo).
        width (int): The original (inner) width of the landscape (without halo).
        landscape (np.ndarray): The landscape array with halo.
        max_mice_density (float): Maximum mice density encountered.
        mice_density (np.ndarray): Current mice density map (with halo).
        max_foxes_density (float): Maximum fox density encountered.
        foxes_density (np.ndarray): Current fox density map (with halo).
        mice_colours (np.ndarray): 2D array that will store mouse color intensities.
        foxes_colours (np.ndarray): 2D array that will store fox color intensities.

    Returns:
        tuple: The updated (mice_colours, foxes_colours) arrays with intensity values in [0,255].
    """
    for x in range(1, height + 1):
        for y in range(1, width + 1):
            if landscape[x, y]:
                # Calculate mice color based on fraction of max density
                if max_mice_density != 0:
                    mcol = (mice_density[x, y] / max_mice_density) * 255
                else:
                    mcol = 0

                # Calculate foxes color based on fraction of max density
                if max_foxes_density != 0:
                    fcol = (foxes_density[x, y] / max_foxes_density) * 255
                else:
                    fcol = 0

                mice_colours[x - 1, y - 1] = mcol
                foxes_colours[x - 1, y - 1] = fcol
    return mice_colours, foxes_colours


def run_predator_prey_simulation(
    mice_birth_rate,
    mice_death_rate,
    mice_diffusion_rate,
    foxes_birth_rate,
    foxes_death_rate,
    foxes_diffusion_rate,
    delta_t,
    time_step,
    duration,
    landscape_file,
    mouse_seed,
    fox_seed,
):
    """
    Main routine to run the predator-prey simulation with the given parameters.

    This function performs the following:
    1. Loads the landscape and initializes the arrays.
    2. Calculates the number of land neighbors for each cell.
    3. Initializes mice and fox densities based on landscape and random seeds.
    4. Repeatedly updates the densities over the specified duration.
    5. Periodically logs averages to CSV and outputs a `.ppm` map for visualization.

    Parameters:
        mice_birth_rate (float): Birth rate for mice.
        mice_death_rate (float): Death rate for mice.
        mice_diffusion_rate (float): Diffusion rate for mice.
        foxes_birth_rate (float): Birth rate for foxes.
        foxes_death_rate (float): Death rate for foxes.
        foxes_diffusion_rate (float): Diffusion rate for foxes.
        delta_t (float): Time step size.
        time_step (int): Interval (in number of timesteps) at which data is logged/printed.
        duration (float): Total simulation duration in same units as delta_t.
        landscape_file (str): Path to the file describing the landscape.
        mouse_seed (int): Random seed for mice density initialization.
        fox_seed (int): Random seed for foxes density initialization.

    Returns:
        None
    """

    # 1. Load the landscape
    landscape, width, height = load_landscape(landscape_file=landscape_file)

    # 2. Count total land cells
    num_lands = get_num_lands(landscape)

    # 3. Calculate the number of land neighbors for each cell
    num_neighbours = calculate_neighbours(landscape, height, width)

    # 4. Initialize mice and fox densities
    mice_density = initialize_densities(landscape, height, width, mouse_seed)
    foxes_density = initialize_densities(landscape, height, width, fox_seed)

    # Prepare arrays for new densities (used each step) and color mappings
    new_mice_density = mice_density.copy()
    new_foxes_density = foxes_density.copy()
    mice_colours = zeros(height, width)
    foxes_colours = zeros(height, width)

    # Compute initial average densities
    if num_lands != 0:
        average_mice_density = sum(mice_density) / num_lands
        average_foxes_density = sum(foxes_density) / num_lands
    else:
        average_mice_density = 0
        average_foxes_density = 0

    # Print initial info and initialize the CSV
    print_info(0, 0, average_mice_density, average_foxes_density)
    initialize_csv()

    # Determine total number of iterations
    total_time_steps = int(duration / delta_t)

    # 5. Main simulation loop
    for i in range(0, total_time_steps):
        # Periodically output data and produce .ppm files
        if not i % time_step:
            max_mice_density = max(mice_density)
            max_foxes_density = max(foxes_density)

            if num_lands != 0:
                average_mice_density = sum(mice_density) / num_lands
                average_foxes_density = sum(foxes_density) / num_lands
            else:
                average_mice_density = 0
                average_foxes_density = 0

            # Print and record simulation data
            print_info(i, i * delta_t, average_mice_density, average_foxes_density)
            write_averages_to_csv(
                i, i * delta_t, average_mice_density, average_foxes_density
            )

            # Convert densities to color values
            mice_colours, foxes_colours = get_colour(
                height,
                width,
                landscape,
                max_mice_density,
                mice_density,
                max_foxes_density,
                foxes_density,
                mice_colours,
                foxes_colours,
            )

            # Write out a PPM map file for visualization
            ppm_file_name = "map_{:04d}.ppm".format(i)
            write_map_file(
                ppm_file_name, landscape, mice_colours, foxes_colours, width, height
            )

        # Update densities for the next timestep
        new_mice_density, new_foxes_density = update_densities_vec(
            mice_density,
            foxes_density,
            mice_birth_rate,
            foxes_birth_rate,
            mice_death_rate,
            foxes_death_rate,
            mice_diffusion_rate,
            foxes_diffusion_rate,
            num_neighbours,
            landscape,
            delta_t,
        )

        # Swap arrays for next iteration.
        tmp = mice_density
        mice_density = new_mice_density
        new_mice_density = tmp

        tmp = foxes_density
        foxes_density = new_foxes_density
        new_foxes_density = tmp


if __name__ == "__main__":
    # Parse arguments from the command line
    args = parse_command_line_args()

    # Print version info
    getVersion()

    # Run the simulation with the parsed arguments
    run_predator_prey_simulation(
        mice_birth_rate=args.birth_mice,
        mice_death_rate=args.death_mice,
        mice_diffusion_rate=args.diffusion_mice,
        foxes_birth_rate=args.birth_foxes,
        foxes_death_rate=args.death_foxes,
        foxes_diffusion_rate=args.diffusion_foxes,
        delta_t=args.delta_t,
        time_step=args.time_step,
        duration=args.duration,
        landscape_file=args.landscape_file,
        mouse_seed=args.mouse_seed,
        fox_seed=args.fox_seed,
    )

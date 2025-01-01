import os
import tempfile
import pytest
from unittest.mock import patch
from predator_prey import simulate_predator_prey


def testGetVersion():
    # Check that getVersion returns the expected value
    ver = simulate_predator_prey.getVersion()
    assert ver == 3.0, "Version should be 3.0"
    assert isinstance(ver, float), "Version should be a float"


def test_parse_command_line_args_defaults():
    # Test that default arguments are set as expected
    test_args = ["simulate_predator_prey.py", "-f", "dummy_landscape.dat"]
    with patch("sys.argv", test_args):
        args = simulate_predator_prey.parse_command_line_args()
        assert args.birth_mice == 0.1, "Default birth rate of mice should be 0.1"
        assert args.death_mice == 0.05, "Default death rate of mice should be 0.05"
        assert (
            args.landscape_file == "dummy_landscape.dat"
        ), "Landscape file should be as passed"


def test_parse_command_line_args_missing_file():
    # Test that missing required arguments cause an error
    test_args = ["simulate_predator_prey.py"]
    with patch("sys.argv", test_args), pytest.raises(SystemExit):
        _ = simulate_predator_prey.parse_command_line_args()


def test_initial_populations_with_seeds():
    # Create a temporary landscape file
    # Minimal landscape: 2x2 all land
    landscape_data = "2 2\n1 1\n1 1\n"
    with tempfile.NamedTemporaryFile(delete=False, mode="w") as tmpfile:
        tmpfile_name = tmpfile.name
        tmpfile.write(landscape_data)

    # Run simulation with mouse_seed=0 and fox_seed=0 and check initial densities
    # If seeds are 0, all densities should be zero.
    args_list = [
        "simulate_predator_prey.py",
        "-f",
        tmpfile_name,
        "--mouse-seed",
        "0",
        "--fox-seed",
        "0",
        "--duration",
        "1",  # short duration
        "--delta-t",
        "0.5",
    ]

    with patch("sys.argv", args_list):
        args = simulate_predator_prey.parse_command_line_args()
        # We don't have direct access to the densities after init without modifying the code.
        # Instead, we can run the simulation and check the averages.csv generated.
        simulate_predator_prey.run_predator_prey_simulation(
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

    # Check that the averages.csv file has zero densities at timestep 0
    with open("averages.csv", "r") as f:
        lines = f.readlines()
        # Header + one line of output for timestep 0
        # Example: Timestep,Time,Mice,Foxes
        #          0,0.0,0.0,0.0
        header = lines[0].strip().split(",")
        assert header == ["Timestep", "Time", "Mice", "Foxes"], "CSV header mismatch"
        first_line = lines[1].strip().split(",")
        mice_density = float(first_line[2])
        fox_density = float(first_line[3])
        assert mice_density == 0.0, "Mice should be initialized to 0.0 if seed=0"
        assert fox_density == 0.0, "Foxes should be initialized to 0.0 if seed=0"

    # Cleanup temporary files
    os.remove(tmpfile_name)
    if os.path.exists("averages.csv"):
        os.remove("averages.csv")
    # Remove any generated ppm files if they exist
    for fn in os.listdir("."):
        if fn.startswith("map_") and fn.endswith(".ppm"):
            os.remove(fn)


def test_initial_populations_nonzero_seeds():
    # With non-zero seeds, we expect some non-zero densities.
    # Create a temporary landscape file
    landscape_data = "2 2\n1 1\n1 1\n"
    with tempfile.NamedTemporaryFile(delete=False, mode="w") as tmpfile:
        tmpfile_name = tmpfile.name
        tmpfile.write(landscape_data)

    args_list = [
        "simulate_predator_prey.py",
        "-f",
        tmpfile_name,
        "--mouse-seed",
        "42",
        "--fox-seed",
        "42",
        "--duration",
        "1",  # short duration
        "--delta-t",
        "0.5",
    ]

    with patch("sys.argv", args_list):
        args = simulate_predator_prey.parse_command_line_args()
        simulate_predator_prey.run_predator_prey_simulation(
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

    # Check that the averages.csv file has some non-zero values at timestep 0
    with open("averages.csv", "r") as f:
        lines = f.readlines()
        # The second line corresponds to timestep=0
        first_line = lines[1].strip().split(",")
        mice_density = float(first_line[2])
        fox_density = float(first_line[3])
        # With random.seed(42) and 2x2 grid, it's very unlikely all densities are zero.
        # We only check that they are not zero.
        assert (
            mice_density > 0.0
        ), "Mice should have non-zero initial density if seed != 0"
        assert (
            fox_density > 0.0
        ), "Foxes should have non-zero initial density if seed != 0"

    # Cleanup
    os.remove(tmpfile_name)
    if os.path.exists("averages.csv"):
        os.remove("averages.csv")
    for fn in os.listdir("."):
        if fn.startswith("map_") and fn.endswith(".ppm"):
            os.remove(fn)


def create_temp_landscape(content):
    """Helper function to create a temporary landscape file."""
    tmpfile = tempfile.NamedTemporaryFile(delete=False, mode="w")
    tmpfile_name = tmpfile.name
    tmpfile.write(content)
    tmpfile.close()
    return tmpfile_name


def clean_up_files(patterns):
    """Helper to clean up files generated by the simulation."""
    for fn in os.listdir("."):
        for pattern in patterns:
            if fn.startswith(pattern["start"]) and fn.endswith(pattern["end"]):
                os.remove(fn)
    if os.path.exists("averages.csv"):
        os.remove("averages.csv")


def test_all_water_landscape():
    # Create an all-water 3x3 landscape
    landscape = "3 3\n0 0 0\n0 0 0\n0 0 0\n"
    fname = create_temp_landscape(landscape)

    args_list = [
        "simulate_predator_prey.py",
        "-f",
        fname,
        "--duration",
        "10",
        "--delta-t",
        "1.0",  # a few steps to check
    ]

    with patch("sys.argv", args_list):
        args = simulate_predator_prey.parse_command_line_args()
        simulate_predator_prey.run_predator_prey_simulation(
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

    with open("averages.csv", "r") as f:
        lines = f.readlines()
        # For an all-water map, number of land cells = 0, so average mice and foxes should remain 0
        for line in lines[1:]:  # skip header
            _, _, mice, foxes = line.strip().split(",")
            assert float(mice) == 0.0, "Mice average should remain 0 on all-water map"
            assert float(foxes) == 0.0, "Foxes average should remain 0 on all-water map"

    # Clean up
    os.remove(fname)
    clean_up_files([{"start": "map_", "end": ".ppm"}])


def test_single_land_cell():
    # Create a 1x1 landscape that is all land
    landscape = "1 1\n1\n"
    fname = create_temp_landscape(landscape)

    args_list = [
        "simulate_predator_prey.py",
        "-f",
        fname,
        "--duration",
        "2",
        "--delta-t",
        "0.5",
        "--mouse-seed",
        "1",
        "--fox-seed",
        "1",
    ]

    with patch("sys.argv", args_list):
        args = simulate_predator_prey.parse_command_line_args()
        simulate_predator_prey.run_predator_prey_simulation(
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

    with open("averages.csv", "r") as f:
        lines = f.readlines()
        # Just checking we got averages. With a single cell, average = that cell's density
        # We'll just confirm we have some non-zero mice and foxes at t=0
        header = lines[0].strip().split(",")
        assert header == ["Timestep", "Time", "Mice", "Foxes"]
        first_line = lines[1].strip().split(",")
        assert (
            float(first_line[2]) >= 0.0
        ), "Mice density should be a non-negative number"
        assert (
            float(first_line[3]) >= 0.0
        ), "Fox density should be a non-negative number"

    # Clean up
    os.remove(fname)
    clean_up_files([{"start": "map_", "end": ".ppm"}])


def test_ppm_file_creation():
    # Test that a ppm file is created at timestep 0 when time_step=1
    landscape = "2 2\n1 1\n1 1\n"
    fname = create_temp_landscape(landscape)

    args_list = [
        "simulate_predator_prey.py",
        "-f",
        fname,
        "--duration",
        "1",
        "--delta-t",
        "0.4",  # one step
        "--time_step",
        "1",
    ]

    with patch("sys.argv", args_list):
        args = simulate_predator_prey.parse_command_line_args()
        simulate_predator_prey.run_predator_prey_simulation(
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

    # Check that map_0000.ppm is created
    assert os.path.exists("map_0000.ppm"), "PPM file should be created at timestep 0"
    with open("map_0000.ppm", "r") as f:
        content = f.read()
        # Check basic PPM header format: starts with 'P3', has width/height, max color 255
        assert content.startswith("P3\n"), "PPM file should start with P3 header"
        assert "255" in content, "PPM file should contain max color value 255"

    # Clean up
    os.remove(fname)
    clean_up_files([{"start": "map_", "end": ".ppm"}])


def test_invalid_negative_parameters():
    # Test what happens if a negative duration is passed
    landscape = "2 2\n1 1\n1 1\n"
    fname = create_temp_landscape(landscape)

    # Negative duration does not make sense; we expect SystemExit (argparse) or no run
    args_list = ["simulate_predator_prey.py", "-f", fname, "--duration", "-10"]

    with patch("sys.argv", args_list), pytest.raises(SystemExit):
        # Argparse should fail because it expects a positive integer.
        simulate_predator_prey.parse_command_line_args()

    # Clean up
    os.remove(fname)


def test_zero_duration():
    # If duration=0, no timesteps should run.
    landscape = "2 2\n1 1\n1 1\n"
    fname = create_temp_landscape(landscape)

    args_list = [
        "simulate_predator_prey.py",
        "-f",
        fname,
        "--duration",
        "0",
        "--delta-t",
        "0.5",
    ]

    with patch("sys.argv", args_list):
        args = simulate_predator_prey.parse_command_line_args()
        simulate_predator_prey.run_predator_prey_simulation(
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

    # For duration=0, no simulation steps occur. Only the header should be present.
    with open("averages.csv", "r") as f:
        lines = f.readlines()
        # Expect only a header line in the CSV
        assert len(lines) == 1, "Should have only the header line for duration=0"
        header = lines[0].strip().split(",")
        assert header == ["Timestep", "Time", "Mice", "Foxes"], "CSV header mismatch"

    # Clean up
    os.remove(fname)
    clean_up_files([{"start": "map_", "end": ".ppm"}])

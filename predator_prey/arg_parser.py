from argparse import ArgumentParser, ArgumentTypeError

def nonnegative_int(value_str):
    value = int(value_str)
    if value < 0:
        raise ArgumentTypeError(f"{value} is not a valid non-negative integer.")
    return value

def nonnegative_float(value_str):
    value = float(value_str)
    if value < 0:
        raise ArgumentTypeError(f"{value} is not a valid non-negative float.")
    return value

def parse_command_line_args():
    par = ArgumentParser()
    par.add_argument(
        "-f",
        "--landscape-file",
        type=str,
        required=True,
        help="Input landscape file [REQUIRED]",
    )
    par.add_argument(
        "-r",
        "--birth-mice",
        type=nonnegative_float,
        default=0.1,
        help="Birth rate of mice",
    )
    par.add_argument(
        "-a",
        "--death-mice",
        type=nonnegative_float,
        default=0.05,
        help="Rate at which foxes eat mice",
    )
    par.add_argument(
        "-k",
        "--diffusion-mice",
        type=nonnegative_float,
        default=0.2,
        help="Diffusion rate of mice",
    )
    par.add_argument(
        "-b",
        "--birth-foxes",
        type=nonnegative_float,
        default=0.03,
        help="Birth rate of foxes",
    )
    par.add_argument(
        "-m",
        "--death-foxes",
        type=nonnegative_float,
        default=0.09,
        help="Fox mortality rate",
    )
    par.add_argument(
        "-l",
        "--diffusion-foxes",
        type=nonnegative_float,
        default=0.2,
        help="Diffusion rate of foxes",
    )
    par.add_argument(
        "-dt",
        "--delta-t",
        type=nonnegative_float,
        default=0.5,
        help="Time step size",
    )
    par.add_argument(
        "-t",
        "--time_step",
        type=nonnegative_int,
        default=10,
        help="Number of time steps at which to output files",
    )
    par.add_argument(
        "-d",
        "--duration",
        type=nonnegative_int,
        default=500,
        help="Time to run the simulation (in timesteps)",
    )
    par.add_argument(
        "-ms",
        "--mouse-seed",
        type=int,
        default=1,
        help="Random seed for initialising mouse densities",
    )
    par.add_argument(
        "-fs",
        "--fox-seed",
        type=int,
        default=1,
        help="Random seed for initialising fox densities",
    )
    return par.parse_args()

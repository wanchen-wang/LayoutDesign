"""
Driver script for batch data generation using V_Wave.run_simulation.

Modify `n_runs` below to control how many independent simulations
are executed (each creates its own timestamped subdirectory).

Usage:
    python V_Wave_Data_Generate.py      # prompts for number of runs or uses default

The original `V_Wave.py` no longer executes anything on import so it
can be reused as a module in other projects.
"""

from V_Wave import run_simulation


def generate_v_wave_data(n_runs=1, save=True):
    """Run `run_simulation` repeatedly.

    Parameters
    ----------
    n_runs : int
        Number of simulation realizations to perform.
    save : bool
        Passed through to `run_simulation`; keep True to write files.
    """
    for idx in range(1, n_runs + 1):
        print(f"\n=== Simulation {idx} of {n_runs} ===")
        run_simulation(save=save)


if __name__ == "__main__":
    # default value can be edited manually
    default_runs = 5

    try:
        val = input(f"Enter number of runs [{default_runs}]: ")
        if val.strip():
            n = int(val)
        else:
            n = default_runs
    except ValueError:
        print("Invalid input, using default.")
        n = default_runs

    generate_v_wave_data(n_runs=n, save=True)

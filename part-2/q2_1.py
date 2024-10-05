import numpy as np
import pandas as pd
from tabulate import tabulate

# Given constants
T_FWHM = 10 * 1e-12  # s
P0 = 1  # Peak power in W
C = 0  # Chirp parameter
gamma = 1.25  # W^-1 km^-1
alpha = 0.0461  # km^-1
alpha_db = 0.2  # dB km^-1
phi_NL_values = [
    0.5 * np.pi,
    1.5 * np.pi,
    2.5 * np.pi,
    3.5 * np.pi,
]  # Given nonlinear phase shifts


# Function to calculate effective length (L_eff)
def effective_length(z, alpha):
    return (1 - np.exp(-alpha * z)) / alpha


# a) Function to calculate transmission distance for given phi_NL
def transmission_distance(phi_NL, gamma, P0, alpha):
    # Using the relation phi_NL = gamma * P0 * Leff
    # Leff = phi_NL / (gamma * P0)
    # Solve for z: Leff = (1 - e^(-alpha * z)) / alpha
    L_eff_required = phi_NL / (gamma * P0)
    z = -np.log(1 - alpha * L_eff_required) / alpha
    return z, L_eff_required


def transmission_distances():
    results = []

    for phi in phi_NL_values:
        z, L_eff = transmission_distance(phi, gamma, P0, alpha)
        results.append(
            {
                "φNL_max (radians)": phi,
                "Transmission Distance z (km)": z,
                "Effective Length Leff (km)": L_eff,
            }
        )

    return pd.DataFrame(results)


def main():
    # Calculate transmission distances and effective lengths
    print("a, b, c)")
    print(tabulate(transmission_distances(), headers="keys", tablefmt="psql"))


if __name__ == "__main__":
    main()

import numpy as np
import pandas as pd
from tabulate import tabulate

# Given constants
P0 = 1.0  # Peak power in W
gamma = 1.25  # Nonlinear coefficient in W^-1 km^-1
alpha = 0.0461  # Attenuation coefficient in km^-1
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
    Leff_required = phi_NL / (gamma * P0)
    z = -np.log(1 - alpha * Leff_required) / alpha
    return z, Leff_required


def get_transmission_distances():
    results = []

    for phi in phi_NL_values:
        z, Leff = transmission_distance(phi, gamma, P0, alpha)
        results.append(
            {
                "φNL_max (radians)": phi,
                "Transmission Distance z (km)": z,
                "Effective Length Leff (km)": Leff,
            }
        )

    return pd.DataFrame(results)


def main():
    # Calculate transmission distances and effective lengths
    df_results = get_transmission_distances()

    print(tabulate(df_results, headers="keys", tablefmt="psql"))


if __name__ == "__main__":
    main()

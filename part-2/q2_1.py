import numpy as np
import pandas as pd
from tabulate import tabulate

# Given constants
gamma = 1.25  # Nonlinear coefficient in W^-1 km^-1
P0 = 1.0  # Peak power in W
alpha = 0.0461  # Attenuation coefficient in km^-1
phi_NL_values = [
    0.5 * np.pi,
    1.5 * np.pi,
    2.5 * np.pi,
    3.5 * np.pi,
]  # Given nonlinear phase shifts


# Function to calculate effective length (Leff)
def effective_length(z, alpha):
    return (1 - np.exp(-alpha * z)) / alpha


# Function to calculate transmission distance for given phi_NL
def transmission_distance(phi_NL, gamma, P0, alpha):
    # Using the relation phi_NL = gamma * P0 * Leff
    # Leff = phi_NL / (gamma * P0)
    # Solve for z: Leff = (1 - e^(-alpha * z)) / alpha
    Leff_required = phi_NL / (gamma * P0)
    z = -np.log(1 - alpha * Leff_required) / alpha
    return z, Leff_required


def main():
    # Calculate transmission distances and effective lengths
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

    # Convert results to a dataframe for easy visualization
    df_results = pd.DataFrame(results)

    print(tabulate(df_results, headers="keys", tablefmt="psql"))


if __name__ == "__main__":
    main()

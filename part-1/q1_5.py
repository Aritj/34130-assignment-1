import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tabulate import tabulate
from q1_1 import N, C_VALUES, A0, sampling_and_frequency_params
from q1_2 import T0, t, electrical_field_envelope, power_of_pulse
from q1_3 import measure_FWHM

# Set figure DPI to 300 (increasing plot resolution)
plt.rcParams["savefig.dpi"] = 300

# Get T_sa from Q1-1
T_sa, F_sa, Delta_F, F_min = sampling_and_frequency_params()

# Fiber parameters
beta_2 = -21.68  # Group Velocity Dispersion (ps^2/km)
Z_VALUES = [0, 0.3199, 1.6636, 3.3272]  # Propagation distances in km
f = np.fft.fftfreq(N, T_sa * 1e12)  # Frequency vector in Hz
omega_vector = 2 * np.pi * f  # Angular frequency vector


def spectrum(A_f: np.ndarray, f: np.ndarray, z: float) -> np.ndarray:
    omega_vector = 2 * np.pi * f
    first_term = 1j * (beta_2 / 2) * z * omega_vector**2
    second_term = 0  # assuming β_3 = 0
    return A_f * np.exp(first_term + second_term)


def propagate_pulse(A_0, C_values: list[int], z_values: list[float]) -> dict:
    # Compute the evolution for each chirp value and distance
    results = {}  # Store results for analysis

    # Calculates the pulse propogation for each C and z value
    for C in C_values:
        results[C] = {}
        A_t = electrical_field_envelope(A_0, T0, C, t)
        A_f = np.fft.fft(A_t)  # convert to frequency domain

        # Calculates for each distance
        for z in z_values:
            A_zf = spectrum(A_f, f, z)
            A_zt = np.fft.ifft(A_zf)  # Convert back to time domain
            P_zt = power_of_pulse(A_zt)  # b) Calculate power
            results[C][z] = (t, A_zt, P_zt)  # Store results

    return results


def main() -> None:
    # Calculating propogated pulses for each C and z value
    propagated_pulses = propagate_pulse(A0, C_VALUES, Z_VALUES)

    # Measuring FWHM for each C and z value
    table_data = []
    for C in C_VALUES:
        row = {"C": C}
        for i, z in enumerate(Z_VALUES):
            time_vector, A_zt, P_zt = propagated_pulses[C][z]
            header = f"T_FWHM{f'(z_{i})'*bool(z)} (ps)\nz{f'_{i}'*bool(z)} = {z} km"
            row[header] = measure_FWHM(time_vector, P_zt)
        table_data.append(row)

    # Printing the results in a tabular form
    print(
        tabulate(
            pd.DataFrame(table_data), headers="keys", tablefmt="psql", showindex=False
        )
    )

    # Plotting the results, one row of subplots for each z value
    fig, axs = plt.subplots(1, len(Z_VALUES), figsize=(14, 3))

    for j, z in enumerate(Z_VALUES):
        axs[j].set_xlim(-100, 100)

        # Plotting all C values for this z value
        for C in C_VALUES:
            time_vector, A_zt, P_zt = propagated_pulses[C][z]
            axs[j].plot(time_vector, P_zt, label=f"C={C}")

        axs[j].set_xlabel("Time (ps)")
        axs[j].set_ylabel("Normalized Power Spectrum")
        axs[j].legend()
        axs[j].grid()
        axs[j].set_title(f"z={z} km")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

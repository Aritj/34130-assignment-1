import numpy as np
import matplotlib.pyplot as plt

from q1_1 import N, TW, sampling_and_frequency_params
from q1_2 import A0, C_VALUES
from q1_3 import measure_FWHM
from q1_5 import Z_VALUES, propagate_pulse

# Define constants and parameters
beta_2 = -21.68  # Group Velocity Dispersion (ps^2/km)
T0 = 10 / (2 * np.sqrt(np.log(2)))  # Convert FWHM to T0 for Gaussian pulse
z_range = np.arange(0, 5.001, 0.001)  # 0-5 km range

# Define the parameters for pulse propagation
T_sa, _, _, _ = sampling_and_frequency_params(N, TW)


def analytical_ratio(beta_2: float, C: int, z: float) -> float:
    return np.sqrt((1 + beta_2 * C * z / T0**2) ** 2 + (beta_2 * z / T0**2) ** 2)


def main():
    # Calculate the analytical ratio for each C and z
    analytical_ratios = {C: analytical_ratio(beta_2, C, z_range) for C in C_VALUES}

    # Propogate the pulse for each C and z and measure the FWHM
    propagated_pulses = propagate_pulse(A0, C_VALUES, Z_VALUES)
    measured_FWHM_values = {C: [] for C in C_VALUES}

    for C in C_VALUES:
        for z in Z_VALUES:
            t, A_zt, P_zt = propagated_pulses[C][z]
            measured_FWHM = measure_FWHM(t, P_zt)
            measured_FWHM_values[C].append(measured_FWHM)

    # Calculate numerical ratios: TFWHM(z) / TFWHM(0)
    numerical_ratios = {
        C: np.array(measured_FWHM_values[C]) / measured_FWHM_values[C][0]
        for C in C_VALUES
    }

    plt.figure(figsize=(10, 6))

    # Plot analytical ratios (line plot)
    for C in C_VALUES:
        plt.plot(z_range, analytical_ratios[C], label=f"Analytical C={C}")

    # Plot numerical ratios (scatter plot)
    plt.scatter(Z_VALUES, numerical_ratios[-10], label="Numerical C=-10", marker="o")
    plt.scatter(Z_VALUES, numerical_ratios[0], label="Numerical C=0", marker="o")
    plt.scatter(Z_VALUES, numerical_ratios[5], label="Numerical C=5", marker="o")

    # Plot settings
    plt.xlabel("Distance z (km)")
    plt.ylabel("Ratio $T_{FWHM1}(z)/T_{FWHM}$")
    # plt.title("Ratio of Temporal Width vs Distance for Different Chirp Values")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

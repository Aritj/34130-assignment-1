import numpy as np
import matplotlib.pyplot as plt

from q1_1 import N, TW, calculate_sampling_and_frequency_params
from q1_2 import A0, t, C_VALUES, calculate_electrical_field_envelope, calculate_power_of_pulse
from q1_3 import measure_FWHM
from q1_5 import Z_VALUES, calculate_spectrum

# Define constants and parameters
beta_2 = -21.68  # Group Velocity Dispersion (ps^2/km)
T0 = 10 / (2 * np.sqrt(np.log(2)))  # Convert FWHM to T0 for Gaussian pulse
z_analytical_range = np.arange(0, 5.001, 0.001)  # 0-5 km range

# Define the parameters for pulse propagation
T_sa, _, _, _ = calculate_sampling_and_frequency_params(N, TW)


def analytical_ratio(beta_2: float, C: int, z: float) -> float:
    return np.sqrt((1 + beta_2 * C * z / T0**2) ** 2 + (beta_2 * z / T0**2) ** 2)


# Function to propagate in the frequency domain
def propagate_pulse(A_t, time_vector, z):
    # Perform FFT
    f = np.fft.fftfreq(len(time_vector), T_sa * 1e12)  # Frequency vector
    A_f = np.fft.fft(A_t)  # Frequency domain representation of input pulse
    A_zf = calculate_spectrum(A_f, f, z)
    # Perform inverse FFT
    return np.fft.ifft(A_zf)


def main():
    # Calculate the analytical ratio for each C and z
    analytical_ratios = {C: analytical_ratio(beta_2, C, z_analytical_range) for C in C_VALUES}

    # Initial pulses for C values
    initial_pulses = {
        C: calculate_electrical_field_envelope(A0, T0, C, t) for C in C_VALUES
    }

    # Measure and store FWHM for each z and C value
    measured_FWHM_values = {C: [] for C in C_VALUES}

    for C, pulse in initial_pulses.items():
        for z in Z_VALUES:
            if z == 0:
                power = calculate_power_of_pulse(pulse)
            else:
                propagated_pulse = propagate_pulse(pulse, t, z)
                power = calculate_power_of_pulse(propagated_pulse)

            measured_FWHM = measure_FWHM(t, power)
            measured_FWHM_values[C].append(measured_FWHM)

    # Calculate numerical ratios: TFWHM(z) / TFWHM(0)
    numerical_ratios = {
        C: np.array(measured_FWHM_values[C]) / measured_FWHM_values[C][0] for C in C_VALUES
    }

    plt.figure(figsize=(10, 6))
    
    # Plot analytical ratios (line plot)
    for C in C_VALUES:
        plt.plot(z_analytical_range, analytical_ratios[C], label=f"Analytical C={C}")

    # Plot numerical ratios (scatter plot)
    plt.scatter(Z_VALUES, numerical_ratios[-10], label="Simulated C=-10", marker="o")
    plt.scatter(Z_VALUES, numerical_ratios[0], label="Simulated C=0", marker="o")
    plt.scatter(Z_VALUES, numerical_ratios[5], label="Simulated C=5", marker="o")

    # Plot settings
    plt.xlabel("Distance z (km)")
    plt.ylabel("Ratio $T_{FWHM1}(z)/T_{FWHM}$")
    plt.title("Ratio of Temporal Width vs Distance for Different Chirp Values")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()

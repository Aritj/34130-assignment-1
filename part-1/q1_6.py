import numpy as np
import matplotlib.pyplot as plt

from q1_1 import calculate_sampling_and_frequency_params
from q1_2 import A0, TIME_VECTOR, C_VALUES, calculate_gaussian_field
from q1_3 import calculate_FWHM
from q1_5 import Z_VALUES

# Define constants and parameters
beta_2 = -21.68  # Group Velocity Dispersion (ps^2/km)
T0 = 10 / (2 * np.sqrt(np.log(2)))  # Convert FWHM to T0 for Gaussian pulse
z_analytical_values = np.arange(0, 5.001, 0.001)  # 0-5 km range

# Define the parameters for pulse propagation
T_sa, _, _, _ = calculate_sampling_and_frequency_params()


# Define pulse envelope for different chirp values
def gaussian_pulse(t, A0, T0, C):
    return A0 * np.exp(-(1 + 1j * C) * (t**2) / (T0**2))


# Define the analytical ratio function as specified
def analytical_ratio(C, z):
    return np.sqrt(1 + (beta_2 * C * z / T0**2) ** 2 + (beta_2 * z / T0**2) ** 2)


# Function to propagate in the frequency domain
def propagate_pulse(A0, time_vector, z, beta2):
    # Perform FFT
    freq_vector = np.fft.fftfreq(len(time_vector), T_sa * 1e12)  # Frequency vector
    A_freq = np.fft.fft(A0)  # Frequency domain representation of input pulse

    # Apply phase shift due to dispersion
    H = np.exp(1j * (beta2 * (2 * np.pi * freq_vector) ** 2 / 2) * z)
    A_freq_out = A_freq * H

    # Perform inverse FFT
    A_out = np.fft.ifft(A_freq_out)
    return A_out


def main():
    # Calculate the analytical ratio for each C and z
    analytical_ratios = {C: analytical_ratio(C, z_analytical_values) for C in C_VALUES}

    # Initial pulses for C values
    initial_pulses = {
        C: calculate_gaussian_field(A0, T0, C, TIME_VECTOR) for C in C_VALUES
    }

    # Store FWHM values for each chirp and distance
    FWHM_values = {C: [] for C in C_VALUES}

    # Calculate FWHM for each distance and chirp value
    for C, pulse in initial_pulses.items():
        for z in Z_VALUES:
            if z == 0:
                power = np.abs(pulse) ** 2
            else:
                propagated_pulse = propagate_pulse(pulse, TIME_VECTOR, z, beta_2)
                power = np.abs(propagated_pulse) ** 2

            FWHM = calculate_FWHM(TIME_VECTOR, power)
            FWHM_values[C].append(FWHM)

    # Calculate numerical ratios: TFWHM(z) / TFWHM(0)
    numerical_ratios = {
        C: np.array(FWHM_values[C]) / FWHM_values[C][0] for C in C_VALUES
    }

    # Regenerate the scatter plot with the exact numerical values
    plt.figure(figsize=(10, 6))
    # Plot analytical ratios
    for C in C_VALUES:
        plt.plot(z_analytical_values, analytical_ratios[C], label=f"Analytical C={C}")

    # Scatter points for numerical ratios using actual calculated FWHM values
    plt.scatter(Z_VALUES, numerical_ratios[-10], label="Simulated C=-10", marker="o")
    plt.scatter(Z_VALUES, numerical_ratios[0], label="Simulated C=0", marker="o")
    plt.scatter(Z_VALUES, numerical_ratios[5], label="Simulated C=5", marker="o")

    # Labels and legend
    plt.xlabel("Distance z (km)")
    plt.ylabel("Ratio $T_{FWHM1}(z)/T_{FWHM}$")
    plt.title("Ratio of Temporal Width vs Distance for Different Chirp Values")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()

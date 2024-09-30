import numpy as np
import matplotlib.pyplot as plt
from q2_1 import phi_NL_values, effective_length, get_transmission_distances
from q2_2 import (
    calculate_sampling_and_frequency_params,
    generate_time_and_frequency_vectors,
)

# Constants
TW = 2500e-12  # Total time window in seconds
N = 2**14  # Number of samples
TFWHM = 10e-12  # Full Width Half Maximum (FWHM) in seconds
P0 = 1.0  # Peak power in Watts
C = 0  # Chirp parameter

# Get transmission lengths from Q2-1
z_values = get_transmission_distances()["Transmission Distance z (km)"].tolist()

# Get parameters from Q2-2
T_sa, F_sa, Delta_F, F_min = calculate_sampling_and_frequency_params()

# Get Time and Frequency Vector from Q2-2
time_vector, frequency_vector = generate_time_and_frequency_vectors(
    T_sa, Delta_F, F_min
)

# Pulse properties
T0 = TFWHM / (2 * np.sqrt(2 * np.log(2)))  # T0 is the width parameter
A0 = np.sqrt(P0)  # Peak amplitude

# Define initial time-domain Gaussian pulse
A_time_initial = A0 * np.exp(
    -(1 + 1j * C) * (time_vector**2) / (2 * T0**2)
)  # Complex field envelope

# Convert initial time-domain pulse to frequency domain
A_freq_initial = np.fft.fftshift(
    np.fft.fft(A_time_initial)
)  # Frequency-domain representation

# Fiber parameters from previous sections
alpha = 0.0461  # Attenuation coefficient in km^-1
gamma = 1.25  # Non-linear coefficient in W^-1 km^-1


def main():
    # (b) Calculate the spectrum at the output of the fiber for each distance
    output_spectra = {}
    for z in z_values:
        Leff = effective_length(z, alpha)  # Calculate the effective length
        # Modify the spectrum using the phase shift caused by nonlinearity and dispersion
        A_freq_output = A_freq_initial * np.exp(1j * gamma * P0 * Leff * z)
        output_spectra[z] = np.abs(A_freq_output) ** 2  # Store the power spectrum

    # Normalize the input and output spectra to the peak of the input spectrum
    P_freq_initial = np.abs(A_freq_initial) ** 2
    P_freq_initial_normalized = P_freq_initial / np.max(P_freq_initial)

    # Normalize each output spectrum to the input peak power
    for z in output_spectra:
        output_spectra[z] = output_spectra[z] / np.max(P_freq_initial)

    # (c) Plot the input and output spectra with distinctive styles
    plt.figure(figsize=(14, 7))
    # Plot Input Spectrum with a thick dashed line
    plt.plot(
        frequency_vector * 1e-9,
        P_freq_initial_normalized,
        label="Input Spectrum",
        linestyle="--",
        color="black",
        linewidth=2,
    )

    # Plot each output spectrum with a distinct style and marker
    styles = ["-", "--", "-.", ":"]
    # markers = ["o", "s", "^", "d"]
    colors = ["red", "blue", "green", "orange"]

    for idx, z in enumerate(z_values):
        plt.plot(
            frequency_vector * 1e-9,
            output_spectra[z],
            label=f"Output Spectrum (z = {z:.2f} km)",
            linestyle=styles[idx],
            color=colors[idx],
            # marker=markers[idx],
            # markevery=1000,  # Place markers at every 1000 data points to highlight
            linewidth=1.5,
            alpha=0.8,
        )

    # Final plot settings
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Normalized Power")
    plt.title("Input and Output Spectra at Various Transmission Distances")
    plt.legend()
    plt.grid()
    plt.xlim(-100, 100)
    plt.show()


if __name__ == "__main__":
    main()

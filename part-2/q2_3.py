import numpy as np
import matplotlib.pyplot as plt
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

# Calculate parameters from previous steps
T_sa, F_sa, Delta_F, F_min = calculate_sampling_and_frequency_params()

# (e) Time Vector
time_vector, frequency_vector = generate_time_and_frequency_vectors(
    T_sa, Delta_F, F_min
)


def main():
    # (a) Calculate the power of the pulse in time (normalized to temporal peak power)
    # Calculate T0 from TFWHM: TFWHM = 2 * sqrt(2 * ln(2)) * T0
    T0 = TFWHM / (2 * np.sqrt(2 * np.log(2)))  # T0 is the width parameter
    A0 = np.sqrt(P0)  # Peak amplitude
    # Time-domain Gaussian pulse
    A_time = A0 * np.exp(
        -(1 + 1j * C) * (time_vector**2) / (2 * T0**2)
    )  # Complex field envelope
    P_time = np.abs(A_time) ** 2  # Power is the squared modulus of the field

    # Normalize power in time to peak power
    P_time_normalized = P_time / np.max(P_time)

    # (b) Calculate the power of the pulse in frequency (normalized to spectral peak power)
    # Perform FFT and shift zero frequency to center
    A_freq = np.fft.fftshift(np.fft.fft(A_time))  # Frequency-domain representation
    P_freq = np.abs(A_freq) ** 2  # Power is the squared modulus of the frequency field

    # Normalize power in frequency to peak power
    P_freq_normalized = P_freq / np.max(P_freq)

    # Plot the power of the pulse in time and frequency side-by-side
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(16, 5)
    )  # Create side-by-side subplots

    # (a) Plot the power of the pulse in time
    ax1.plot(time_vector * 1e12, P_time_normalized, label=f"Power of Pulse (C = {C})")
    ax1.set_xlabel("Time (ps)")
    ax1.set_ylabel("Normalized Power")
    ax1.set_title("Normalized Power of Gaussian Pulse in Time Domain")
    ax1.grid()
    ax1.legend()
    # Set scale for time-domain plot
    ax1.set_xlim(-12, 12)

    # (b) Plot the power of the pulse in frequency
    ax2.plot(
        frequency_vector * 1e-9, P_freq_normalized, label=f"Power of Pulse (C = {C})"
    )
    ax2.set_xlabel("Frequency (GHz)")
    ax2.set_ylabel("Normalized Power")
    ax2.set_title("Normalized Power of Gaussian Pulse in Frequency Domain")
    ax2.grid()
    ax2.legend()
    # Set scale for frequency-domain plot
    ax2.set_xlim(-100, 100)

    # Show both plots side-by-side
    plt.tight_layout()
    plt.show()

    # (c) State Full Width Half Maximum for the pulse in both time and frequency
    # Calculate the FWHM in time
    half_max_time = np.max(P_time_normalized) / 2.0
    indices_time = np.where(P_time_normalized >= half_max_time)[0]
    FWHM_time = (
        time_vector[indices_time[-1]] - time_vector[indices_time[0]]
    ) * 1e12  # in ps

    # Calculate the FWHM in frequency
    half_max_freq = np.max(P_freq_normalized) / 2.0
    indices_freq = np.where(P_freq_normalized >= half_max_freq)[0]
    FWHM_freq = (
        frequency_vector[indices_freq[-1]] - frequency_vector[indices_freq[0]]
    ) * 1e-9  # in GHz

    print(f"(c) FWHM in Time Domain: {FWHM_time:.2f} ps")
    print(f"(c) FWHM in Frequency Domain: {FWHM_freq:.2f} GHz")


if __name__ == "__main__":
    main()

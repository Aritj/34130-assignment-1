import numpy as np
import matplotlib.pyplot as plt
from q2_1 import T_FWHM, C, P0
from q2_2 import (
    sampling_and_frequency_params,
    generate_time_and_frequency_vectors,
)

# Set figure DPI to 300 (increasing plot resolution)
plt.rcParams["savefig.dpi"] = 300

# Constants
T0 = T_FWHM / (2 * np.sqrt(np.log(2)))
A0 = np.sqrt(P0)  # Peak amplitude

# Calculate parameters from previous steps
T_sa, F_sa, Delta_F, F_min = sampling_and_frequency_params()

# Time and Frequency vectors
t, f = generate_time_and_frequency_vectors(T_sa, Delta_F, F_min)


def electrical_field_envelope(
    A0: int, T0: float, C: list[int], t: np.ndarray
) -> np.ndarray:
    """Calculates pre-chirped Gaussian field envelope"""
    return A0 * np.exp(-((1 + 1j * C) / 2) * (t / T0) ** 2)


def power_of_pulse(A_t: np.ndarray) -> np.ndarray:
    """Calculates the power spectrum"""
    return A_t * np.conjugate(A_t)


def measure_FWHM(t, P):
    half_max = np.max(P) / 2
    indices = np.where(P >= half_max)[0]
    return t[indices[-1]] - t[indices[0]]  # Return the difference (FWHM)


def normalize(vector):
    return np.abs(vector / np.max(vector))


def main():
    # a, b)Calculate the power of the pulse in time (normalized to temporal peak power)
    A_t = electrical_field_envelope(A0, T0, C, t)
    P_t = power_of_pulse(A_t)

    # Normalize power in time to peak power
    P_t_normalized = normalize(P_t)

    # Calculate the power of the pulse in frequency (normalized to spectral peak power)
    A_f = np.fft.fftshift(np.fft.fft(A_t))  # Frequency-domain representation
    P_f = power_of_pulse(A_f)

    # Normalize power in frequency to peak power
    P_f_normalized = normalize(P_f)

    # Plot the power of the pulse in time and frequency side-by-side
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(16, 5)
    )  # Create side-by-side subplots

    # a) Plot the power of the pulse in time
    ax1.plot(t * 1e12, P_t_normalized, label=f"Power of Pulse (C = {C})")
    ax1.set_xlabel("Time (ps)")
    ax1.set_ylabel("Normalized Power")
    ax1.set_title("Normalized Power of Gaussian Pulse in Time Domain")
    ax1.grid(True)
    ax1.set_xlim(-20, 20)

    # b) Plot the power of the pulse in frequency
    ax2.plot(f * 1e-9, P_f_normalized, label=f"Power of Pulse (C = {C})")
    ax2.set_xlabel("Frequency (GHz)")
    ax2.set_ylabel("Normalized Power")
    ax2.set_title("Normalized Power of Gaussian Pulse in Frequency Domain")
    ax2.grid(True)
    ax2.set_xlim(-100, 100)

    # Show both plots side-by-side
    plt.tight_layout()
    plt.show()

    # c) State Full Width Half Maximum for the pulse in both time and frequency
    # Calculate the FWHM in time and frequency
    FWHM_time = measure_FWHM(t, P_t_normalized) * 1e12  # in ps
    FWHM_freq = measure_FWHM(f, P_f_normalized) * 1e-9  # in GHz

    print(f"(c) FWHM in Time Domain: {FWHM_time:.2f} ps")
    print(f"(c) FWHM in Frequency Domain: {FWHM_freq:.2f} GHz")


if __name__ == "__main__":
    main()

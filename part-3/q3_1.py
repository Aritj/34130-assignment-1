import numpy as np
import matplotlib.pyplot as plt

# Define constants and parameters
TFWHM = 10e-12  # Full Width Half Maximum (FWHM) in seconds
C_values = [-10, 0, +5]  # Chirp parameters to investigate
A0 = 1.0  # Pulse peak amplitude (W^0.5)

# Fiber characteristics
beta2 = -21.68e-24  # Dispersion parameter in s^2/m
gamma = 0.0  # Non-linearity coefficient in W^-1m^-1 (set to 0 for this exercise)
alpha = 0.0  # Attenuation in m^-1

# Simulation settings
N = 2**14  # Number of samples
TW = 2500e-12  # Total time window in seconds
dz = 3.3272e3 / 5000  # Segment length in meters
z_values = [0.3199e3, 1.6636e3, 3.3272e3]  # Propagation distances in meters

# Generate time and frequency vectors
Tsa = TW / N  # Sampling period in time
time_vector = np.linspace(-TW / 2, TW / 2 - Tsa, N)
freq_vector = np.fft.fftshift(np.fft.fftfreq(N, Tsa))

# Calculate T0 from TFWHM
T0 = TFWHM / (2 * np.sqrt(2 * np.log(2)))  # T0 is the pulse duration parameter


# Function to define Gaussian pulse with chirp
def gaussian_pulse(A0, T0, C, time_vector):
    return A0 * np.exp(-(1 + 1j * C) * time_vector**2 / (2 * T0**2))


# Function to apply dispersion in the frequency domain
def apply_dispersion(A_freq, beta2, dz, freq_vector):
    dispersion_operator = np.exp(-1j * beta2 * (2 * np.pi * freq_vector) ** 2 * dz / 2)
    return A_freq * dispersion_operator


# Split-Step Fourier Method
def split_step_method(
    A_initial, beta2, dz, gamma, alpha, Nseg, z_total, freq_vector, time_vector
):
    dz_segment = z_total / Nseg
    A = A_initial.copy()

    # Propagate through each segment
    for _ in range(Nseg):
        # Step 1: Apply dispersion in frequency domain
        A_freq = np.fft.fft(A)
        A_freq = apply_dispersion(A_freq, beta2, dz_segment, freq_vector)
        A = np.fft.ifft(A_freq)

    return A


def main():
    # Plot results
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle("Pulse Propagation using Split-Step Fourier Method")

    line_styles = ["--", "-.", ":"]

    for idx, C in enumerate(C_values):
        # Define initial pulse
        A_initial = gaussian_pulse(A0, T0, C, time_vector)

        # Propagate pulse for each specified distance
        for jdx, z in enumerate(z_values):
            Nseg = int(z / dz)  # Number of segments for the given distance
            A_out = split_step_method(
                A_initial, beta2, dz, gamma, alpha, Nseg, z, freq_vector, time_vector
            )

            # Time Domain Plot
            axes[idx, 0].plot(
                time_vector * 1e12, np.abs(A_out) ** 2, label=f"z = {z/1e3:.3f} km"
            )
            axes[idx, 0].set_xlabel("Time (ps)")
            axes[idx, 0].set_ylabel("Power (W)")
            axes[idx, 0].set_title(f"Time Domain for C = {C}")
            axes[idx, 0].grid()
            axes[idx, 0].set_xlim(-100, 100)
            axes[idx, 0].legend()

            # Frequency Domain Plot
            A_freq_out = np.fft.fftshift(np.fft.fft(A_out))
            axes[idx, 1].plot(
                freq_vector * 1e-9,
                np.abs(A_freq_out) ** 2,
                label=f"z = {z/1e3:.3f} km",
                linestyle=line_styles[jdx],
            )
            axes[idx, 1].set_xlabel("Frequency (GHz)")
            axes[idx, 1].set_ylabel("Power Spectral Density")
            axes[idx, 1].set_title(f"Frequency Domain for C = {C}")
            axes[idx, 1].grid()
            axes[idx, 1].set_xlim(-1000, 1000)
            axes[idx, 1].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


if __name__ == "__main__":
    main()

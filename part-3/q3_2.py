import numpy as np
import matplotlib.pyplot as plt

# Define constants and parameters
TFWHM = 10e-12  # Full Width Half Maximum (FWHM) in seconds
A0 = 1.0  # Pulse peak amplitude (W^0.5)
gamma = 1.25  # Non-linear coefficient in W^-1 km^-1
alpha = 0.0461  # Attenuation coefficient in km^-1
beta2 = 0  # Dispersion parameter (set to 0 for this exercise)

# Simulation settings
N = 2**14  # Number of samples
TW = 2500e-12  # Total time window in seconds
dz = 0.001  # Segment length in km
z_values = [1.2945, 4.1408, 7.4172, 11.2775]  # Propagation distances in km

# Calculate parameters
Tsa = TW / N  # Sampling period in time
time_vector = np.linspace(-TW / 2, TW / 2 - Tsa, N)  # Time vector in seconds
freq_vector = np.fft.fftshift(np.fft.fftfreq(N, Tsa))  # Frequency vector

# Calculate T0 from TFWHM
T0 = TFWHM / (2 * np.sqrt(2 * np.log(2)))  # T0 is the pulse duration parameter


# Define Gaussian pulse
def gaussian_pulse(A0, T0, time_vector):
    return A0 * np.exp(-(time_vector**2) / (2 * T0**2))


# Split-Step Method without Dispersion
def split_step_nonlinear(A_initial, dz, gamma, alpha, Nseg):
    A = A_initial.copy()
    for _ in range(Nseg):
        # Step 1: Apply non-linearity in the time domain
        A = A * np.exp(1j * gamma * np.abs(A) ** 2 * dz)

        # Step 2: Apply attenuation
        A = A * np.exp(-alpha * dz / 2)

    return A


def main():
    # Initialize the initial pulse
    A_initial = gaussian_pulse(A0, T0, time_vector)

    # Calculate the initial peak power for normalization
    initial_peak_power = np.max(np.abs(A_initial) ** 2)
    # Initial spectrum for frequency normalization
    A_freq_initial = np.fft.fftshift(np.fft.fft(A_initial))
    initial_peak_spectrum = np.max(np.abs(A_freq_initial) ** 2)

    # Set up the figure with 2 subplots: one for time and one for frequency
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Pulse Propagation with Non-Linearity and Attenuation")

    # Color and line styles for distinct visualization
    # colors = ["blue", "orange", "green", "red"]
    line_styles = ["-", "--", "-.", ":"]

    # Propagate and plot for each specified distance
    for idx, z in enumerate(z_values):
        Nseg = int(z / dz)  # Number of segments for the given distance
        A_out = split_step_nonlinear(A_initial, dz, gamma, alpha, Nseg)

        # Normalize in time domain by the initial peak power
        power_time = np.abs(A_out) ** 2 / initial_peak_power

        # Time Domain Plot
        ax1.plot(
            time_vector * 1e12,
            power_time,
            label=f"z = {z:.4f} km",
            # color=colors[idx],
            # linestyle=line_styles[idx],
        )
        ax1.set_xlabel("Time (ps)")
        ax1.set_ylabel("Normalized Power")
        ax1.set_title("Time Domain Pulse Evolution")
        # ax1.grid()
        ax1.set_xlim(-10, 10)
        ax1.set_ylim(0, 1)
        ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

        # Normalize in frequency domain by the initial peak spectral density
        A_freq_out = np.fft.fftshift(np.fft.fft(A_out))
        power_spectrum = np.abs(A_freq_out) ** 2 / initial_peak_spectrum

        # Frequency Domain Plot
        ax2.plot(
            freq_vector * 1e-9,
            power_spectrum,
            label=f"z = {z:.4f} km",
            # color=colors[idx],
            # linestyle=line_styles[idx],
        )
        ax2.set_xlabel("Frequency (GHz)")
        ax2.set_ylabel("Normalized Power Spectral Density")
        ax2.set_title("Frequency Domain Pulse Evolution")
        ax2.set_xlim(-500, 500)
        ax2.set_ylim(0, 1)
        ax2.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    # Add legends and adjust layout
    ax1.legend()
    ax2.legend()
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


if __name__ == "__main__":
    main()

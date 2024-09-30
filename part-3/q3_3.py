import numpy as np
import matplotlib.pyplot as plt

# Define constants and parameters
TFWHM = 10e-12  # Full Width Half Maximum (FWHM) in seconds
A0 = 1.0  # Pulse peak amplitude (W^0.5)
gamma = 1.25  # Non-linear coefficient in W^-1 km^-1
alpha = 0.0461  # Attenuation coefficient in km^-1
beta2 = -21.68e-24  # Dispersion parameter in s^2/m

# Simulation settings
N = 2**14  # Number of samples
TW = 2500e-12  # Total time window in seconds
dz = 0.001  # Segment length in km
z_values = [1.5532, 5.2621, 8.9709, 12.6798]  # Propagation distances in km

# Calculate parameters
Tsa = TW / N  # Sampling period in time
time_vector = np.linspace(-TW / 2, TW / 2 - Tsa, N)  # Time vector in seconds
freq_vector = np.fft.fftshift(np.fft.fftfreq(N, Tsa))  # Frequency vector in Hz

# Calculate T0 from TFWHM
T0 = TFWHM / (2 * np.sqrt(2 * np.log(2)))  # T0 is the pulse duration parameter


# Define Gaussian pulse
def gaussian_pulse(A0, T0, time_vector):
    return A0 * np.exp(-(time_vector**2) / (2 * T0**2))


# Dispersion operator in the frequency domain
def dispersion_operator(beta2, dz, freq_vector):
    omega = 2 * np.pi * freq_vector  # Angular frequency vector
    return np.exp(-1j * beta2 * omega**2 * dz / 2)


# Split-Step Method with Dispersion and Non-Linearity
def split_step_fourier(A_initial, dz, beta2, gamma, alpha, Nseg, freq_vector):
    A = A_initial.copy()
    dispersion_op = dispersion_operator(
        beta2, dz * 1e3, freq_vector
    )  # Convert dz to meters for calculation

    for _ in range(Nseg):
        # Step 1: Apply non-linearity in the time domain
        A = A * np.exp(1j * gamma * np.abs(A) ** 2 * dz)

        # Step 2: Apply dispersion in the frequency domain
        A_freq = np.fft.fft(A)
        A_freq = A_freq * dispersion_op
        A = np.fft.ifft(A_freq)

        # Step 3: Apply attenuation
        A = A * np.exp(-alpha * dz / 2)

    return A


# Initialize the initial pulse
A_initial = gaussian_pulse(A0, T0, time_vector)

# Calculate initial peak power and peak spectral density for normalization
initial_peak_power = np.max(np.abs(A_initial) ** 2)
A_freq_initial = np.fft.fftshift(np.fft.fft(A_initial))
initial_peak_spectral_density = np.max(np.abs(A_freq_initial) ** 2)

# Set up the figure with 2 subplots: one for time and one for frequency
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle(
    "Pulse Propagation with Dispersion and Non-Linearity (Normalized Relative to Initial Peak Power)"
)

# Color and line styles for distinct visualization
colors = ["blue", "orange", "green", "red"]
line_styles = ["-", "--", "-.", ":"]

# Propagate and plot for each specified distance
for idx, z in enumerate(z_values):
    Nseg = int(z / dz)  # Number of segments for the given distance
    A_out = split_step_fourier(A_initial, dz, beta2, gamma, alpha, Nseg, freq_vector)

    # Normalize in time domain by the initial peak power (1 W)
    power_time = np.abs(A_out) ** 2 / initial_peak_power

    # Time Domain Plot
    ax1.plot(
        time_vector * 1e12,
        power_time,
        label=f"z = {z:.4f} km",
        # color=colors[idx],
        linestyle=line_styles[idx],
    )
    ax1.set_xlabel("Time (ps)")
    ax1.set_ylabel("Normalized Power (Initial Peak = 1)")
    ax1.set_title("Time Domain Pulse Evolution")
    ax1.set_xlim(-100, 100)
    ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    # Normalize in frequency domain by the initial peak spectral density
    A_freq_out = np.fft.fftshift(np.fft.fft(A_out))
    power_spectrum = np.abs(A_freq_out) ** 2 / initial_peak_spectral_density

    # Frequency Domain Plot
    ax2.plot(
        freq_vector * 1e-9,
        power_spectrum,
        label=f"z = {z:.4f} km",
        # color=colors[idx],
        linestyle=line_styles[idx],
    )
    ax2.set_xlabel("Frequency (GHz)")
    ax2.set_ylabel("Normalized Power Spectral Density (Initial Peak = 1)")
    ax2.set_title("Frequency Domain Pulse Evolution")
    ax2.set_xlim(-100, 100)
    ax2.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

# Add legends and adjust layout
ax1.legend()
ax2.legend()
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

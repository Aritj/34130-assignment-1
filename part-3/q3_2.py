import numpy as np
import matplotlib.pyplot as plt

from q3_1 import split_step

# Set figure DPI to 300 (increasing plot resolution)
plt.rcParams["savefig.dpi"] = 300

# Define constants
TW = 2500e-12  # Total time window (2500 ps)
N = 2**14  # Number of samples
N_seg = 5000  # Number of segments

# Time and frequency vectors
t = np.linspace(-TW / 2, TW / 2, N)
fsa = 1 / (t[1] - t[0])  # Sampling frequency
f = np.linspace(-fsa / 2, fsa / 2, N)
w = 2 * np.pi * f  # Angular frequency

# Fiber and pulse parameters
T_FWHM = 10e-12  # 10 ps
A0 = 1  # W^(1/2)
T0 = T_FWHM / (2 * np.sqrt(np.log(2)))
alpha = 0.0461  # km^-1
gamma = 1.25  # W^-1 km^-1
beta2 = 0  # ps^2/km
beta3 = 0  # ps^3/km
z_values = [1.2945, 4.1408, 7.4172, 11.2775]  # km


# Function to create input pulse
def create_pulse(t, A0, T0):
    return A0 * np.exp(-(t**2) / (2 * T0**2))


def analytical_nonlinear(t, A0, T0, z, alpha, gamma):
    L_eff = (1 - np.exp(-alpha * z)) / alpha
    P0 = np.abs(A0) ** 2
    return np.abs(A0) * np.exp(-alpha * z / 2) * np.exp(1j * gamma * P0 * L_eff)


# Function to plot all z-values in time and frequency domains
def plot_all_z(t, f, A_in, z_values, w, beta2, beta3, alpha, gamma, N_seg):
    plt.figure(figsize=(12, 8))

    # Time domain subplot
    plt.subplot(2, 1, 1)
    plt.plot(
        t * 1e12, np.abs(A_in) ** 2 / np.max(np.abs(A_in) ** 2), "k--", label="Input"
    )
    for z in z_values:
        A_out = split_step(A_in, z, w, beta2, beta3, alpha, gamma, N_seg)
        plt.plot(
            t * 1e12,
            np.abs(A_out) ** 2 / np.max(np.abs(A_in) ** 2),
            label=f"z = {z:.4f} km",
        )
    plt.xlabel("Time (ps)")
    plt.ylabel("Normalized Power")
    plt.title("Time Domain for Different Propagation Distances")
    plt.legend()
    plt.grid(True)
    plt.xlim(-50, 50)
    plt.ylim(0, 1)

    # Frequency domain subplot
    A_in_w = np.fft.fftshift(np.fft.fft(A_in))
    plt.subplot(2, 1, 2)
    plt.plot(
        f * 1e-12,
        np.abs(A_in_w) ** 2 / np.max(np.abs(A_in_w) ** 2),
        "k--",
        label="Input",
    )
    for z in z_values:
        A_out = split_step(A_in, z, w, beta2, beta3, alpha, gamma, N_seg)
        A_out_w = np.fft.fftshift(np.fft.fft(A_out))
        plt.plot(
            f * 1e-12,
            np.abs(A_out_w) ** 2 / np.max(np.abs(A_in_w) ** 2),
            label=f"z = {z:.4f} km",
        )
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Normalized Power Spectrum")
    plt.title("Frequency Domain for Different Propagation Distances")
    plt.legend()
    plt.grid(True)
    plt.xlim(-0.5, 0.5)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()


def sanity_check(A_in):
    zcheck = z_values[-1]  # Use the longest propagation distance
    A_out_numerical = split_step(A_in, zcheck, w, beta2, beta3, alpha, gamma, N_seg)
    A_out_analytical = A_out_analytical = analytical_nonlinear(
        t, A_in, T0, zcheck, alpha, gamma
    )

    # Normalize the outputs to their own peak values
    A_out_numerical_normalized = np.abs(A_out_numerical) ** 2 / np.max(
        np.abs(A_out_numerical) ** 2
    )
    A_out_analytical_normalized = np.abs(A_out_analytical) ** 2 / np.max(
        np.abs(A_out_analytical) ** 2
    )

    # Plotting the sanity check
    plt.figure(figsize=(8, 5))
    plt.plot(t * 1e12, A_out_numerical_normalized, "b-", label="Simulated")
    plt.plot(t * 1e12, A_out_analytical_normalized, "r--", label="Analytical")
    plt.xlabel("Time (ps)", fontsize=12)
    plt.ylabel("Normalized Power", fontsize=12)
    # plt.title('Sanity Check: Simulated vs. Analytical Output Pulse', fontsize=14, weight='bold')
    plt.legend()
    plt.xlim(-50, 50)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Main function to run the simulation
def main():
    # Create input pulse
    A_in = create_pulse(t, A0, T0)

    # Plot all z-values in both time and frequency domains
    plot_all_z(t, f, A_in, z_values, w, beta2, beta3, alpha, gamma, N_seg)
    sanity_check(A_in)


# Execute the main function if this script is run directly
if __name__ == "__main__":
    main()

import numpy as np
import matplotlib.pyplot as plt

# Set figure DPI to 300 (increasing plot resolution)
plt.rcParams["savefig.dpi"] = 300

# Define constants
TW = 2500e-12  # Total time window (2500 ps)
N = 2**14  # Number of samples

# Time and frequency vectors
t = np.linspace(-TW / 2, TW / 2, N)
fsa = 1 / (t[1] - t[0])  # Sampling frequency
f = np.linspace(-fsa / 2, fsa / 2, N)
w = 2 * np.pi * f  # Angular frequency

# Fiber and pulse parameters
T_FWHM = 10e-12  # 10 ps
A0 = 1  # W^(1/2)
C_values = [-10, 0, 5]
T0 = T_FWHM / (2 * np.sqrt(np.log(2)))

# Dispersive case (Question 3-1)
alpha = 0  # km^-1
gamma = 0  # W^-1 km^-1
beta2 = -21.68e-24  # s^2/km
beta3 = 0  # s^3/km
z_values = [0.3199, 1.6636, 3.3272]  # km
N_seg = 5000


# Function to create input pulse
def create_pulse(t, A0, T0, C):
    return A0 * np.exp(-(1 + 1j * C) * (t**2) / (2 * T0**2))


# Split-step Fourier method
def split_step(A_in, z, w, beta2, beta3, alpha, gamma, N_seg):
    dz = z / N_seg
    A = A_in
    for _ in range(N_seg):
        # Dispersive step (frequency domain)
        A_w = np.fft.fftshift(np.fft.fft(A))
        first_term = 1j * (beta2 / 2) * w**2
        second_term = 0  # assuming beta3 = 0
        A_w *= np.exp((first_term + second_term - alpha / 2) * dz)
        A = np.fft.ifft(np.fft.ifftshift(A_w))

        # Non-linear step (time domain)
        A *= np.exp(1j * gamma * np.abs(A) ** 2 * dz)
    return A


# Analytical dispersive solution
def analytical_dispersive(t, A0, T0, C, z, beta2):
    Q = 1 + ((1j * beta2 * z) / T0**2)
    return (A0 / np.sqrt(Q)) * np.exp(-(1 + 1j * C) * (t**2) / (2 * T0**2 * Q))


# Function to plot separate subplots for each C value
def plot_separate_per_C():
    for C in C_values:
        fig, axs = plt.subplots(2, len(z_values), figsize=(18, 6))
        # fig.suptitle(f'Dispersive Case: C = {C}', fontsize=16, weight='bold')

        A_in = create_pulse(t, A0, T0, C)

        for j, z in enumerate(z_values):
            A_out = split_step(A_in, z, w, beta2, beta3, alpha, gamma, N_seg)

            # Time domain subplot
            axs[0, j].plot(
                t * 1e12,
                np.abs(A_in) ** 2 / np.max(np.abs(A_in) ** 2),
                "b-",
                label=f"Input",
            )
            axs[0, j].plot(
                t * 1e12,
                np.abs(A_out) ** 2 / np.max(np.abs(A_in) ** 2),
                "r--",
                label=f"Output",
            )
            axs[0, j].set_xlim([-200, 200])
            axs[0, j].set_ylim([0, 1])
            axs[0, j].set_xlabel("Time (ps)", fontsize=10)
            axs[0, j].set_ylabel("Normalized Power", fontsize=10)
            axs[0, j].set_title(f"Time Domain (C={C}, z = {z:.4f} km)", fontsize=11)
            axs[0, j].grid()
            axs[0, j].legend(fontsize=9)

            # Frequency domain subplot
            A_in_w = np.fft.fftshift(np.fft.fft(A_in))
            A_out_w = np.fft.fftshift(np.fft.fft(A_out))
            axs[1, j].plot(
                f * 1e-12,
                np.abs(A_in_w) ** 2 / np.max(np.abs(A_in_w) ** 2),
                "b-",
                label="Input",
            )
            axs[1, j].plot(
                f * 1e-12,
                np.abs(A_out_w) ** 2 / np.max(np.abs(A_in_w) ** 2),
                "r--",
                label="Output",
            )
            axs[1, j].set_xlim([-1, 1])
            axs[1, j].set_ylim([0, 1])
            axs[1, j].set_xlabel("Frequency (THz)", fontsize=10)
            axs[1, j].set_ylabel("Normalized Power Spectrum", fontsize=10)
            axs[1, j].set_title(
                f"Frequency Domain (C={C}, z = {z:.4f} km)", fontsize=11
            )
            axs[1, j].grid()
            axs[1, j].legend(fontsize=9)

        plt.tight_layout()  # Leave space for the main title
        plt.show()


# Sanity check for numerical and analytical solution comparison
def sanity_check():
    Ccheck = 0  # Use pulse without chirp for simplification
    zcheck = z_values[-1]  # Use the longest propagation distance
    A_in = create_pulse(t, A0, T0, Ccheck)
    A_out_numerical = split_step(A_in, zcheck, w, beta2, beta3, alpha, gamma, N_seg)
    A_out_analytical = analytical_dispersive(t, A0, T0, Ccheck, zcheck, beta2)

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


# Main function to run the plots
def main():
    plot_separate_per_C()  # Generate separate plots per C value
    sanity_check()


if __name__ == "__main__":
    main()

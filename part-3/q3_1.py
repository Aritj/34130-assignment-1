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
beta_2 = -21.68e-24  # s^2/km
beta_3 = 0
z_values = [0.3199, 1.6636, 3.3272]  # km
N_seg = 5000


def normalize(A: np.ndarray, B: np.ndarray = None) -> np.ndarray:
    # Normalize A w.r.t. B (if provided) else normalize A w.r.t. A
    return A / np.max(B) if B is not None and B.size != 0 else A / np.max(A)


def power_of_pulse(A: np.ndarray) -> np.ndarray:
    return A * np.conj(A)


def create_pulse(t: np.ndarray, A0: int, T0: float, C: int) -> np.ndarray:
    return A0 * np.exp(-(1 + 1j * C) * (t**2) / (2 * T0**2))


def split_step(
    A: np.ndarray,
    z: float,
    w: np.ndarray,
    beta_2: float,
    beta_3: float,
    alpha: float,
    gamma: float,
    N_seg: int,
) -> np.ndarray:
    dz = z / N_seg

    # Calculate dispersive phase
    beta_2_term = 1j * (beta_2 / 2) * w**2
    beta_3_term = 1j * (beta_3 / 6) * w**3
    dispersive_phase = np.exp((beta_2_term + beta_3_term - alpha / 2) * dz)

    for _ in range(N_seg):
        # Dispersive step (frequency domain)
        A_w = np.fft.fftshift(np.fft.fft(A))  # Use fftshift before fft
        A_w *= dispersive_phase  # Apply dispersive phase

        # Non-linear step (time domain)
        A = np.fft.ifft(np.fft.ifftshift(A_w))  # Use ifftshift before ifft
        A *= np.exp(1j * gamma * np.abs(A) ** 2 * dz)

    return A


def analytical_dispersive(
    t: np.ndarray, A0: int, T0: int, C: int, z: float
) -> np.ndarray:
    Q = 1 + ((1j * beta_2 * z) / T0**2)
    return (A0 / np.sqrt(Q)) * np.exp(-(1 + 1j * C) * (t**2) / (2 * T0**2 * Q))


def plot_separate_per_C() -> None:
    for C in C_values:
        fig, axs = plt.subplots(2, len(z_values), figsize=(18, 6))

        # Time domain
        A_in = create_pulse(t, A0, T0, C)
        P_in = power_of_pulse(A_in)  # Calculate power
        P_in_norm = normalize(P_in)  # Normalize Power

        # Frequency domain
        A_in_w = np.fft.fftshift(np.fft.fft(A_in))
        P_in_w = power_of_pulse(A_in_w)
        P_in_w_norm = normalize(P_in_w)

        for j, z in enumerate(z_values):
            # Time Domain calculations
            A_out = split_step(A_in, z, w, beta_2, beta_3, alpha, gamma, N_seg)
            P_out = power_of_pulse(A_out)
            P_out_norm = normalize(P_out, P_in)  # Normalize to input peak

            # Time domain subplot
            axs[0, j].plot(t * 1e12, P_in_norm, "b-", label=f"Input")
            axs[0, j].plot(t * 1e12, P_out_norm, "r--", label=f"Output")
            axs[0, j].set_xlim(-200, 200)
            axs[0, j].set_xlabel("Time (ps)", fontsize=10)
            axs[0, j].set_ylabel("Normalized Power", fontsize=10)
            axs[0, j].set_title(f"Time Domain (C={C}, z = {z:.4f} km)", fontsize=11)
            axs[0, j].grid()
            axs[0, j].legend(fontsize=9)

            # Frequency domain calculations
            A_out_w = np.fft.fftshift(np.fft.fft(A_out))
            P_out_w = power_of_pulse(A_out_w)
            P_out_w_norm = normalize(P_out_w, P_in_w)  # Normalize to input peak

            # Frequency domain subplot
            axs[1, j].plot(f * 1e-12, P_in_w_norm, "b-", label="Input")
            axs[1, j].plot(f * 1e-12, P_out_w_norm, "r--", label="Output")
            axs[1, j].set_xlim(-1, 1)
            axs[1, j].set_xlabel("Frequency (THz)", fontsize=10)
            axs[1, j].set_ylabel("Normalized Power Spectrum", fontsize=10)
            axs[1, j].set_title(
                f"Frequency Domain (C={C}, z = {z:.4f} km)", fontsize=11
            )
            axs[1, j].grid()
            axs[1, j].legend(fontsize=9)

        plt.tight_layout()
        plt.show()


def sanity_check() -> None:
    C_check = 0  # Use pulse without chirp for simplification
    z_check = z_values[-1]  # Use the longest propagation distance

    A_in = create_pulse(t, A0, T0, C_check)  # Input signal
    A_out_numerical = split_step(A_in, z_check, w, beta_2, beta_3, alpha, gamma, N_seg)
    A_out_analytical = analytical_dispersive(t, A0, T0, C_check, z_check)
    P_out_numerical = power_of_pulse(A_out_numerical)
    P_out_analytical = power_of_pulse(A_out_analytical)

    # Normalize the outputs to their own peak values
    A_out_numerical_normalized = normalize(P_out_numerical)
    A_out_analytical_normalized = normalize(P_out_analytical)

    # Plotting the sanity check
    plt.figure(figsize=(8, 5))
    plt.plot(t * 1e12, A_out_numerical_normalized, "b-", label="Simulated")
    plt.plot(t * 1e12, A_out_analytical_normalized, "r--", label="Analytical")
    plt.xlabel("Time (ps)", fontsize=12)
    plt.ylabel("Normalized Power", fontsize=12)
    plt.legend()
    plt.xlim(-50, 50)
    plt.ylim(0, 1)
    plt.grid()
    plt.tight_layout()
    plt.show()


def main() -> None:
    plot_separate_per_C()  # Generate separate plots per C value
    sanity_check()


if __name__ == "__main__":
    main()

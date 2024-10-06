import numpy as np
import matplotlib.pyplot as plt

from q3_1 import power_of_pulse, normalize

# Set figure DPI to 300 (increasing plot resolution)
plt.rcParams["savefig.dpi"] = 300

TW = 2500e-12  # Total time window (2500 ps)
N = 2**14  # Number of samples
N_seg = 5000  # Number of segments

# Specified time and frequency vectors
t = np.linspace(-TW / 2, TW / 2, N)
fsa = 1 / (t[1] - t[0])  # Sampling frequency
f = np.linspace(-fsa / 2, fsa / 2, N)
w = 2 * np.pi * f  # Angular frequency

# Assumption
A0 = 1  # W^(1/2)
P0 = abs(A0) ** 2
gamma = 1.25  # W^-1 km^-1
beta_2 = -21.68e-24  # s^2/km
beta_3 = 0  # s^3/km
L = 20  # km


def create_sech_pulse(t: np.ndarray, A0: float, T0: float) -> np.ndarray:
    return A0 / np.cosh(t / T0)


def split_step(A: np.ndarray, z: float, alpha: float) -> np.ndarray:
    dz = z / N_seg

    # Precompute dispersive phase term (assuming beta_3=0) for efficiency
    dispersive_phase = np.exp((1j * (beta_2 / 2) * w**2 - alpha / 2) * dz)

    for _ in range(N_seg):
        # Dispersive step (frequency domain)
        A_w = np.fft.fftshift(np.fft.fft(A))  # Use fftshift before fft
        A_w *= dispersive_phase

        # Non-linear step (time domain)
        A = np.fft.ifft(np.fft.ifftshift(A_w))  # Use ifftshift before ifft
        A *= np.exp(1j * gamma * np.abs(A) ** 2 * dz)

    return A


def plot_results(
    t: np.ndarray, f: np.ndarray, A_in: np.ndarray, A_out: np.ndarray, title: str
):
    # Calculate the power in Time Domain
    P_in = power_of_pulse(A_in)
    P_out = power_of_pulse(A_out)

    # Normalize the power
    P_in_norm = normalize(P_in)  # normalize to self
    P_out_norm = normalize(P_out, P_in)  # normalize to peak input

    # Plot settings
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t * 1e12, P_in_norm, "b-", label="Input")
    plt.plot(t * 1e12, P_out_norm, "r:", label="Output")
    plt.xlabel("Time (ps)")
    plt.xlim(-40, 40)
    plt.ylabel("Normalized Power Spectrum")
    plt.title(f"{title} (Time Domain)")
    plt.legend()
    plt.grid()

    # Calculate the power in Frequency Domain
    A_in_w = np.fft.fftshift(np.fft.fft(A_in))
    A_out_w = np.fft.fftshift(np.fft.fft(A_out))

    P_in_w = power_of_pulse(A_in_w)
    P_out_w = power_of_pulse(A_out_w)

    # Normalize the power
    P_in_w_norm = normalize(P_in_w)  # normalize to self
    P_out_w_norm = normalize(P_out_w, P_in_w)  # normalize to peak input

    # Plot settings
    plt.subplot(2, 1, 2)
    plt.plot(f * 1e-9, P_in_w_norm, "b-", label="Input")
    plt.plot(f * 1e-9, P_out_w_norm, "r:", label="Output")
    plt.xlabel("Frequency (GHz)")
    plt.xlim(-100, 100)
    plt.ylabel("Normalized Power Spectrum")
    plt.title(f"{title} (Frequency Domain)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


def main() -> None:
    # Calculate pulse width for the soliton
    T0_sech_pulse = np.sqrt(np.abs(beta_2) / (gamma * P0))  # hyperbolic secant pulse
    print(f"T0 for the sech pulse is {T0_sech_pulse*1e12} ps")
    quit()
    # Create the initial sech pulse
    A_in = create_sech_pulse(t, A0, T0_sech_pulse)

    # Propagation without loss
    alpha = 0  # km^-1
    A_out = split_step(A_in, L, alpha)
    plot_results(t, f, A_in, A_out, f"Sech Pulse Propagation (alpha={alpha}, z={L})")

    # Introduce loss
    alpha = 0.0461  # km^-1
    A_out_l = split_step(A_in, L, alpha)
    plot_results(t, f, A_in, A_out_l, f"Sech Pulse Propagation (alpha={alpha}, z={L})")


if __name__ == "__main__":
    main()

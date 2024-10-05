import numpy as np
import matplotlib.pyplot as plt

# Set figure DPI to 300 (increasing plot resolution)
plt.rcParams["savefig.dpi"] = 300

# Total time window (2500 ps)
TW = 2500e-12
# Number of samples
N = 2**14
N_seg = 5000

# Specified time and frequency vectors
t = np.linspace(-TW / 2, TW / 2, N)
fsa = 1 / (t[1] - t[0])  # Sampling frequency
f = np.linspace(-fsa / 2, fsa / 2, N)
w = 2 * np.pi * f  # Angular frequency

# Fiber and pulse parameters
T_FWHM = 10e-12  # 10 ps
T0 = T_FWHM / (2 * np.sqrt(np.log(2)))

A0 = 1  # W^(1/2)
gamma = 1.25  # W^-1 km^-1
beta2 = -21.68e-24  # s^2/km
beta3 = 0  # s^3/km
L = 20  # km


# Create sech pulse
def create_sech_pulse(t, A0, T0):
    return A0 / np.cosh(t / T0)


# Split-step Fourier Method function
def split_step(A_in, z, w, beta2, beta3, alpha, gamma, N_seg):
    dz = z / N_seg
    A = A_in
    for _ in range(N_seg):
        # Dispersive step (frequency domain)
        A_w = np.fft.fftshift(np.fft.fft(A))
        first_term = 1j * (beta2 / 2 * w**2)
        second_term = 0  # assuming beta_3 = 0
        A_w *= np.exp((first_term + second_term - alpha / 2) * dz)
        A = np.fft.ifft(np.fft.ifftshift(A_w))

        # Non-linear step (time domain)
        A = A * np.exp(1j * gamma * np.abs(A) ** 2 * dz)
    return A


# Function to plot results
def plot_results(t, f, A_in, A_out, title_str):
    plt.figure(figsize=(10, 6))
    # Time domain
    plt.subplot(2, 1, 1)
    plt.plot(
        t * 1e12, np.abs(A_in) ** 2 / np.max(np.abs(A_in) ** 2), "b-", label="Input"
    )
    plt.plot(
        t * 1e12, np.abs(A_out) ** 2 / np.max(np.abs(A_in) ** 2), "r:", label="Output"
    )
    plt.xlabel("Time (ps)")
    plt.xlim(-40, 40)
    plt.ylabel("Normalized Power")
    plt.title(f"{title_str} (Time Domain)")
    plt.legend()
    plt.grid(True)

    # Frequency domain
    plt.subplot(2, 1, 2)
    A_in_w = np.fft.fftshift(np.fft.fft(A_in))
    A_out_w = np.fft.fftshift(np.fft.fft(A_out))
    plt.plot(
        f * 1e-9,
        np.abs(A_in_w) ** 2 / np.max(np.abs(A_in_w) ** 2),
        "b-",
        label="Input",
    )
    plt.plot(
        f * 1e-9,
        np.abs(A_out_w) ** 2 / np.max(np.abs(A_in_w) ** 2),
        "r:",
        label="Output",
    )
    plt.xlabel("Frequency (GHz)")
    plt.xlim([-100, 100])
    plt.ylabel("Normalized Power Spectrum")
    plt.title(f"{title_str} (Frequency Domain)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main() -> None:
    # Calculate pulse width for the soliton
    P0 = abs(A0) ** 2
    T0_sech_pulse = np.sqrt(abs(beta2) / (gamma * P0))  # hyperbolic secant pulse
    print(f"T0 for the sech pulse is {T0_sech_pulse:.2e}")

    # Create the initial sech pulse
    A_in = create_sech_pulse(t, A0, T0_sech_pulse)

    # Propagation without loss
    alpha = 0  # km^-1
    A_out = split_step(A_in, L, w, beta2, beta3, alpha, gamma, N_seg)
    plot_results(t, f, A_in, A_out, f"Sech Pulse Propagation (alpha={alpha})")

    # Introduce loss
    alpha = 0.0461  # km^-1
    A_out_loss = split_step(A_in, L, w, beta2, beta3, alpha, gamma, N_seg)
    plot_results(t, f, A_in, A_out_loss, f"Sech Pulse Propagation (alpha={alpha})")


if __name__ == "__main__":
    main()

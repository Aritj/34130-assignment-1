import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tabulate import tabulate
from q1_1 import N, C_VALUES, A0, sampling_and_frequency_params
from q1_2 import t, T0, electrical_field_envelope, power_of_pulse

# Set figure DPI to 300 (increasing plot resolution)
plt.rcParams["savefig.dpi"] = 300

# Use the sampling period calculated in Q1-1
T_sa, F_sa, Delta_F, F_min = sampling_and_frequency_params()

# Creating frequency vector and shifting zero frequency component to the center
f = np.fft.fftfreq(N, T_sa)
f = np.fft.fftshift(f)


def normalize(A: np.ndarray, B: np.ndarray = None) -> np.ndarray:
    # Normalize A w.r.t. B (if provided) else normalize A w.r.t. A
    return A / np.max(B) if B is not None and B.size != 0 else A / np.max(A)


def normalized_power_spectrum(A_t: np.ndarray) -> np.ndarray:
    A_f = np.fft.fft(A_t)  # FFT time domain -> frequency domain
    A_f = np.fft.fftshift(A_f)  # Shift FFT
    P_f = power_of_pulse(A_f)  # Calculate power of pulse
    return normalize(P_f)  # Normalize the power spectrum


def measure_FWHM(t: np.ndarray, P: np.ndarray) -> np.float64:
    indices = np.where(P >= np.max(P) / 2)[0]
    return t[indices[-1]] - t[indices[0]]  # Return the difference (FWHM)


def main() -> None:
    # Calculate the field envelope A_t and power P_t for every C
    A_t_list = [electrical_field_envelope(A0, T0, C, t) for C in C_VALUES]
    P_f_list = [normalized_power_spectrum(A_t) for A_t in A_t_list]

    # Measure the FWHM width (in GHz) of the spectra
    measured_fwhm_list = [measure_FWHM(f / 1e9, P_f) for P_f in P_f_list]

    # Print the measured values in tabular form
    print(
        tabulate(
            pd.DataFrame({"C": C_VALUES, "Measured FWHM (GHz)": measured_fwhm_list}),
            headers="keys",
            tablefmt="psql",
            showindex=False,
        )
    )

    # Plot normalized power spectrum for each chirp value
    plt.figure(figsize=(10, 6))

    # Plot for each chirp value
    for i, C in enumerate(C_VALUES):
        plt.plot(f / 1e9, P_f_list[i], label=f"Chirp C={C}")

    # Plot settings
    plt.xlim(-750, 750)
    plt.ylim(0, 1)
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Normalized Power")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

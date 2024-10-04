import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tabulate import tabulate
from q1_1 import N, TW, sampling_and_frequency_params
from q1_2 import (
    C_VALUES,
    A0,
    T0,
    t,
    electrical_field_envelope,
    power_of_pulse,
)

# Use the sampling period calculated in Q1-1
T_sa, F_sa, Delta_F, F_min = sampling_and_frequency_params(N, TW)

# Creating frequency vector and shifting zero frequency component to the center
FREQ_VECTOR = np.fft.fftfreq(N, T_sa)
FREQ_VECTOR = np.fft.fftshift(FREQ_VECTOR)


def normalized_power_spectrum(A_t: np.ndarray) -> np.ndarray:
    # a) Calculate the spectra of electical field envelope
    A_f = np.fft.fft(A_t)  # FFT of the time domain signal
    A_f = np.fft.fftshift(A_f)  # Shift FFT
    # b) Calculate the power spectra
    P_f = power_of_pulse(A_f)  # Power spectrum |A(f)|^2
    P_f_normalized = P_f / np.max(P_f)  # Normalize the power spectrum
    return P_f_normalized


def measure_FWHM(t, P):
    half_max = np.max(P) / 2
    indices = np.where(P >= half_max)[0]
    return t[indices[-1]] - t[indices[0]]  # Return the difference (FWHM)


def main() -> None:
    # a, b) Calculate the field envelope A_t and power P_t for every C
    A_t_list = [electrical_field_envelope(A0, T0, C, t) for C in C_VALUES]
    P_f_list = [normalized_power_spectrum(A_t) for A_t in A_t_list]

    # d) Measure the FWHM width (in GHz) of the spectra
    measured_fwhm_list = [measure_FWHM(FREQ_VECTOR / 1e9, P_f) for P_f in P_f_list]

    # Print the measured values in tabular form
    print(
        tabulate(
            pd.DataFrame({"C": C_VALUES, "Measured FWHM (GHz)": measured_fwhm_list}),
            headers="keys",
            tablefmt="psql",
            showindex=False,
        )
    )

    # c) Plot normalized power spectrum for each chirp value
    plt.figure(figsize=(10, 6))

    # Plot each chirp value in a single plot
    for i, C in enumerate(C_VALUES):
        plt.plot(FREQ_VECTOR / 1e9, P_f_list[i], label=f"Chirp C={C}")

    # Plot settings
    plt.xlim(-750, 750)
    plt.ylim(0, 1)
    # plt.title("Normalized Power Spectrum vs Frequency for Different Chirp Values")
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Normalized Power")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

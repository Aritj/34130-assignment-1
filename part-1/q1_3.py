import numpy as np
import matplotlib.pyplot as plt

from q1_1 import N, calculate_sampling_and_frequency_params
from q1_2 import C_VALUES, A0, T0, TIME_VECTOR, calculate_gaussian_field

# Use the sampling period calculated in Q1-1
T_sa, _, _, _ = calculate_sampling_and_frequency_params()

# Frequency vector creation
FREQ_VECTOR = np.fft.fftfreq(N, T_sa)  # Create the frequency vector
FREQ_VECTOR = np.fft.fftshift(FREQ_VECTOR)  # Shift zero frequency component to the center

def calculate_normalized_power_spectrum(A_t):
    # a) Calculate the spectra of electical field envelope
    A_f = np.fft.fft(A_t)  # FFT of the time domain signal
    A_f = np.fft.fftshift(A_f)  # Shift FFT
    # b) Calculate the power spectra
    P_f = np.abs(A_f)**2  # Power spectrum |A(f)|^2
    P_f_normalized = P_f / np.max(P_f)  # Normalize the power spectrum
    return P_f_normalized

def calculate_FWHM(t, P):
    half_max = np.max(P) / 2
    indices = np.where(P >= half_max)[0]
    return t[indices[-1]] - t[indices[0]] # Return the difference (FWHM)

def main() -> None:
    # Plot the spectra for each chirp value
    plt.figure(figsize=(10, 6))
    print(f"| {'C':^5} | {'FWHM':^18} |")
    print("-" * 30)

    for C in C_VALUES:
        A_t = calculate_gaussian_field(A0, T0, C, TIME_VECTOR)
        P_f_normalized = calculate_normalized_power_spectrum(A_t)
        fwhm = calculate_FWHM(FREQ_VECTOR, P_f_normalized)
        
        # c) Plot normalized power spectrum
        plt.plot(FREQ_VECTOR / 1e9, P_f_normalized, label=f'Chirp C={C}')

        # d) Measure the FWHM width of the spectra
        print(f"| {C:^5d} | {fwhm/1e9:^18.2f} |")

    xscale = 1500
    plt.xlim(-xscale, xscale)
    plt.ylim(0, 1)

    # Plot settings
    plt.title('Normalized Power Spectrum P(0,f) vs Frequency for Different Chirp Values')
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Normalized Power Spectrum')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
import numpy as np
import matplotlib.pyplot as plt

from q1_1 import N, calculate_sampling_and_frequency_params
from q1_2 import C_VALUES, A0, T0, TIME_VECTOR, gaussian_field

T_sa, F_sa, ΔF, F_min = calculate_sampling_and_frequency_params()

# Frequency vector creation
FREQ_VECTOR = np.fft.fftfreq(N, T_sa)  # Create the frequency vector
FREQ_VECTOR = np.fft.fftshift(FREQ_VECTOR)  # Shift zero frequency component to the center


# Function to calculate FFT and power spectrum
def compute_spectrum(A_t):
    A_f = np.fft.fft(A_t)  # FFT of the time domain signal
    A_f = np.fft.fftshift(A_f)  # Shift FFT
    P_f = np.abs(A_f)**2  # Power spectrum |A(f)|^2
    P_f_normalized = P_f / np.max(P_f)  # Normalize the power spectrum
    return P_f_normalized

def main() -> None:
    # Plot the spectra for each chirp value
    plt.figure(figsize=(10, 6))
    for C in C_VALUES:
        A_t = gaussian_field(A0, T0, C, TIME_VECTOR)
        P_f_normalized = compute_spectrum(A_t)
        
        # Plot normalized power spectrum
        plt.plot(FREQ_VECTOR / 1e9, P_f_normalized, label=f'Chirp C={C}')

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
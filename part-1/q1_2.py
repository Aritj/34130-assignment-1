import numpy as np
import matplotlib.pyplot as plt

from q1_1 import TW, N, calculate_sampling_and_frequency_params

_, F_sa, _, _ = calculate_sampling_and_frequency_params()

# Assumptions
TFWHM = 10  # Full Width Half Maximum in ps
C_VALUES = [-10, 0, +5]  # Chirp parameters
T0 = TFWHM / (2 * np.sqrt(2 * np.log(2)))  # T0 in ps
A0 = 1  # Peak amplitude (W^1/2)

# Creating the time vector
TIME_VECTOR = np.linspace(-TW / 2, TW / 2, N)


def gaussian_field(A0, T0, C, t):
    first_term = (1 + 1j * C) / 2 # 1j is a complex number in Python
    second_term = (t / T0)**2
    
    return A0 * np.exp(-(first_term * second_term))

def main() -> None:
    # Calculate field envelope and power for each chirp value
    plt.figure(figsize=(10, 6))
    for C in C_VALUES:
        A_t = gaussian_field(A0, T0, C, TIME_VECTOR)
        P_t = np.abs(A_t)**2  # Power is |A(0,t)|^2
        
        # Plot power
        plt.plot(TIME_VECTOR, P_t, label=f'Chirp C={C}')
        
    # Set plot scale
    xscale = 20
    plt.xlim(-xscale, xscale)

    # Plot settings
    plt.title('Power P(0,t) vs Time for Different Chirp Values')
    plt.xlabel('Time (ps)')
    plt.ylabel('Power (W)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
if __name__ == '__main__':
    main()
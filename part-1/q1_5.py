import numpy as np
import matplotlib.pyplot as plt

from q1_1 import N, calculate_sampling_and_frequency_params
from q1_2 import C_VALUES, A0, T0, TIME_VECTOR
from q1_3 import calculate_FWHM

# Fiber parameters
beta_2 = -21.68  # Dispersion parameter (ps^2/km)
Z_VALUES = [0, 0.3199, 1.6636, 3.3272]  # Propagation distances in km

# Define Gaussian pulse with chirp
def initial_pulse(t, A0, T0, C):
    return A0 * np.exp(-((1 + 1j * C) * (t / T0)**2))

def calculate_pulse_evolution(z_values):
    # Compute the evolution for each chirp value and distance
    results = {}  # Store results for analysis
    T_sa, _, _, _ = calculate_sampling_and_frequency_params()
    frequency_vector = np.fft.fftfreq(N, T_sa * 1e12)  # Frequency vector in Hz
    omega_vector = 2 * np.pi * frequency_vector  # Angular frequency

    for C in C_VALUES:
        results[C] = {}
        A_0t = initial_pulse(TIME_VECTOR, A0, T0, C)  # Initial pulse at z=0
        A_0f = np.fft.fft(A_0t)  # Frequency domain representation

        # Calculate for each distance
        for z in z_values:
            phase_shift = np.exp(1j * (beta_2 / 2) * omega_vector**2 * z)
            A_zf = A_0f * phase_shift  # Apply phase shift
            A_zt = np.fft.ifft(A_zf)  # Convert back to time domain
            P_zt = np.abs(A_zt)**2  # b) Calculate power
            results[C][z] = (TIME_VECTOR, A_zt, P_zt)  # Store results

    return results

def main():
    results = calculate_pulse_evolution(Z_VALUES)
    
    # Print table header
    print(f"| {'C Value':^10} | {'Z Value (km)':^15} | {'T_FWHM (ps)':^15} |")
    
    # Calculate and display FWHM in a table format
    for C in C_VALUES:
        print("-" * 50)
        for z in Z_VALUES:
            time_vector, A_zt, P_zt = results[C][z]
            fwhm = calculate_FWHM(time_vector, P_zt)
            print(f"| {C:^10} | {z:^15.4f} | {fwhm:^15.2f} |")
    
    # Modified Plotting: Plot all C values for each z value
    fig, axs = plt.subplots(1, len(Z_VALUES), figsize=(18, 6))  # Create one row of subplots for each z value
    for j, z in enumerate(Z_VALUES):
        xscale = 500
        axs[j].set_xlim(-xscale, xscale)
        axs[j].set_ylim(0, 0.25)

        # Plot all C values for this z value
        for C in C_VALUES:
            time_vector, A_zt, P_zt = results[C][z]
            axs[j].plot(time_vector, P_zt, label=f'C={C}')
        
        axs[j].set_xlabel('Time (ps)')
        axs[j].set_ylabel('Power')
        axs[j].legend()
        axs[j].set_title(f'Power vs Time\nz={z} km')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()

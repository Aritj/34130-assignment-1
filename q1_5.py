import numpy as np
import matplotlib.pyplot as plt

from q1_3 import FREQ_VECTOR
from q1_2 import C_VALUES, A0, T0, TIME_VECTOR, gaussian_field

# Constants
beta_2 = -21.68 * 1e-12  # beta_2 in s^2/km
z_values = [0.3199, 1.6636, 3.3272]  # Distances in km

def propagate_in_frequency(A_f, beta_2, z, FREQ_VECTOR):
    phase_shift = np.exp(1j * (beta_2 / 2) * (2 * np.pi * FREQ_VECTOR)**2 * z)
    return A_f * phase_shift

def fwhm(t, P_t):
    half_max = np.max(P_t) / 2.0
    indices_above_half_max = np.where(P_t >= half_max)[0]
    t_min = t[indices_above_half_max[0]]
    t_max = t[indices_above_half_max[-1]]
    
    return t_max - t_min

def calculate_fwhm_table(C_VALUES, z_values, FREQ_VECTOR, beta_2, A0, T0, TIME_VECTOR):
    fwhm_table = {C: [] for C in C_VALUES}
    for C in C_VALUES:
        A_t = gaussian_field(A0, T0, C, TIME_VECTOR)
        A_f = np.fft.fftshift(np.fft.fft(A_t))  # FFT to frequency domain
        
        for z in z_values:
            # Propagate the field in the frequency domain
            A_f_propagated = propagate_in_frequency(A_f, beta_2, z, FREQ_VECTOR)
            
            # Inverse FFT to get back to time domain
            A_t_propagated = np.fft.ifft(np.fft.ifftshift(A_f_propagated))
            
            # Calculate power in time domain
            P_t = np.abs(A_t_propagated)**2
            
            # Calculate FWHM and store in table
            temporal_fwhm = fwhm(TIME_VECTOR, P_t)
            fwhm_table[C].append(temporal_fwhm)

    return fwhm_table

# Plotting function for Q1-5
def plot_power(C_VALUES, z_values, FREQ_VECTOR, beta_2, A0, T0, TIME_VECTOR):
    for C in C_VALUES:
        A_t = gaussian_field(A0, T0, C, TIME_VECTOR)
        A_f = np.fft.fftshift(np.fft.fft(A_t))  # FFT to frequency domain
        
        plt.figure(figsize=(10, 6))
        for z in z_values:
            # Propagate the field in the frequency domain
            A_f_propagated = propagate_in_frequency(A_f, beta_2, z, FREQ_VECTOR)
            
            # Inverse FFT to get back to time domain
            A_t_propagated = np.fft.ifft(np.fft.ifftshift(A_f_propagated))
            
            # Calculate power in time domain
            P_t = np.abs(A_t_propagated)**2
            
            # Plot power for this z
            plt.plot(TIME_VECTOR, P_t, label=f'z={z:.4f} km')
        
        # Set the plot scale
        xscale = 75
        plt.xlim(-xscale, xscale)
        
        # Plot settings
        plt.title(f'Power P(z,t) vs Time for Chirp C={C}')
        plt.xlabel('Time (ps)')
        plt.ylabel('Power (W)')
        plt.legend()
        plt.grid(True)
        plt.show()

def main():
    # Call the FWHM calculation function
    fwhm_table = calculate_fwhm_table(C_VALUES, z_values, FREQ_VECTOR, beta_2, A0, T0, TIME_VECTOR)

    # Plot power for each chirp and distance
    plot_power(C_VALUES, z_values, FREQ_VECTOR, beta_2, A0, T0, TIME_VECTOR)
    
    # Example of printing FWHM results
    print(f"{'C':<8}{'TFWHM(z=0)':>15}{'TFWHM(z1)':>15}{'TFWHM(z2)':>15}{'TFWHM(z3)':>15}")
    for C in C_VALUES:
        fwhm_z0 = fwhm(TIME_VECTOR, np.abs(gaussian_field(A0, T0, C, TIME_VECTOR))**2)
        fwhm_z1, fwhm_z2, fwhm_z3 = fwhm_table[C]
        print(f"{C:<8}{fwhm_z0:>15.3f}{fwhm_z1:>15.3f}{fwhm_z2:>15.3f}{fwhm_z3:>15.3f}")

if __name__ == '__main__':
    main()

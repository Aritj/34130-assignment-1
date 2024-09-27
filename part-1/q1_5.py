import numpy as np
import matplotlib.pyplot as plt
from q1_2 import C_VALUES, A0, T0, TIME_VECTOR, gaussian_field
from q1_3 import FREQ_VECTOR

# Given parameters
β2 = -21.68  # ps^2/km
Z_VALUES = [0.3199, 1.6636, 3.3272]  # km

def propagate_pulse(A_0_t, z):
    A_0_f = np.fft.fftshift(np.fft.fft(A_0_t))
    ω = 2 * np.pi * FREQ_VECTOR
    H = np.exp(1j * (β2/2) * ω**2 * z)
    A_z_f = A_0_f * H
    return np.fft.ifft(np.fft.ifftshift(A_z_f))

def calculate_fwhm(t, P):
    half_max = np.max(P) / 2
    indices = np.where(P >= half_max)[0]
    return t[indices[-1]] - t[indices[0]]

def main():    
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    
    for i, C in enumerate(C_VALUES):
        A_0_t = gaussian_field(A0, T0, C, TIME_VECTOR)
        P_0_t = np.abs(A_0_t)**2
        
        for j, z in enumerate(Z_VALUES):
            A_z_t = propagate_pulse(A_0_t, z)
            P_z_t = np.abs(A_z_t)**2
            
            axs[i, j].plot(TIME_VECTOR, P_z_t / np.max(P_0_t))
            axs[i, j].set_title(f'C={C}, z={z} km')
            axs[i, j].set_xlabel('Time (ps)')
            axs[i, j].set_ylabel('Normalized Power')
            axs[i, j].set_xlim(-50, 50)
            
            fwhm = calculate_fwhm(TIME_VECTOR, P_z_t)
            print(f"C={C}, z={z} km, FWHM={fwhm:.2f} ps")
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()

import numpy as np

from q1_2 import C_VALUES, A0, T0, TIME_VECTOR, calculate_gaussian_field
from q1_3 import FREQ_VECTOR, calculate_normalized_power_spectrum, calculate_FWHM

def theoretical_FWHM(T0_ps, C):
    return (np.sqrt(np.log(2)) / (np.pi * T0_ps * 1e-12)) * np.sqrt(1 + C**2)

def main():
    print("| Chirp | Measured FWHM (GHz) | Theoretical FWHM (GHz) |")
    print("-" * 56)
    
    for C in C_VALUES:
        theoretical_value = theoretical_FWHM(T0, C)
        A_t = calculate_gaussian_field(A0, T0, C, TIME_VECTOR)
        P_f_normalized = calculate_normalized_power_spectrum(A_t)
        numerical_value = calculate_FWHM(FREQ_VECTOR, P_f_normalized)
        
        print(f"| {C:^5d} | {numerical_value/1e9:^19.2f} | {theoretical_value/1e9:^22.2f} |")


if __name__ == '__main__':
    main()
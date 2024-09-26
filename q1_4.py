import numpy as np

from q1_2 import C_VALUES, A0, T0, TIME_VECTOR, gaussian_field
from q1_3 import FREQ_VECTOR, compute_spectrum

def theoretical_FWHM(T0_ps, C):
    return (np.sqrt(np.log(2)) / (np.pi * T0_ps * 1e-12)) * np.sqrt(1 + C**2)

def compute_numerical_FWHM(freq_vector, P_f_normalized):
    half_max = np.max(P_f_normalized) / 2.0
    indices_above_half_max = np.where(P_f_normalized >= half_max)[0]
    
    # Find the frequencies at the half-maximum points
    f_min = freq_vector[indices_above_half_max[0]]
    f_max = freq_vector[indices_above_half_max[-1]]
    
    # Return the difference between the two frequencies (FWHM)
    return f_max - f_min

def main():
    print(f'C\tTheoretical value\tNumerical value\t|Δ|')
    for C in C_VALUES:
        theoretical_value = theoretical_FWHM(T0, C)
        
        A_t = gaussian_field(A0, T0, C, TIME_VECTOR)
        P_f_normalized = compute_spectrum(A_t)
        numerical_value = compute_numerical_FWHM(FREQ_VECTOR, P_f_normalized)
        print(f'{C}\t{int(theoretical_value)}\t\t{int(numerical_value)}\t{abs(int(theoretical_value)-int(numerical_value))}')


if __name__ == '__main__':
    main()
import numpy as np
import pandas as pd

from tabulate import tabulate
from q1_2 import C_VALUES, A0, T0, t, calculate_electrical_field_envelope
from q1_3 import FREQ_VECTOR, calculate_normalized_power_spectrum, measure_FWHM


def calculate_F_FWHM(T0_ps, C):
    return (np.sqrt(np.log(2)) / (np.pi * T0_ps * 1e-12)) * np.sqrt(1 + C**2)


def main():
    # a) Verify the spectral widths determined in the previous question
    A_t_list: list[np.ndarray] = [calculate_electrical_field_envelope(A0, T0, C, t) for C in C_VALUES]
    P_f_list: list[np.ndarray] = [calculate_normalized_power_spectrum(A_t) for A_t in A_t_list]
    measured_fwhm_list = [measure_FWHM(FREQ_VECTOR / 1e9, P_f) for P_f in P_f_list]
    theoretical_fwhm_list = [calculate_F_FWHM(T0, C) / 1e9 for C in C_VALUES]

    # Print table with measured and theoretical FWHM values for each C value
    print(tabulate(pd.DataFrame(
        {
        "C": C_VALUES,
        "Measured FHWM (GHz)": measured_fwhm_list,
        "F_FWHM (GHz)": theoretical_fwhm_list
        }
    ), headers="keys", tablefmt="psql", showindex=False))

if __name__ == "__main__":
    main()

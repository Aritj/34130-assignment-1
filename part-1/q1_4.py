import numpy as np
import pandas as pd

from tabulate import tabulate
from q1_2 import C_VALUES, A0, T0, TIME_VECTOR, calculate_gaussian_field
from q1_3 import FREQ_VECTOR, calculate_normalized_power_spectrum, calculate_FWHM


def calculate_F_FWHM(T0_ps, C):
    return (np.sqrt(np.log(2)) / (np.pi * T0_ps * 1e-12)) * np.sqrt(1 + C**2)


def main():
    results = []

    for C in C_VALUES:
        A_t = calculate_gaussian_field(A0, T0, C, TIME_VECTOR)
        P_f_normalized = calculate_normalized_power_spectrum(A_t)

        results.append(
            {
                "C": C,
                "Measured FHWM (GHz)": calculate_FWHM(FREQ_VECTOR, P_f_normalized)
                / 1e9,
                "F_FWHM (GHz)": calculate_F_FWHM(T0, C) / 1e9,
            }
        )

    print(tabulate(pd.DataFrame(results), headers="keys", tablefmt="psql"))


if __name__ == "__main__":
    main()

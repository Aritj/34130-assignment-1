import numpy as np
import matplotlib.pyplot as plt

from q1_1 import TW, N


# Assumptions
T_FWHM = 10  # Full Width Half Maximum in ps
C_VALUES = [-10, 0, +5]  # Chirp parameters
T0 = T_FWHM / (2 * np.sqrt(np.log(2)))  # T0 in ps  
A0 = 1  # Peak amplitude (W^1/2)

# Creating the time vector
t = np.linspace(-TW / 2, TW / 2, N)


def calculate_electrical_field_envelope(A0: int, T0: float, C: list[int], t: np.ndarray) -> np.ndarray:
    '''Calculates pre-chirped Gaussian field envelope'''
    return A0 * np.exp(-((1 + 1j * C) / 2) * (t / T0) ** 2)


def calculate_power_of_pulse(A_t: np.ndarray) -> np.ndarray:
    '''Calculates the power of the pulses as |A(0,t)|^2'''
    return np.abs(A_t) ** 2


def main() -> None:
    # a) Calculate the field envelope A_t and power P_t for every C
    A_t_list: list[np.ndarray] = [calculate_electrical_field_envelope(A0, T0, C, t) for C in C_VALUES]
    P_t_list: list[np.ndarray] = [calculate_power_of_pulse(A_t) for A_t in A_t_list]

    # Plot P_t for every C
    line_styles = ["--", "-.", ":"] # to highlight the overlapping plots
    plt.figure(figsize=(10, 6))

    # Plot each chirp value in a single plot
    for i, C in enumerate(C_VALUES):
        plt.plot(t, P_t_list[i], linestyle=line_styles[i], label=f"Chirp C={C}")

    # Plot settings
    plt.xlim(-15, 15)
    plt.ylim(0, 1)
    plt.title("Power P(0,t) vs Time for Different Chirp Values")
    plt.xlabel("Time (ps)")
    plt.ylabel("Power (W)")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()

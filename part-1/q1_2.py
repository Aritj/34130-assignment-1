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


def electrical_field_envelope(
    A0: int, T0: float, C: list[int], t: np.ndarray
) -> np.ndarray:
    """Calculates pre-chirped Gaussian field envelope"""
    return A0 * np.exp(-((1 + 1j * C) / 2) * (t / T0) ** 2)


def power_of_pulse(A_t: np.ndarray) -> np.ndarray:
    """Calculates the power spectrum"""
    return A_t * np.conjugate(A_t)


def main() -> None:
    # a) Calculate the field envelope A_t and power P_t for every C
    A_t_list = [electrical_field_envelope(A0, T0, C, t) for C in C_VALUES]
    P_t_list = [power_of_pulse(A_t) for A_t in A_t_list]

    # Plot P_t for every C
    line_styles = ["--", "-.", ":"]  # to highlight the overlapping plots
    plt.figure()
    """
    # Plot A_t (real) for each chirp value
    plt.subplot(1, 3, 1)
    for i, C in enumerate(C_VALUES):
        plt.plot(t, np.real(A_t_list[i]), "-", label=f"Chirp C={C}")
    plt.xlim(-20, 20)
    plt.ylim(-1, 1)
    plt.title(f"Electrical Field Envelope (real) vs Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (W^0.5)")
    plt.legend()
    plt.grid(True)

    # Plot A_t (imaginary) for each chirp value
    plt.subplot(1, 3, 2)
    for i, C in enumerate(C_VALUES):
        plt.plot(t, np.imag(A_t_list[i]), "-", label=f"Chirp C={C}")
    plt.xlim(-20, 20)
    plt.ylim(-1, 1)
    plt.title(f"Electrical Field Envelope (imaginary) vs Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (W^0.5)")
    plt.legend()
    plt.grid(True)
    """
    # Plot P_t for each chirp value
    # plt.subplot(1, 3, 3)
    for i, C in enumerate(C_VALUES):
        plt.plot(t, P_t_list[i], linestyle=line_styles[i], label=f"Chirp C={C}")
    plt.xlim(-20, 20)
    plt.ylim(0, 1)
    plt.title("Power vs Time")
    plt.xlabel("Time (ps)")
    plt.ylabel("Power (W)")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()

import numpy as np
import matplotlib.pyplot as plt

from q1_1 import TW, N, T_FWHM, C_VALUES, A0

# Set figure DPI to 300 (increasing plot resolution)
plt.rcParams["savefig.dpi"] = 300

# Calculating T0
T0 = T_FWHM / (2 * np.sqrt(np.log(2)))  # T0 in ps

# Creating the time vector
t = np.linspace(-TW / 2, TW / 2, N)


def electrical_field_envelope(A0: int, T0: float, C: int, t: np.ndarray) -> np.ndarray:
    return A0 * np.exp(-((1 + 1j * C) / 2) * (t / T0) ** 2)


def power_of_pulse(A_t: np.ndarray) -> np.ndarray:
    return A_t * np.conjugate(A_t)


def main() -> None:
    # Calculate the field envelope A_t and power P_t for every C
    A_t_list = [electrical_field_envelope(A0, T0, C, t) for C in C_VALUES]
    P_t_list = [power_of_pulse(A_t) for A_t in A_t_list]

    # Plot P_t for every C
    plt.figure()

    # Plot for each chirp value
    for i, C in enumerate(C_VALUES):
        plt.plot(t, P_t_list[i], linestyle=["--", "-.", ":"][i], label=f"Chirp C={C}")

    # Plot settings
    plt.xlim(-20, 20)
    plt.ylim(0, 1)
    plt.title("Power vs Time")
    plt.xlabel("Time (ps)")
    plt.ylabel("Power (W)")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()

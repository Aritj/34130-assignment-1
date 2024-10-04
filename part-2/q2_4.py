import numpy as np
import matplotlib.pyplot as plt
from q2_1 import P0, C, alpha, transmission_distances
from q2_2 import (
    sampling_and_frequency_params,
    generate_time_and_frequency_vectors,
)
from q2_3 import T0, electrical_field_envelope, power_of_pulse, normalize

# Get transmission lengths from Q2-1
transmission_values = transmission_distances()
z_values = transmission_values["Transmission Distance z (km)"]
L_eff_values = transmission_values["Effective Length Leff (km)"]

# Get parameters from Q2-2
T_sa, F_sa, Delta_F, F_min = sampling_and_frequency_params()

# Generate Time and Frequency Vectors
t, freq = generate_time_and_frequency_vectors(T_sa, Delta_F, F_min)
f = np.fft.fftshift(freq)

# Define Gaussian pulse in time and frequency domains
beta_2 = -21.68 * 1e-24  # [s**2 km**-1]
w = 2 * np.pi * f  # Angular frequency


def propogate_pulse(vector, z, L_eff):
    power = power_of_pulse(vector)
    return np.sqrt(power) * np.exp(-alpha * z * 0.5) * np.exp(1j * power * L_eff)


def main():
    A_0_t = electrical_field_envelope(P0, T0, C, t)
    A_0_w = np.fft.fft(A_0_t)

    A_z_t_values = []
    A_z_w_values = []
    for z, L_eff in zip(z_values, L_eff_values):
        A_z_t = propogate_pulse(A_0_t, z, L_eff)
        A_z_w = np.fft.fft(A_z_t)
        A_z_t_values.append(A_z_t)
        A_z_w_values.append(A_z_w)

    # Normalizing the Power Spectra
    P_0_w = power_of_pulse(A_0_w)
    n_w = np.max(P_0_w)
    P_0_w = normalize(P_0_w)

    # Input Power Spectra
    plt.plot(f * 1e-9, P_0_w, label=f"Input C={C}")

    # Output Power Spectra
    for A_z_w, z in zip(A_z_w_values, z_values):
        P_z_w = power_of_pulse(A_z_w)
        P_z_w = P_z_w / np.abs(n_w)  # normalizing with the input max
        plt.plot(f * 1e-9, P_z_w, "-", label=f"Output C={C}, z={round(z,2)} km")

    plt.xlabel("Frequency [GHz]")
    plt.ylabel("Normalized Power")
    plt.xlim(-300, 300)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

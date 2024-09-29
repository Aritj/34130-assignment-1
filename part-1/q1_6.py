import numpy as np
import matplotlib.pyplot as plt

from q1_2 import C_VALUES, T_FWHM
from q1_3 import calculate_FWHM
from q1_5 import beta_2, Z_VALUES, calculate_pulse_evolution

# Define parameters for the calculation
Z_analytical = np.arange(0, 5.001, 0.001)  # Distance values from 0 to 5 km with step size 0.001 km
T0 = T_FWHM / (2 * np.sqrt(np.log(2)))  # Convert FWHM to T0 for Gaussian pulse (T0 = TFWHM / (2 * sqrt(ln(2)))

def analytical_ratio(C, z):
    return np.sqrt(1 + (beta_2 * C * z / T0**2)**2 + (beta_2 * z / T0**2)**2)

def main():
    # Analytical Ratio Calculation
    analytical_ratios = {C: analytical_ratio(C, Z_analytical) for C in C_VALUES}

    # Get the actual FWHM values from the previous question's results
    results = calculate_pulse_evolution(Z_VALUES) # skip z = 0 km
    theoretical_ratios = {C: [] for C in C_VALUES}

    # Calculate the ratios for the actual FWHM values
    for C in C_VALUES:
        initial_FWHM_z0 = calculate_FWHM(results[C][0][0], results[C][0][2])  # Calculate FWHM at z=0 km
        for z in results[C]:
            time_vector, _, P_zt = results[C][z]
            FWHM_z = calculate_FWHM(time_vector, P_zt)
            ratio = FWHM_z / initial_FWHM_z0
            theoretical_ratios[C].append((z, ratio))

    # Plotting
    plt.figure(figsize=(12, 8))
    for C in C_VALUES:
        # Plot analytical ratios
        plt.plot(Z_analytical, analytical_ratios[C], label=f'C={C}')

        # Plot theoretical ratios as scatter points
        z_actual, ratios_actual = zip(*sorted(theoretical_ratios[C]))  # Extract z and ratios
        plt.scatter(z_actual, ratios_actual, label=f'Theoretical (C={C})', marker='x')

    plt.xlabel('Propagation Distance z (km)')
    plt.ylabel(r'$T_{FWHM1}(z) / T_{FWHM}(0)$')
    plt.title('Temporal Width Ratio vs Propagation Distance')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
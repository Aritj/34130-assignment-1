import numpy as np
import matplotlib.pyplot as plt
from q1_2 import C_VALUES
from q1_5 import Z_VALUES

beta2 = -21.68  # Group Velocity Dispersion (GVD) parameter in ps^2/km
T_FWHM_initial = 10  # Initial Full Width at Half Maximum (FWHM) of the pulse in ps
T0 = T_FWHM_initial / (2 * np.sqrt(np.log(2)))  # Convert FWHM to T0 for Gaussian pulse (T0 = TFWHM / (2 * sqrt(ln(2)))
z_values = np.arange(0, 5.001, 0.001)  # Propagation distance in km from 0 to 5 km in steps of 0.001 km

def analytical_ratio(C, z):
    return np.sqrt(1 + (beta2 * C * z / T0**2)**2 + (beta2 * z / T0**2)**2)

def main(): 
    ratios = {C: analytical_ratio(C, z_values) for C in C_VALUES}
    
    plt.figure(figsize=(12, 6))
    for C, ratio in ratios.items():
        plt.plot(z_values, ratio, label=f'C = {C}')

    # ?????
    T_FWHM_ratios = {
        -10: np.array([3, 12, 22]),
        0: np.array([1.1, 1.5, 1.9]),
        5: np.array([0.4, 4.3, 9.5])
    }
        
    colors = { -10: 'red', 0: 'green', 5: 'blue' }
    for C, T_ratio in T_FWHM_ratios.items():
        plt.scatter(Z_VALUES, T_ratio, color=colors[C], s=100, edgecolor='black', label=f'Simulated C={C}')

    plt.title("Analytical Temporal Width Ratio vs. Distance with Scattered Points for Different Chirp Parameters")
    plt.xlabel("Propagation Distance (km)")
    plt.ylabel(r"$T_{FWHM}(z) / T_{FWHM}(0)$")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()
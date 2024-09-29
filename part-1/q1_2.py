import numpy as np
import matplotlib.pyplot as plt

from q1_1 import TW, N


# Assumptions
T_FWHM = 10  # Full Width Half Maximum in ps
C_VALUES = [-10, 0, +5]  # Chirp parameters
T0 = T_FWHM / (2 * np.sqrt(2 * np.log(2)))  # T0 in ps
A0 = 1  # Peak amplitude (W^1/2)

# Creating the time vector
TIME_VECTOR = np.linspace(-TW / 2, TW / 2, N)

# Function to generate the Gaussian field
def calculate_gaussian_field(A0, T0, C, t):
    first_term = (1 + 1j * C) / 2  # 1j is a complex number in Python
    second_term = (t / T0) ** 2
    return A0 * np.exp(-first_term * second_term)

# Main function to plot the curves in a single plot
def main() -> None:
    plt.figure(figsize=(10, 6))

    # Color and line style for each chirp value
    #colors = ['red', 'green', 'blue']
    line_styles = ['--', '-.', ':']

    # Plot each chirp value in a single plot
    for i, C in enumerate(C_VALUES):
        # Calculate the field envelope and power
        A_t = calculate_gaussian_field(A0, T0, C, TIME_VECTOR)
        P_t = np.abs(A_t) ** 2  # Power is |A(0,t)|^2

        # Plot with distinct color and line style
        plt.plot(TIME_VECTOR, P_t, linestyle=line_styles[i], label=f'Chirp C={C}')

    # Set labels and title
    plt.xlabel('Time (ps)')
    plt.ylabel('Power (W)')
    plt.title('Power P(0,t) vs Time for Different Chirp Values')
    plt.legend()  # Show legend to distinguish curves
    plt.grid(True)
    plt.xlim(-20, 20)  # Set x-axis limits for better visualization
    plt.show()

# Run the main function
if __name__ == '__main__':
    main()

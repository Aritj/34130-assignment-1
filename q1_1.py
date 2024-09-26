from typing import Tuple

# Assumptions
TW: float = 2500 # picoseconds
N: int = 2**14  # Number of samples

def calculate_sampling_and_frequency_params() -> Tuple[float, float, float, float]:
    """
    Calculate and return key parameters for sampling and frequency analysis.

    This function calculates:
    - The sampling period (`T_sa`) based on a given time window (`T_w`)
    - The sampling frequency (`F_sa`)
    - The frequency bin size (`ΔF`)
    - The minimum frequency (`F_min`) based on FFT conventions.

    Assumptions:
    - Time window (`T_w`) is 2500 picoseconds (ps), converted to seconds.
    - The number of samples (`N`) is set to 2^14 (16384).

    Returns:
        Tuple containing:
        - T_sa (float): Sampling period in seconds.
        - F_sa (float): Sampling frequency in Hz.
        - ΔF (float): Frequency bin size in Hz.
        - F_min (float): Minimum frequency in Hz.
    """
    T_sa: float = TW * 1e-12 / N  # Sampling period in seconds
    F_sa: float = 1 / T_sa  # Sampling frequency in Hz
    ΔF: float = F_sa / N  # Frequency bin in Hz
    F_min: float = -F_sa / 2  # Minimum frequency based on FFT conventions
    
    return T_sa, F_sa, ΔF, F_min

def main() -> None:
    T_sa, F_sa, ΔF, F_min = calculate_sampling_and_frequency_params()
    print(f'T_sa : {T_sa:>30} s')
    print(f'F_sa : {F_sa:>30} Hz')
    print(f'ΔF   : {ΔF:>30} Hz')
    print(f'F_min: {F_min:>30} Hz')


if __name__ == '__main__':
    main()
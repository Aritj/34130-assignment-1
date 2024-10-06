import numpy as np

# Assumptions
TW: float = 2500  # Time Window in picoseconds
N: int = 2**14  # Number of samples


def sampling_and_frequency_params() -> tuple[float, float, float, float]:
    T_sa: float = TW * 1e-12 / N  # Sampling period in seconds
    F_sa: float = 1 / T_sa  # Sampling frequency in Hz
    Delta_F: float = F_sa / N  # Frequency bin in Hz
    F_min: float = -F_sa / 2  # Minimum frequency based on FFT conventions

    return T_sa, F_sa, Delta_F, F_min


def generate_time_and_frequency_vectors(
    T_sa: float, Delta_F: float, F_min: float
) -> tuple[np.ndarray, np.ndarray]:
    # (e) Make a time vector based on the above time choices
    time_vector = np.linspace(-TW / 2 * 1e-12, TW / 2 * 1e-12, N)

    # (f) Make a frequency vector based on the above frequency choices
    frequency_vector = np.linspace(F_min, F_min + (N - 1) * Delta_F, N)

    return time_vector, frequency_vector


def main() -> None:
    # Calculate sampling and frequency parameters
    T_sa, F_sa, Delta_F, F_min = sampling_and_frequency_params()

    # Generate time and frequency vectors
    t, f = generate_time_and_frequency_vectors(T_sa, Delta_F, F_min)

    print(f"a) T_sa : {T_sa:>30} s")
    print(f"b) F_sa : {F_sa:>30} Hz ({F_sa/1e9} GHz)")
    print(f"c) ΔF   : {Delta_F:>30} Hz ({Delta_F/1e6} MHz)")
    print(f"d) F_min: {F_min:>30} Hz ({F_min/1e9} GHz)")
    print(f"d) F_max: {F_min:>30} Hz ({F_min/1e9} GHz)")

    print(f"e) Time Vector: {t}")
    print(f"f) Frequency Vector: {f}")


if __name__ == "__main__":
    main()

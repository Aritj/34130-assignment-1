# Assumptions
TW: int = 2500  # Time Window in picoseconds
N: int = 2**14  # Number of samples


def calculate_sampling_and_frequency_params(N: int, TW: int) -> tuple[float, float, float, float]:
    T_sa: float = TW * 1e-12 / N  # Sampling period in seconds
    F_sa: float = 1 / T_sa  # Sampling frequency in Hz
    Delta_F: float = F_sa / N  # Frequency bin in Hz
    F_min: float = -F_sa / 2  # Minimum frequency based on FFT conventions
    return T_sa, F_sa, Delta_F, F_min


def main() -> None:
    T_sa, F_sa, Delta_F, F_min = calculate_sampling_and_frequency_params(N, TW)
    print(f"T_sa : {T_sa:>30} s")
    print(f"F_sa : {F_sa:>30} Hz ({F_sa/1e9} GHz)")
    print(f"ΔF   : {Delta_F:>30} Hz ({Delta_F/1e6} MHz)")
    print(f"F_min: {F_min:>30} Hz ({F_min/1e9} GHz)")


if __name__ == "__main__":
    main()

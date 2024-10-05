# Assumptions
TW: int = 2500  # Time Window in picoseconds
N: int = 2**14  # Number of samples
T_FWHM = 10  # Full Width Half Maximum in ps
C_VALUES = [-10, 0, +5]  # Chirp parameters
A0 = 1  # Peak amplitude (W^1/2)


def sampling_and_frequency_params() -> tuple[float, float, float, float]:
    T_sa: float = TW * 1e-12 / N  # Sampling period in seconds
    F_sa: float = 1 / T_sa  # Sampling frequency in Hz
    Delta_F: float = F_sa / N  # Frequency bin in Hz
    F_min: float = -F_sa / 2  # Minimum frequency based on FFT conventions
    return T_sa, F_sa, Delta_F, F_min


def main() -> None:
    T_sa, F_sa, Delta_F, F_min = sampling_and_frequency_params()
    print(f"a) T_sa : {T_sa:>30} s ({T_sa*1e12} ps)")
    print(f"b) F_sa : {F_sa:>30} Hz ({F_sa*1e-9} GHz)")
    print(f"c) ΔF   : {Delta_F:>30} Hz ({Delta_F*1e-6} MHz)")
    print(f"d) F_min: {F_min:>30} Hz ({F_min*1e-9} GHz)")


if __name__ == "__main__":
    main()

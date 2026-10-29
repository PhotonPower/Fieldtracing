"""
Prüfrechnungen zum Bericht "Theoretische Grundlagen des Field Tracing".
1D-Modelle; Fourier-Konvention F{f}(k) = 1/(2 pi) * int f(x) exp(-i k x) dx.

Tests:
  1  SFT (Gl. sft_formula) gegen Brute-Force-Quadratur, harte und subpixelgewichtete Blende
  2  Gleichmäßige erfc-Kantenformel (Gl. fresnel_edge) gegen direkte Quadratur
  3  Exakte Phasenkompensation bei sphärischer Linsenphase (Gl. exact_legendre)
  4  Jacobi-Amplitudengesetz (Gl. jacobian_amplitude) gegen exaktes Winkelspektrum
"""

import numpy as np
from scipy.special import erfc


def test_sft():
    c, a = 1.0e5, 1.0  # Krümmung [mm^-2], Blendenhalbbreite [mm]
    U = lambda x: np.exp(-(x**2) / 0.8)
    k_test = np.array([0.0, 5e4, 0.99e5, 1.0e5, 1.01e5, 1.2e5])

    # Referenz: direkte Quadratur, sehr fein
    x = np.linspace(-a, a, 4_000_001)
    dx = x[1] - x[0]
    ref = np.array(
        [np.trapezoid(U(x) * np.exp(1j * (c * x**2 / 2 - k * x)), dx=dx) for k in k_test]
    ) / (2 * np.pi)

    def sft(N, L, subpixel):
        h = L / N
        xs = (np.arange(N) - N // 2) * h
        chi = np.clip((a - np.abs(xs)) / h + 0.5, 0, 1) if subpixel else (np.abs(xs) <= a)
        V = U(xs) * chi
        kap = 2 * np.pi * np.fft.fftfreq(N, d=h)
        G = (
            np.sqrt(2 * np.pi / abs(c))
            * np.exp(1j * np.pi / 4 * np.sign(c))
            * np.exp(-1j * kap**2 / (2 * c))
        )  # Transferfunktion der virtuellen Fresnel-Propagation
        W = np.fft.fftshift(np.fft.ifft(np.fft.fft(np.fft.ifftshift(V)) * G))
        Wi = np.interp(k_test / c, xs, W.real) + 1j * np.interp(k_test / c, xs, W.imag)
        return np.exp(-1j * k_test**2 / (2 * c)) / (2 * np.pi) * Wi

    print(
        f"Test 1 SFT: Brute-Force-Nyquist ~ {int(2 * c * a * a / np.pi)} Punkte, w_c = {np.sqrt(2 / c) * 1e3:.2f} um"
    )
    for N in (4096, 16384, 65536):
        for sp in (False, True):
            e = np.max(np.abs(sft(N, 4.0, sp) - ref)) / np.max(np.abs(ref))
            print(f"   N={N:6d}  {'subpixel' if sp else 'hart    '}  rel. Fehler = {e:.2e}")


def test_edge():
    c2, xe = 2000.0, 0.3
    x = np.linspace(-3, xe, 2_000_001)
    d = x[1] - x[0]
    print("Test 2 erfc-Kante:")
    for k in (c2 * 0.25, c2 * 0.29, c2 * 0.30, c2 * 0.31, c2 * 0.35):
        xs = k / c2
        exact = np.trapezoid(np.exp(1j * (c2 * x**2 / 2 - k * x)) * np.exp(-(x**2) / 4), dx=d) / (
            2 * np.pi
        )
        pft = (
            np.sqrt(2 * np.pi / c2)
            * np.exp(1j * np.pi / 4)
            * np.exp(-(xs**2) / 4)
            * np.exp(1j * (c2 * xs**2 / 2 - k * xs))
            / (2 * np.pi)
        )
        unif = pft * 0.5 * erfc(np.exp(-1j * np.pi / 4) * np.sqrt(c2 / 2) * (xs - xe))
        print(f"   x_s-x_e = {xs - xe:+.3f}: rel. Fehler = {abs(unif - exact) / abs(exact):.2e}")


def test_focus_phase():
    k0, f = 2 * np.pi / 0.5e-3, 25.0  # mm
    rho = np.linspace(0, 12.5, 50)
    r = np.sqrt(f**2 + rho**2)
    kp = k0 * rho / r
    psit = -k0 * (r - f) + kp * rho  # Legendre-Transformierte
    kz = np.sqrt(k0**2 - kp**2)
    print(
        f"Test 3 Phasenkompensation (sphärisch): max Rest = {np.max(np.abs(psit + kz * f - k0 * f)):.1e} rad"
    )
    kr = 0.5 * k0
    print(
        f"       parabolisch + exaktes kz, NA 0.5: Rest = {f * (np.sqrt(k0**2 - kr**2) - k0 + kr**2 / (2 * k0)):.0f} rad"
    )


def test_jacobian():
    k0, N, L = 2 * np.pi / 0.5e-3, 2**16, 4.0
    x = (np.arange(N) - N // 2) * (L / N)
    R0, d = 50.0, 30.0
    U1 = np.exp(-((x / 0.6) ** 8))
    E1 = U1 * np.exp(1j * k0 * (np.sqrt(R0**2 + x**2) - R0))
    kx = 2 * np.pi * np.fft.fftfreq(N, d=L / N)
    E2 = np.fft.ifft(np.fft.fft(E1) * np.exp(1j * np.sqrt(k0**2 - kx**2 + 0j) * d))
    x1 = np.array([0.0, 0.2, 0.4])
    x2 = x1 + d * x1 / R0  # Strahlabbildung
    J = 1 + d / R0  # 1D-Jacobi-Faktor
    print(
        "Test 4 Jacobi-Gesetz: numerisch",
        np.round(np.abs(E2)[np.searchsorted(x, x2)], 4),
        " Vorhersage",
        np.round(U1[np.searchsorted(x, x1)] / np.sqrt(J), 4),
    )


if __name__ == "__main__":
    test_sft()
    test_edge()
    test_focus_phase()
    test_jacobian()

"""
SVI volatility surface slice — Numerix "natural" parameterization.

Implements the variant described in:
    Konikov & Trainor, "SVI Volatility Surface", Numerix Support Papers, March 2024.

Two equivalent per-maturity representations of a single SVI slice:

  * SVIRaw      — the original Gatheral parameters (a, b, m, sigma, rho), eq. (2.1)
  * SVINatural  — the paper's natural parameters (w0, w1, w2, beta_minus, beta_plus),
                  eqs. (2.2)-(2.6), i.e. curve value/slope/curvature at k=0 and the
                  two asymptotic wing slopes.

The map is *exact* in both directions (eqs. 2.7-2.11), and is verified by round-trip
tests in __main__.

NOTE on naming. This "natural" parameterization is NOT Gatheral-Jacquier SVI-JW:
everything here lives in TOTAL-VARIANCE space and is maturity-free per slice.
  - w0 = w(0) is ATM *total variance*  (not ATM vol, not v_T)
  - w1 = w'(0) is the ATM *total-variance* skew (not the v_T-normalized JW skew)
  - w2 = w''(0) is the ATM total-variance curvature
  - beta_minus, beta_plus are the raw asymptotic slopes lim_{k->-+inf} w(k)/k
    (= b(rho-1) and b(rho+1)), the Lee-moment wing slopes.

Dependencies: numpy + stdlib only.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import copysign, erfc, pi, sqrt
from typing import Optional

import numpy as np

ArrayLike = np.ndarray


# --------------------------------------------------------------------------- #
# Forward / strike helpers
# --------------------------------------------------------------------------- #
# Every parameterization works internally in log-moneyness k = log(K / F), where
# F is the forward. Carrying F on a slice is what lets it speak in strikes K and
# report the risk-neutral density in strike space. F is optional: leave it None
# and the k-based API is unchanged; set it to use the *_strike helpers.
def forward_to_logm(F: Optional[float], K: ArrayLike) -> ArrayLike:
    """k = log(K / F)."""
    if F is None or F <= 0:
        raise ValueError("A positive forward F is required for strike <-> log-moneyness.")
    return np.log(np.asarray(K, dtype=float) / F)


def logm_to_strike(F: Optional[float], k: ArrayLike) -> ArrayLike:
    """K = F * exp(k)."""
    if F is None or F <= 0:
        raise ValueError("A positive forward F is required for strike <-> log-moneyness.")
    return F * np.exp(np.asarray(k, dtype=float))


# Vectorized standard-normal CDF via stdlib erf (keeps numpy + stdlib only; no scipy).
_norm_cdf = np.vectorize(lambda x: 0.5 * erfc(-x / sqrt(2.0)), otypes=[float])


def black_price(F: float, K: ArrayLike, w: ArrayLike, kind: str = "call") -> ArrayLike:
    """Undiscounted (forward) Black-76 option price.

    Parameterized by TOTAL variance w = sigma^2 * T rather than (sigma, T), so the
    maturity cancels — this is what the SVI slices hand back directly. With
        d1 = (log(F/K) + w/2) / sqrt(w),   d2 = d1 - sqrt(w),
        call = F N(d1) - K N(d2),   put = K N(-d2) - F N(-d1).
    For w <= 0 the price degenerates to intrinsic (forward) value.
    """
    F = float(F)
    K = np.asarray(K, dtype=float)
    w = np.asarray(w, dtype=float)
    sw = np.sqrt(np.maximum(w, 0.0))
    intrinsic = np.maximum(F - K, 0.0) if kind == "call" else np.maximum(K - F, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(F / K) + 0.5 * w) / sw
        d2 = d1 - sw
        if kind == "call":
            price = F * _norm_cdf(d1) - K * _norm_cdf(d2)
        elif kind == "put":
            price = K * _norm_cdf(-d2) - F * _norm_cdf(-d1)
        else:
            raise ValueError("kind must be 'call' or 'put'.")
    return np.where(sw > 0.0, price, intrinsic)


# --------------------------------------------------------------------------- #
# Raw (original) parameterization — eq. (2.1)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SVIRaw:
    """Original SVI parameters for one maturity. w is TOTAL implied variance.

        w(k) = a + b [ rho (k - m) + sqrt(sigma^2 + (k - m)^2) ]
    """

    a: float
    b: float
    m: float
    sigma: float
    rho: float
    T: Optional[float] = None  # maturity, only needed for vol conversion
    F: Optional[float] = None  # forward, only needed for strike-based methods

    # -- evaluation ---------------------------------------------------------- #
    def total_variance(self, k: ArrayLike) -> ArrayLike:
        k = np.asarray(k, dtype=float)
        d = k - self.m
        return self.a + self.b * (self.rho * d + np.sqrt(self.sigma**2 + d * d))

    def implied_vol(self, k: ArrayLike, T: Optional[float] = None) -> ArrayLike:
        T = self.T if T is None else T
        if T is None or T <= 0:
            raise ValueError("A positive maturity T is required for implied_vol.")
        return np.sqrt(self.total_variance(k) / T)

    def d_dk(self, k: ArrayLike) -> ArrayLike:
        k = np.asarray(k, dtype=float)
        d = k - self.m
        return self.b * (self.rho + d / np.sqrt(self.sigma**2 + d * d))

    def d2_dk2(self, k: ArrayLike) -> ArrayLike:
        k = np.asarray(k, dtype=float)
        d = k - self.m
        return self.b * self.sigma**2 / (self.sigma**2 + d * d) ** 1.5

    # -- global minimum (vertex), Section 4.3 -------------------------------- #
    def vertex(self) -> tuple[float, float]:
        """Returns (k*, w(k*)) — the minimum of the smile (root of dw/dk)."""
        r = self.rho
        k_star = self.m - self.sigma * r / sqrt(1.0 - r * r)
        w_star = self.a + self.b * self.sigma * sqrt(1.0 - r * r)
        return k_star, w_star

    # -- conversion to natural params, eqs. (2.2)-(2.6) ---------------------- #
    def to_natural(self) -> "SVINatural":
        a, b, m, s, r = self.a, self.b, self.m, self.sigma, self.rho
        root = sqrt(m * m + s * s)
        return SVINatural(
            w0=a + b * (root - r * m),
            w1=b * (r - m / root),
            w2=b * s * s / root**3,
            beta_minus=b * (r - 1.0),
            beta_plus=b * (r + 1.0),
            T=self.T,
            F=self.F,
        )

    # -- strike <-> log-moneyness, strike-space evaluation ------------------- #
    def log_moneyness(self, K: ArrayLike) -> ArrayLike:
        """k = log(K / F)."""
        return forward_to_logm(self.F, K)

    def strike(self, k: ArrayLike) -> ArrayLike:
        """K = F * exp(k)."""
        return logm_to_strike(self.F, k)

    def total_variance_strike(self, K: ArrayLike) -> ArrayLike:
        return self.total_variance(self.log_moneyness(K))

    def implied_vol_strike(self, K: ArrayLike, T: Optional[float] = None) -> ArrayLike:
        return self.implied_vol(self.log_moneyness(K), T)

    def risk_neutral_density_strike(self, K: ArrayLike) -> ArrayLike:
        """Density in strike space, q_K(K) = q_k(log(K/F)) / K (change of variables
        k = log(K/F), dk/dK = 1/K). Integrates to 1 over K for an arb-free slice."""
        K = np.asarray(K, dtype=float)
        return self.risk_neutral_density(self.log_moneyness(K)) / K

    # -- no-arbitrage diagnostics -------------------------------------------- #
    def positivity_conditions(self) -> dict[str, bool]:
        """Conditions (4.3a)-(4.3d) ensuring w(k) >= 0 everywhere."""
        _, w_star = self.vertex()
        return {
            "sigma>=0": self.sigma >= 0,
            "b>=0": self.b >= 0,
            "|rho|<=1": abs(self.rho) <= 1,
            "w(k*)>=0": w_star >= 0,
        }

    def durrleman_g(self, k: ArrayLike) -> ArrayLike:
        """Durrleman butterfly g-function. Butterfly-arbitrage-free iff g(k) >= 0
        for all k (and w > 0). This is the real test the paper leans on implicitly;
        it is the practitioner standard, and the one I'd actually gate a slice on."""
        k = np.asarray(k, dtype=float)
        w = self.total_variance(k)
        wp = self.d_dk(k)
        wpp = self.d2_dk2(k)
        return (1.0 - k * wp / (2.0 * w)) ** 2 - (wp * wp / 4.0) * (1.0 / w + 0.25) + wpp / 2.0

    def is_butterfly_free(self, k_grid: Optional[ArrayLike] = None) -> bool:
        if k_grid is None:
            k_grid = np.linspace(-3.0, 3.0, 2001)
        w = self.total_variance(k_grid)
        return bool(np.all(w > 0) and np.all(self.durrleman_g(k_grid) >= -1e-12))

    def risk_neutral_density(self, k: ArrayLike) -> ArrayLike:
        """Risk-neutral density in log-moneyness,
            q(k) = g(k) / sqrt(2 pi w) * exp(-d2^2 / 2),  d2 = -k/sqrt(w) - sqrt(w)/2,
        with g the Durrleman butterfly function. Integrates to 1 over k for an
        arbitrage-free slice (g >= 0, w > 0); negative where g < 0 flags butterfly
        arbitrage. Raw SVI is smooth, so this is exact everywhere (no kink)."""
        k = np.asarray(k, dtype=float)
        w = self.total_variance(k)
        g = self.durrleman_g(k)
        d2 = -k / np.sqrt(w) - np.sqrt(w) / 2.0
        return g / np.sqrt(2.0 * pi * w) * np.exp(-d2 * d2 / 2.0)

    # -- calibration: quadratic-form regression, Appendix C ------------------ #
    @classmethod
    def from_quotes(
        cls, k: ArrayLike, w: ArrayLike, T: Optional[float] = None, F: Optional[float] = None
    ) -> "SVIRaw":
        """Closed-form least-squares calibration (Appendix C).

        Linearizes (2.1) into  c^T x_i = w_i^2  with x_i = [1, k, w, k^2, k*w],
        solves the linear system for c, then recovers (a,b,m,sigma,rho).

        Caveat (paper's own): this minimizes residuals in w^2-space, not w-space,
        and is intended only as a *bootstrap* initial guess — not a final fit. Feed
        the result into a proper nonlinear objective (with the arbitrage penalties).
        """
        k = np.asarray(k, dtype=float)
        w = np.asarray(w, dtype=float)
        X = np.column_stack([np.ones_like(k), k, w, k * k, k * w])
        y = w * w
        c, *_ = np.linalg.lstsq(X, y, rcond=None)
        c0, c1, c2, c3, c4 = c

        b = 0.5 * sqrt(c4 * c4 + 4.0 * c3)
        rho = c4 / (2.0 * b)
        m = -(c1 + 0.5 * c2 * c4) / (2.0 * b * b)
        a = 0.5 * c2 + b * rho * m
        sigma = sqrt(c0 + a * a - 2.0 * b * rho * a * m - b * b * (1.0 - rho * rho) * m * m) / abs(b)
        return cls(a=a, b=b, m=m, sigma=sigma, rho=rho, T=T, F=F)


# --------------------------------------------------------------------------- #
# Natural parameterization — Section 3
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SVINatural:
    """SVI slice in natural parameters (Section 3).

        w0          ATM total variance          w(0)
        w1          ATM total-variance skew     w'(0)
        w2          ATM total-variance curvature w''(0)
        beta_minus  left  (put)  asymptotic slope  lim_{k->-inf} w/k = b(rho-1)
        beta_plus   right (call) asymptotic slope  lim_{k->+inf} w/k = b(rho+1)
    """

    w0: float
    w1: float
    w2: float
    beta_minus: float
    beta_plus: float
    T: Optional[float] = None
    F: Optional[float] = None

    # -- conversion to raw params, eqs. (2.7)-(2.11) ------------------------- #
    def to_raw(self) -> SVIRaw:
        bm, bp = self.beta_minus, self.beta_plus
        if bp == bm:
            raise ValueError("beta_plus == beta_minus implies b == 0 (degenerate slice).")

        b = 0.5 * (bp - bm)                       # (2.7)
        rho = (bp + bm) / (bp - bm)               # (2.8)

        # t := rho - w1/b == m / sqrt(sigma^2 + m^2) in (-1, 1); sign(m) = sign(t)
        t = rho - self.w1 / b
        one_m_t2 = 1.0 - t * t
        if one_m_t2 < 0.0:
            raise ValueError(
                f"Inconsistent natural params: (rho - w1/b)^2 = {t*t:.4f} > 1; "
                "no real raw SVI slice exists."
            )
        if self.w2 == 0.0:
            raise ValueError("w2 == 0 implies infinite sigma (flat curvature is degenerate).")

        sigma = sqrt(b * b * one_m_t2**3 / self.w2**2)          # (2.9)
        m_abs = sqrt(b * b * one_m_t2**2 * (t * t) / self.w2**2)  # (2.10)
        m = copysign(m_abs, t)                                   # sign(m) = sign(rho - w1/b)

        a = self.w0 - b * (-rho * m + sqrt(sigma * sigma + m * m))  # (2.11)
        return SVIRaw(a=a, b=b, m=m, sigma=sigma, rho=rho, T=self.T, F=self.F)

    # convenience pass-throughs
    def total_variance(self, k: ArrayLike) -> ArrayLike:
        return self.to_raw().total_variance(k)

    def implied_vol(self, k: ArrayLike, T: Optional[float] = None) -> ArrayLike:
        return self.to_raw().implied_vol(k, T)

    def durrleman_g(self, k: ArrayLike) -> ArrayLike:
        return self.to_raw().durrleman_g(k)

    def risk_neutral_density(self, k: ArrayLike) -> ArrayLike:
        return self.to_raw().risk_neutral_density(k)

    def log_moneyness(self, K: ArrayLike) -> ArrayLike:
        return forward_to_logm(self.F, K)

    def strike(self, k: ArrayLike) -> ArrayLike:
        return logm_to_strike(self.F, k)

    def total_variance_strike(self, K: ArrayLike) -> ArrayLike:
        return self.to_raw().total_variance_strike(K)

    def implied_vol_strike(self, K: ArrayLike, T: Optional[float] = None) -> ArrayLike:
        return self.to_raw().implied_vol_strike(K, T)

    def risk_neutral_density_strike(self, K: ArrayLike) -> ArrayLike:
        return self.to_raw().risk_neutral_density_strike(K)

    # -- strike-arbitrage equality constraints, eqs. (3.1)-(3.2) ------------- #
    def strike_arbitrage_residuals(self) -> dict[str, float]:
        """The paper's *equality* constraints that reduce the 5 DOF to 3:
            (3.1)  beta_plus = -beta_minus + w1
            (3.2)  w_tilde   = -4 w0 beta_plus beta_minus / (beta_plus - beta_minus)^2
        Returns how far the slice sits from each. These are a specific arbitrage-free
        *construction*, and (as the paper concedes in 4.3/5) they are often too tight
        to fit market data — in practice you'd relax them to penalties, or just gate
        on Durrleman g instead. Reported here for completeness, not as a hard filter.
        """
        bm, bp = self.beta_minus, self.beta_plus
        raw = self.to_raw()
        w_tilde = raw.a + raw.b * raw.sigma * sqrt(1.0 - raw.rho**2)  # root of dw/dk
        rhs_32 = -4.0 * self.w0 * bp * bm / (bp - bm) ** 2
        return {
            "res_3_1": bp - (-bm + self.w1),
            "res_3_2": w_tilde - rhs_32,
        }


# --------------------------------------------------------------------------- #
# Modified Jump-Wings — a rescaled view of the natural parameters
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SVIModifiedJumpWings:
    """A practitioner-facing reparametrization of SVINatural.

    The five quoting parameters are linear/affine rescalings of the natural ones
    (w0, w1, w2, beta_minus, beta_plus), with maturity carried explicitly:

        w0          = atm_vol**2 * T
        w1          = 2 * atm_skew * sqrt(w0) * 10
        w2          = atm_conv * 100
        beta_minus  = left_slope  * sqrt(w0)
        beta_plus   = right_slope * sqrt(w0)

    Unlike the natural params, atm_vol is an ANNUALIZED VOL and the wing/skew
    quantities are normalized by sqrt(w0) so they're roughly maturity-stable —
    this is what makes the family quotable across tenors.

    NOTE on the constant factors. The x10 on skew and x100 on convexity are
    quoting-unit conventions (e.g. skew per 10% move, convexity in bp-ish units),
    not anything dictated by SVI itself. They're taken verbatim from your spec; if
    they ever look off in a fit, this is the first place to check, since they
    cancel exactly on a round-trip and so won't show up in the assertions below.
    """

    atm_vol: float      # ATM implied (annualized) Black-Scholes vol
    atm_skew: float     # ATM skew, in the scaled units above
    atm_conv: float     # ATM convexity/curvature, scaled
    left_slope: float   # left (put) wing slope, normalized by sqrt(w0)
    right_slope: float  # right (call) wing slope, normalized by sqrt(w0)
    T: float
    F: Optional[float] = None  # forward, only needed for strike-based methods

    def __post_init__(self) -> None:
        if self.T <= 0:
            raise ValueError("T must be positive.")
        if self.atm_vol < 0:
            raise ValueError("atm_vol must be non-negative.")

    # -- natural quantities (forward map) ------------------------------------ #
    @property
    def w0(self) -> float:
        return self.atm_vol**2 * self.T

    @property
    def w1(self) -> float:
        return 2.0 * self.atm_skew * sqrt(self.w0) * 10.0

    @property
    def w2(self) -> float:
        return self.atm_conv * 100.0

    @property
    def beta_minus(self) -> float:
        return self.left_slope * sqrt(self.w0)

    @property
    def beta_plus(self) -> float:
        return self.right_slope * sqrt(self.w0)

    # -- conversions --------------------------------------------------------- #
    def as_natural(self) -> SVINatural:
        return SVINatural(
            w0=self.w0,
            w1=self.w1,
            w2=self.w2,
            beta_minus=self.beta_minus,
            beta_plus=self.beta_plus,
            T=self.T,
            F=self.F,
        )

    def to_raw(self) -> SVIRaw:
        return self.as_natural().to_raw()

    @classmethod
    def from_natural(
        cls, nat: SVINatural, T: Optional[float] = None, F: Optional[float] = None
    ) -> "SVIModifiedJumpWings":
        T = nat.T if T is None else T
        if T is None or T <= 0:
            raise ValueError("A positive maturity T is required (pass T or set nat.T).")
        if nat.w0 <= 0:
            raise ValueError("w0 must be positive to invert the modified-JW map.")
        sqrt_w0 = sqrt(nat.w0)
        return cls(
            atm_vol=sqrt(nat.w0 / T),
            atm_skew=nat.w1 / (2.0 * sqrt_w0 * 10.0),
            atm_conv=nat.w2 / 100.0,
            left_slope=nat.beta_minus / sqrt_w0,
            right_slope=nat.beta_plus / sqrt_w0,
            T=T,
            F=nat.F if F is None else F,
        )

    @classmethod
    def from_raw(
        cls, raw: SVIRaw, T: Optional[float] = None, F: Optional[float] = None
    ) -> "SVIModifiedJumpWings":
        return cls.from_natural(
            raw.to_natural(), T=T if T is not None else raw.T, F=F if F is not None else raw.F
        )

    # -- convenience evaluation (delegates to the raw slice) ----------------- #
    def total_variance(self, k: ArrayLike) -> ArrayLike:
        return self.to_raw().total_variance(k)

    def implied_vol(self, k: ArrayLike, T: Optional[float] = None) -> ArrayLike:
        return self.to_raw().implied_vol(k, self.T if T is None else T)

    def durrleman_g(self, k: ArrayLike) -> ArrayLike:
        return self.to_raw().durrleman_g(k)

    def risk_neutral_density(self, k: ArrayLike) -> ArrayLike:
        return self.to_raw().risk_neutral_density(k)

    def log_moneyness(self, K: ArrayLike) -> ArrayLike:
        return forward_to_logm(self.F, K)

    def strike(self, k: ArrayLike) -> ArrayLike:
        return logm_to_strike(self.F, k)

    def total_variance_strike(self, K: ArrayLike) -> ArrayLike:
        return self.to_raw().total_variance_strike(K)

    def implied_vol_strike(self, K: ArrayLike, T: Optional[float] = None) -> ArrayLike:
        return self.to_raw().implied_vol_strike(K, self.T if T is None else T)

    def risk_neutral_density_strike(self, K: ArrayLike) -> ArrayLike:
        return self.to_raw().risk_neutral_density_strike(K)


# --------------------------------------------------------------------------- #
# Variance-swap fair strike via option replication
# --------------------------------------------------------------------------- #
def variance_swap_strike(
    slice,
    *,
    k_min: float = -10.0,
    k_max: float = 10.0,
    num: int = 4001,
    F: Optional[float] = None,
    T: Optional[float] = None,
) -> float:
    """Fair variance-swap strike of a single smile slice, in vol points
    (e.g. 0.20 = 20 vol). Returns K_var = sqrt(annualized fair variance).

    Model-free Carr-Madan / Demeterfi-Derman-Kamani-Zou replication of the log
    contract, split at the forward K = F so the boundary correction term vanishes:

        K_var^2 * T = 2 [ int_{K<F} P(K)/K^2 dK + int_{K>F} C(K)/K^2 dK ]

    Computed in log-moneyness k = log(K/F) (K = F e^k, dK = K dk):

        K_var^2 * T = 2 int  price_OTM(k) / (F e^k) dk,

    with price_OTM = put for k < 0, call for k >= 0 (continuous at the forward).
    Option prices are Black-76 values of the slice's own implied vols, so the
    result reflects the actual smile -- including any kink (e.g. IVP), which a
    smooth-density estimate would miss.

    Works on any slice exposing `total_variance(k)`, `.F` and `.T` (all five
    SVI/IVP classes do); the IVP classes return their realized/damped smile.

    NOTE: the fair variance is wing-sensitive. The default grid spans roughly
    e^{+-10} in strike; widen [k_min, k_max] or raise `num` for very steep wings.
    """
    F = slice.F if F is None else F
    if F is None or F <= 0:
        raise ValueError("A positive forward F is required (set slice.F or pass F).")
    T = slice.T if T is None else T
    if T is None or T <= 0:
        raise ValueError("A positive maturity T is required (set slice.T or pass T).")

    k = np.linspace(k_min, k_max, num)
    w = np.asarray(slice.total_variance(k), dtype=float)
    K = F * np.exp(k)
    otm = np.where(k < 0.0, black_price(F, K, w, "put"), black_price(F, K, w, "call"))
    integral = np.trapezoid(otm / K, k)  # = int price/K^2 dK
    return float(sqrt(2.0 * integral / T))


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)

    # SX5E-representative ~2Y slice (total variance), arb-free by construction.
    slice_raw = SVIRaw(a=0.018, b=0.110, m=0.085, sigma=0.190, rho=-0.62, T=2.0)
    print("Raw slice:        ", slice_raw)

    # 1) Round-trip raw -> natural -> raw -------------------------------------
    nat = slice_raw.to_natural()
    back = nat.to_raw()
    err = max(
        abs(slice_raw.a - back.a),
        abs(slice_raw.b - back.b),
        abs(slice_raw.m - back.m),
        abs(slice_raw.sigma - back.sigma),
        abs(slice_raw.rho - back.rho),
    )
    print("Natural params:   ", nat)
    print(f"raw->natural->raw max abs err: {err:.3e}")
    assert err < 1e-12, "round-trip failed"

    # 2) Round-trip on the curve itself ---------------------------------------
    ks = np.linspace(-1.0, 1.0, 21)
    curve_err = np.max(np.abs(slice_raw.total_variance(ks) - back.total_variance(ks)))
    print(f"total-variance curve max abs err: {curve_err:.3e}")

    # 3) Natural params match their definitions at k=0 ------------------------
    h = 1e-5
    w0_fd = slice_raw.total_variance(0.0)
    w1_fd = (slice_raw.total_variance(h) - slice_raw.total_variance(-h)) / (2 * h)
    w2_fd = (slice_raw.total_variance(h) - 2 * w0_fd + slice_raw.total_variance(-h)) / h**2
    print(f"w0  exact/FD: {nat.w0:.8f} / {w0_fd:.8f}")
    print(f"w1  exact/FD: {nat.w1:.8f} / {w1_fd:.8f}")
    print(f"w2  exact/FD: {nat.w2:.8f} / {w2_fd:.8f}")

    # 4) No-arbitrage diagnostics ---------------------------------------------
    print("positivity:       ", slice_raw.positivity_conditions())
    print("butterfly-free:   ", slice_raw.is_butterfly_free())
    print("min Durrleman g:  ", float(np.min(slice_raw.durrleman_g(np.linspace(-3, 3, 4001)))))
    print("strike-arb (eq) residuals:", nat.strike_arbitrage_residuals())

    # 5) Quadratic-form calibration recovers params from noiseless quotes -----
    k_q = np.linspace(-0.8, 0.8, 25)
    w_q = slice_raw.total_variance(k_q)
    fit = SVIRaw.from_quotes(k_q, w_q, T=2.0)
    fit_err = np.max(np.abs(fit.total_variance(k_q) - w_q))
    print("calibrated:       ", fit)
    print(f"quadratic-regression fit max abs err (noiseless): {fit_err:.3e}")

    # vertex
    print("vertex (k*, w*):  ", slice_raw.vertex())

    # 6) SVIModifiedJumpWings round-trip --------------------------------------
    mjw = SVIModifiedJumpWings.from_natural(nat)  # T taken from nat.T
    nat2 = mjw.as_natural()
    mjw_err = max(
        abs(nat.w0 - nat2.w0),
        abs(nat.w1 - nat2.w1),
        abs(nat.w2 - nat2.w2),
        abs(nat.beta_minus - nat2.beta_minus),
        abs(nat.beta_plus - nat2.beta_plus),
    )
    print("Modified-JW:      ", mjw)
    print(f"natural->mod-JW->natural max abs err: {mjw_err:.3e}")
    assert mjw_err < 1e-12, "modified-JW round-trip failed"

    # full chain raw -> natural -> mod-JW -> raw must match on the curve
    raw_chain = mjw.to_raw()
    chain_err = np.max(np.abs(slice_raw.total_variance(ks) - raw_chain.total_variance(ks)))
    print(f"raw->natural->mod-JW->raw curve max abs err: {chain_err:.3e}")
    assert chain_err < 1e-12, "modified-JW chain failed"

    # 7) Forward F: strike-based evaluation + density in strike space ---------
    F = 4200.0
    slice_F = SVIRaw(a=0.018, b=0.110, m=0.085, sigma=0.190, rho=-0.62, T=2.0, F=F)
    # F is carried through every conversion
    assert slice_F.to_natural().to_raw().F == F
    assert SVIModifiedJumpWings.from_raw(slice_F).F == F
    # strike <-> log-moneyness round-trip
    K = np.array([3000.0, 4200.0, 5400.0])
    km = slice_F.log_moneyness(K)
    print("\nlog-moneyness of strikes:", km)
    assert np.max(np.abs(slice_F.strike(km) - K)) < 1e-9
    # strike-space evaluation matches k-space evaluation
    assert np.max(np.abs(slice_F.total_variance_strike(K) - slice_F.total_variance(km))) < 1e-12
    print("ATM (K=F) implied vol:   ", float(slice_F.implied_vol_strike(F)))
    # density in strike space integrates to 1 over K
    Kgrid = np.linspace(500.0, 30000.0, 400001)
    integral_K = np.trapezoid(slice_F.risk_neutral_density_strike(Kgrid), Kgrid)
    print(f"strike-space density integral over K: {integral_K:.4f}")
    assert abs(integral_K - 1.0) < 1e-3, "strike-space density must integrate to 1"

    # 8) Variance-swap fair strike (option replication) ----------------------
    # (a) flat smile (b=0) must replicate vol exactly: K_var == sigma
    sigma_flat, T_flat = 0.20, 1.5
    flat = SVIRaw(a=sigma_flat**2 * T_flat, b=0.0, m=0.0, sigma=0.0, rho=0.0, T=T_flat, F=100.0)
    kvar_flat = variance_swap_strike(flat)
    print(f"\nflat-vol slice: K_var={kvar_flat:.6f}  (sigma={sigma_flat})")
    assert abs(kvar_flat - sigma_flat) < 1e-4, "flat-vol variance swap must equal sigma"

    # (b) replication must match the density route -2 E[k]/T on the smooth slice
    kvar = variance_swap_strike(slice_F)
    kk = np.linspace(-12.0, 12.0, 200001)
    Ek = np.trapezoid(kk * slice_F.risk_neutral_density(kk), kk)
    kvar_density = sqrt(-2.0 * Ek / slice_F.T)
    print(f"SX5E slice: K_var (replication)={kvar:.6f}  (density -2E[k]/T)={kvar_density:.6f}")
    assert abs(kvar - kvar_density) < 1e-3, "replication vs density route mismatch"

    # (c) convexity premium: variance-swap strike >= ATM vol for a skewed smile
    atm_vol = float(slice_F.implied_vol_strike(slice_F.F))
    print(f"K_var={kvar:.6f}  ATM vol={atm_vol:.6f}  (premium={kvar - atm_vol:+.6f})")
    assert kvar >= atm_vol, "variance-swap strike should exceed ATM vol under skew/convexity"

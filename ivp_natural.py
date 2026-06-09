"""
IVPNatural — natural (ATM-jet) parameterization for the IVP wing transform.

The IVP smile (Mahdavi Damghani 2015, eq. 26) damps the SVI wings with

    z(k)    = k / beta^(1 + 4|k - m|)
    zeta(k) = z(k) - m
    w(k)    = a + b [ rho * zeta + sqrt(zeta^2 + sigma^2) ]          (skeleton: a,b,m,sigma,rho)

beta = 1 recovers raw SVI exactly.  This module parameterizes a slice by the
*realized* smile's value/slope/curvature at k = 0 plus the skeleton wing slopes
and the damping beta -- the natural analogue of SVINatural for the IVP variant.

Why this is the clean parameterization for IVP
-----------------------------------------------
The transform's kink is at k = m, so for m != 0 the point k = 0 is in a smooth
region and (w0, w1, w2) = (w(0), w'(0), w''(0)) are ordinary derivatives.  Two
facts make the inversion reduce to SVINatural:

  * z(0) = 0  (numerator is k), so  w(0)_IVP = w(0)_raw  exactly -- beta-invariant.
  * the realized slope/curvature are the raw skeleton jet, rescaled:
        w1 = beta^(-1-4|m|) * W1_raw
        w2 = 8 ln(beta) sign(m) beta^(-1-4|m|) * W1_raw  +  beta^(-2-8|m|) * W2_raw

Inverting those two relations recovers the raw jet targets, which we hand to
SVINatural.  The rescaling factors depend on |m|, so to_raw() does a small 1-D
fixed-point in m (exact fixed point at the truth; trivial when beta = 1).

Caveats baked in below:
  * m = 0 (vertex at the money) is rejected: the kink lands on k = 0 and the
    jet becomes one-sided.
  * the realized smile has a SLOPE kink at k = m for m != 0, so its risk-neutral
    density carries a point mass there; the Durrleman g / density helpers use the
    smooth formula and are valid away from k = m only.

Reuses SVIRaw / SVINatural from your module -- adjust the import to its filename.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import copysign, log, sqrt
from typing import Optional

import numpy as np

from svi import (  # <-- adjust to your module name
    SVINatural,
    SVIRaw,
    forward_to_logm,
    logm_to_strike,
    variance_swap_strike,
)

ArrayLike = np.ndarray


# --------------------------------------------------------------------------- #
# IVP transform derivatives (chain rule) for a given raw skeleton
# --------------------------------------------------------------------------- #
def _ivp_w_derivs(raw: SVIRaw, beta: float, k: ArrayLike):
    """Return (w, w', w'') of the realized IVP smile at log-moneyness k.

        zeta(k)  = k * beta^(-1-4|x|) - m,           x = k - m
        zeta'(k) = beta^(-1-4|x|) (1 - 4 ln(beta) k sign(x))         # jumps at k=m
        zeta''(k)= -4 ln(beta) sign(x) beta^(-1-4|x|) (2 - 4 ln(beta) k sign(x))
        w'  = zeta' W'(zeta),   w'' = zeta'' W'(zeta) + zeta'^2 W''(zeta)
    with W'(zeta) = b(rho + zeta/r), W''(zeta) = b sigma^2 / r^3, r = sqrt(zeta^2+sigma^2).

    np.sign(0) = 0, so at exactly k = m this returns the average of the two
    one-sided branches; the density genuinely jumps there.
    """
    a, b, m, sig, rho = raw.a, raw.b, raw.m, raw.sigma, raw.rho
    k = np.asarray(k, dtype=float)
    x = k - m
    ax = np.abs(x)
    s = np.sign(x)
    if beta == 1.0:
        zeta = x
        zp = np.ones_like(x)
        zpp = np.zeros_like(x)
    else:
        L = log(beta)
        D = beta ** (-1.0 - 4.0 * ax)
        zeta = k * D - m
        zp = D * (1.0 - 4.0 * L * k * s)
        zpp = -4.0 * L * s * D * (2.0 - 4.0 * L * k * s)
    r = np.sqrt(zeta * zeta + sig * sig)
    w = a + b * (rho * zeta + r)
    Wp = b * (rho + zeta / r)
    Wpp = b * sig * sig / r**3
    return w, zp * Wp, zpp * Wp + zp * zp * Wpp


# --------------------------------------------------------------------------- #
# IVPNatural
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class IVPNatural:
    """IVP slice in natural parameters.

        w0          realized ATM total variance        w(0)
        w1          realized ATM total-variance skew    w'(0)
        w2          realized ATM total-variance curv.   w''(0)
        beta_minus  skeleton left  asymptotic slope     b(rho - 1)
        beta_plus   skeleton right asymptotic slope     b(rho + 1)
        beta_g      IVP wing damping (>= 1; 1 -> raw SVI / plain SVINatural)
    """

    w0: float
    w1: float
    w2: float
    beta_minus: float
    beta_plus: float
    beta_g: float = 1.0
    T: Optional[float] = None
    F: Optional[float] = None  # forward, only needed for strike-based methods

    _MAXIT = 200
    _TOL = 1e-14

    def __post_init__(self) -> None:
        if self.beta_g < 1.0:
            raise ValueError("beta_g must be >= 1 (beta_g = 1 is plain SVINatural)")

    # -- core: recover the raw skeleton ------------------------------------- #
    def to_raw(self) -> SVIRaw:
        bm, bp = self.beta_minus, self.beta_plus
        if bp == bm:
            raise ValueError("beta_plus == beta_minus implies b == 0 (degenerate slice).")
        b = 0.5 * (bp - bm)
        rho = (bp + bm) / (bp - bm)
        beta = self.beta_g

        # beta = 1: the realized jet IS the raw jet -> plain SVINatural.
        if beta == 1.0:
            return SVINatural(self.w0, self.w1, self.w2, bm, bp, self.T, self.F).to_raw()

        L = log(beta)

        def m_implied(m: float):
            """Map a trial m to the m that SVINatural would return for the
            raw-jet targets implied by that m. Fixed point == solution."""
            am = abs(m)
            sm = copysign(1.0, m)
            W1 = self.w1 * beta ** (1.0 + 4.0 * am)                       # raw slope target
            W2 = (self.w2 - 8.0 * L * sm * self.w1) * beta ** (2.0 + 8.0 * am)  # raw curv target
            t = rho - W1 / b
            if abs(t) >= 1.0 or W2 == 0.0:
                return None
            m_abs = sqrt(b * b * (1.0 - t * t) ** 2 * (t * t) / W2**2)
            return copysign(m_abs, t)

        # initial guess: treat the realized jet as if it were raw (beta=1 solve)
        m = SVINatural(self.w0, self.w1, self.w2, bm, bp).to_raw().m
        if m == 0.0:
            raise ValueError("m = 0 (vertex at the money) has no two-sided IVP jet.")

        for _ in range(self._MAXIT):
            mi = m_implied(m)
            if mi is None:
                raise ValueError("infeasible IVPNatural params: no real skeleton during solve.")
            if abs(mi - m) < self._TOL:
                m = mi
                break
            m = 0.5 * m + 0.5 * mi                       # damped fixed point
        else:
            raise ValueError("IVPNatural solve did not converge.")

        if m == 0.0:
            raise ValueError("m = 0 (vertex at the money) has no two-sided IVP jet.")

        # rebuild the skeleton from the converged raw-jet targets
        am, sm = abs(m), copysign(1.0, m)
        W1 = self.w1 * beta ** (1.0 + 4.0 * am)
        W2 = (self.w2 - 8.0 * L * sm * self.w1) * beta ** (2.0 + 8.0 * am)
        return SVINatural(self.w0, W1, W2, bm, bp, self.T, self.F).to_raw()

    # -- realized-smile evaluation (IVP transform) -------------------------- #
    def total_variance(self, k: ArrayLike) -> ArrayLike:
        return _ivp_w_derivs(self.to_raw(), self.beta_g, k)[0]

    def d_dk(self, k: ArrayLike) -> ArrayLike:
        return _ivp_w_derivs(self.to_raw(), self.beta_g, k)[1]

    def d2_dk2(self, k: ArrayLike) -> ArrayLike:
        return _ivp_w_derivs(self.to_raw(), self.beta_g, k)[2]

    def implied_vol(self, k: ArrayLike, T: Optional[float] = None) -> ArrayLike:
        T = self.T if T is None else T
        if T is None or T <= 0:
            raise ValueError("A positive maturity T is required for implied_vol.")
        return np.sqrt(self.total_variance(k) / T)

    def realized_jet(self) -> tuple[float, float, float]:
        """(w(0), w'(0), w''(0)) of the realized smile -- should reproduce inputs."""
        w, wp, wpp = _ivp_w_derivs(self.to_raw(), self.beta_g, np.array([0.0]))
        return float(w.item()), float(wp.item()), float(wpp.item())

    # -- no-arbitrage diagnostics (smooth-formula; invalid AT k = m) -------- #
    def durrleman_g(self, k: ArrayLike) -> ArrayLike:
        """Durrleman butterfly g-function of the realized smile. Valid away from
        the vertex k = m, where the slope kink injects a density point mass."""
        k = np.asarray(k, dtype=float)
        w, wp, wpp = _ivp_w_derivs(self.to_raw(), self.beta_g, k)
        return (1.0 - k * wp / (2.0 * w)) ** 2 - (wp * wp / 4.0) * (1.0 / w + 0.25) + wpp / 2.0

    def is_butterfly_free(self, k_grid: Optional[ArrayLike] = None) -> bool:
        raw = self.to_raw()
        if k_grid is None:
            # avoid landing exactly on the kink k = m
            k_grid = np.linspace(raw.m - 3.0, raw.m + 3.0, 2001) + 1e-7
        w = _ivp_w_derivs(raw, self.beta_g, k_grid)[0]
        return bool(np.all(w > 0) and np.all(self.durrleman_g(k_grid) >= -1e-12))

    def risk_neutral_density(self, k: ArrayLike):
        """q(k) = g(k)/sqrt(2 pi w) e^{-d2^2/2}, d2 = -k/sqrt(w) - sqrt(w)/2.
        Smooth-formula density; carries a point mass at k = m not captured here."""
        import math
        k = np.asarray(k, dtype=float)
        w, wp, wpp = _ivp_w_derivs(self.to_raw(), self.beta_g, k)
        g = (1.0 - k * wp / (2.0 * w)) ** 2 - (wp * wp / 4.0) * (1.0 / w + 0.25) + wpp / 2.0
        d2 = -k / np.sqrt(w) - np.sqrt(w) / 2.0
        return g / np.sqrt(2.0 * math.pi * w) * np.exp(-d2**2 / 2.0)

    # -- strike <-> log-moneyness, strike-space evaluation ------------------- #
    # These route through the realized (damped) IVP smile, not the skeleton.
    def log_moneyness(self, K: ArrayLike) -> ArrayLike:
        return forward_to_logm(self.F, K)

    def strike(self, k: ArrayLike) -> ArrayLike:
        return logm_to_strike(self.F, k)

    def total_variance_strike(self, K: ArrayLike) -> ArrayLike:
        return self.total_variance(self.log_moneyness(K))

    def implied_vol_strike(self, K: ArrayLike, T: Optional[float] = None) -> ArrayLike:
        return self.implied_vol(self.log_moneyness(K), T)

    def risk_neutral_density_strike(self, K: ArrayLike):
        """Density in strike space, q_K(K) = q_k(log(K/F)) / K. Same kink caveat as
        risk_neutral_density: the point mass at the vertex k = m is not captured."""
        K = np.asarray(K, dtype=float)
        return self.risk_neutral_density(self.log_moneyness(K)) / K

    # -- build from a known skeleton (handy for tests / seeding) ------------ #
    @classmethod
    def from_raw(cls, raw: SVIRaw, beta_g: float, T: Optional[float] = None) -> "IVPNatural":
        w, wp, wpp = _ivp_w_derivs(raw, beta_g, np.array([0.0]))
        b, rho = raw.b, raw.rho
        return cls(
            w0=float(w.item()), w1=float(wp.item()), w2=float(wpp.item()),
            beta_minus=b * (rho - 1.0), beta_plus=b * (rho + 1.0),
            beta_g=beta_g, T=raw.T if T is None else T, F=raw.F,
        )


# --------------------------------------------------------------------------- #
# Modified Jump-Wings quoting wrapper, IVP variant
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class IVPModifiedJumpWings:
    """Practitioner quoting wrapper around IVPNatural — the same affine rescaling
    as SVIModifiedJumpWings, plus the IVP wing damping beta_g.

    The five quoting parameters describe the REALIZED (damped) IVP smile at k = 0:

        w0          = atm_vol**2 * T              realized ATM total variance
        w1          = 2 * atm_skew * sqrt(w0) * 10 realized ATM skew   (scaled)
        w2          = atm_conv * 100               realized ATM convexity (scaled)
        beta_minus  = left_slope  * sqrt(w0)       skeleton left  wing slope
        beta_plus   = right_slope * sqrt(w0)       skeleton right wing slope

    beta_g = 1 reduces this to SVIModifiedJumpWings exactly.

    Two things differ from the SVI version:
      * atm_vol / atm_skew / atm_conv pin the *realized* (post-damping) smile at
        the money -- that's the whole reason for routing through IVPNatural.
        left_slope / right_slope remain SKELETON asymptotic wing slopes, since
        the IVP damping drives the true realized asymptotic slope to 0.
      * evaluation goes through as_natural() (the IVP transform), so
        total_variance / implied_vol are the realized damped smile -- NOT
        to_raw().total_variance(), which would give the undamped skeleton.

    The x10 / x100 constants are the same quoting-unit conventions as the SVI
    wrapper and cancel on a round-trip.
    """

    atm_vol: float
    atm_skew: float
    atm_conv: float
    left_slope: float
    right_slope: float
    T: float
    beta_g: float = 1.0
    F: Optional[float] = None  # forward, only needed for strike-based methods

    def __post_init__(self) -> None:
        if self.T <= 0:
            raise ValueError("T must be positive.")
        if self.atm_vol < 0:
            raise ValueError("atm_vol must be non-negative.")
        if self.beta_g < 1.0:
            raise ValueError("beta_g must be >= 1 (beta_g = 1 is SVIModifiedJumpWings).")

    # -- natural quantities (forward map) ----------------------------------- #
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

    # -- conversions -------------------------------------------------------- #
    def as_natural(self) -> IVPNatural:
        return IVPNatural(
            w0=self.w0, w1=self.w1, w2=self.w2,
            beta_minus=self.beta_minus, beta_plus=self.beta_plus,
            beta_g=self.beta_g, T=self.T, F=self.F,
        )

    def to_raw(self) -> SVIRaw:
        """The raw SKELETON (undamped). Use total_variance() for the damped smile."""
        return self.as_natural().to_raw()

    @classmethod
    def from_natural(
        cls, nat: IVPNatural, T: Optional[float] = None, F: Optional[float] = None
    ) -> "IVPModifiedJumpWings":
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
            beta_g=nat.beta_g,
            F=nat.F if F is None else F,
        )

    @classmethod
    def from_raw(cls, raw: SVIRaw, beta_g: float, T: Optional[float] = None) -> "IVPModifiedJumpWings":
        return cls.from_natural(
            IVPNatural.from_raw(raw, beta_g, T=T if T is not None else raw.T)
        )

    # -- evaluation = realized (damped) smile, via the IVP transform -------- #
    def total_variance(self, k: ArrayLike) -> ArrayLike:
        return self.as_natural().total_variance(k)

    def implied_vol(self, k: ArrayLike, T: Optional[float] = None) -> ArrayLike:
        return self.as_natural().implied_vol(k, self.T if T is None else T)

    def d_dk(self, k: ArrayLike) -> ArrayLike:
        return self.as_natural().d_dk(k)

    def d2_dk2(self, k: ArrayLike) -> ArrayLike:
        return self.as_natural().d2_dk2(k)

    def durrleman_g(self, k: ArrayLike) -> ArrayLike:
        return self.as_natural().durrleman_g(k)

    def risk_neutral_density(self, k: ArrayLike):
        return self.as_natural().risk_neutral_density(k)

    def log_moneyness(self, K: ArrayLike) -> ArrayLike:
        return forward_to_logm(self.F, K)

    def strike(self, k: ArrayLike) -> ArrayLike:
        return logm_to_strike(self.F, k)

    def total_variance_strike(self, K: ArrayLike) -> ArrayLike:
        return self.as_natural().total_variance_strike(K)

    def implied_vol_strike(self, K: ArrayLike, T: Optional[float] = None) -> ArrayLike:
        return self.as_natural().implied_vol_strike(K, self.T if T is None else T)

    def risk_neutral_density_strike(self, K: ArrayLike):
        return self.as_natural().risk_neutral_density_strike(K)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)

    skel = SVIRaw(a=0.04, b=0.40, m=0.05, sigma=0.10, rho=-0.40, T=1.0)

    for beta_g in (1.0, 1.3, 1.4):
        ivp = IVPNatural.from_raw(skel, beta_g=beta_g, T=1.0)
        back = ivp.to_raw()
        err = max(abs(getattr(skel, f) - getattr(back, f))
                  for f in ("a", "b", "m", "sigma", "rho"))
        w0, w1, w2 = ivp.realized_jet()
        print(f"beta_g={beta_g}")
        print(f"  natural: w0={ivp.w0:.6f} w1={ivp.w1:.6f} w2={ivp.w2:.6f} "
              f"b-={ivp.beta_minus:.3f} b+={ivp.beta_plus:.3f}")
        print(f"  skeleton round-trip max abs err: {err:.3e}")
        print(f"  realized jet reproduces inputs:  "
              f"{abs(w0-ivp.w0):.2e}, {abs(w1-ivp.w1):.2e}, {abs(w2-ivp.w2):.2e}")
        assert err < 1e-9, "skeleton round-trip failed"

    # density sanity + butterfly check for the damped slice
    ivp = IVPNatural.from_raw(skel, beta_g=1.3, T=1.0)
    k = np.linspace(-4, 4, 8001) + 1e-7
    integral = np.trapezoid(ivp.risk_neutral_density(k), k)
    print(f"\nbeta_g=1.3 density integral: {integral:.4f}")
    print(f"butterfly-free (away from vertex): {ivp.is_butterfly_free()}")

    # forward F: strike-based eval on the realized (damped) IVP smile ---------
    F = 100.0
    skel_F = SVIRaw(a=0.04, b=0.40, m=0.05, sigma=0.10, rho=-0.40, T=1.0, F=F)
    ivp_F = IVPNatural.from_raw(skel_F, beta_g=1.3, T=1.0)
    assert ivp_F.F == F and ivp_F.to_raw().F == F  # F carried through the solve
    assert IVPModifiedJumpWings.from_natural(ivp_F).F == F
    K = np.array([80.0, 100.0, 125.0])
    km = ivp_F.log_moneyness(K)
    assert np.max(np.abs(ivp_F.strike(km) - K)) < 1e-9
    # strike-space eval = realized (damped) smile, NOT the skeleton
    assert np.max(np.abs(ivp_F.total_variance_strike(K) - ivp_F.total_variance(km))) < 1e-12
    print(f"\nF={F}: ATM (K=F) implied vol: {float(ivp_F.implied_vol_strike(F)):.6f}")
    Kgrid = np.linspace(1.0, 1000.0, 400001)
    integral_K = np.trapezoid(ivp_F.risk_neutral_density_strike(Kgrid), Kgrid)
    print(f"F={F}: strike-space density integral over K: {integral_K:.4f}")

    # variance-swap fair strike on the damped IVP slice (replication prices the
    # realized smile, so the kink is captured exactly -- no point-mass error)
    kvar = variance_swap_strike(ivp_F)
    kvar_fine = variance_swap_strike(ivp_F, num=8001)
    print(f"F={F}: variance-swap strike K_var={kvar:.6f}  (grid-stable: {abs(kvar - kvar_fine):.2e})")
    assert kvar > 0.0 and abs(kvar - kvar_fine) < 1e-3, "IVP variance swap unstable"

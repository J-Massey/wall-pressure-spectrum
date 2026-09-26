# Model summary

Notation: `delta+` = Re_tau, `f+ = f nu/u_tau²`, `T+ = 1/f+`, `f^o = f delta/U_e`,
`T^o = 1/f^o`. The predicted quantity is the pre-multiplied spectrum
`f φ_pp / τ_w²` (= `f+ φ_pp+`), a function of frequency only (one-sided, 1-D).
`U_e` is the free-stream (BL) or centreline (pipe, channel) velocity, so
`f^o = f+ · Re_tau · (u_tau/U_e)`.

The spectrum is the linear sum of an inner-scaled and an outer-scaled component,
evaluated at the same physical frequency.

## Model A (log-normal), `wallpressure.model_a`

`f φ_pp+ = r_v (g1 + g2)`, `g_i = A_i exp(-s_i (log10 T − log10 T̄_i)²)`,
with the viscous step `r_v = exp(r1 T+) / (exp(r1 r2) + exp(r1 T+))`.
The outer centre is `T̄_2+ = T̄^o · Re_tau · u_tau/U_e`.

| | A1 | s1 | T̄1+ | A2 | s2 | T̄^o | r1 | r2 |
|---|---|---|---|---|---|---|---|---|
| BL | 2.20 | 3.9 | 20 | 1.40(log10 δ+ − 2.2) | 1.2 | 0.82 | 0.5 | 7 |
| Pipe | 2.90(1 − 1000/δ+) | 4.3 | 20 | 0.91(log10 δ+ − 2.2) | 1.0 | 0.18 | 0.5 | 7 |
| Channel | 2.10(1 − 100/δ+) | 4.4 | 12 | 0.90(log10 δ+ − 2.2) | 1.0 | 0.60 | 0.5 | 3 |

> **Note on σ.** The code coefficients `s_i` above are what generated the paper's figures.
> Table 1 of the paper lists σ1 equal to `s1` but σ2 = 8.18 / 10 / 10, which correspond to
> `s2 = 1/log10(σ2)²` = 1.2 / 1.0 / 1.0. The two conventions differ between the columns in the
> printed table; this package follows the code (the implementation is regression-tested against it).
> A BL with unknown `u_tau/U_e` uses `Cf/2 ≈ (ln δ+/0.384 + 4.127)^-2`.

## Model B (modified Lorentzian), `wallpressure.model_b`

`g = A 2^r (f/f_b)^{p_low} (1 + (f/f_b)^q)^{-r}`, `r = (p_low − p_high)/q`.

* Inner: A = 1.6, f_b+ = 0.1 (T_b+ = 10), p_low = 1, p_high = −6, q = 3.5 (Re_tau-invariant).
* Outer: f_b^o = 2/3, p_low = 3, with
  `A = 1/(1+exp(a_A(b_A − log10 δ+)))`,
  `p_high = −2 + 1.5/(1+exp(a_p(b_p − log10 δ+)))`,
  `q = 0.2 + 0.6/(1+exp(a_q(b_q − log10 δ+)))`,
  `(a_A, b_A, a_p, b_p, a_q, b_q) = (2.39, 3.55, −0.22, 3.58, −1.09, 3.78)`
  (full-precision values in `model_b.DEFAULT_PARAMS`).

Fitted to the CICLoPE pipe spectra (δ+ = 4794–47 015) with a loss that also weakly penalises departure
from the channel variance trend `⟨p²⟩/τ_w² = 2.24 ln δ+ − 9.18`.

## Validity

Calibrated for zero-pressure-gradient BLs (δ+ 500–11 064), pipes (4794–47 015) and channels
(180–5186). Not intended for pressure gradients, roughness or compressibility. The package
warns when `Re_tau` is outside the calibrated range; e.g. the pipe model A amplitude A1 becomes
negative below δ+ = 1000. Model B is smooth beyond the range but unvalidated there.

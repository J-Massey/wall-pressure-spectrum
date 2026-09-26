"""Predict a wall-pressure spectrum from dimensional flow parameters."""
import numpy as np
import wallpressure as wp

# Zero-pressure-gradient boundary layer in air
f = np.logspace(1, 4.3, 200)  # Hz
spec = wp.predict_spectrum(
    f, model="B", flow="boundary_layer",
    u_tau=1.2, nu=1.5e-5, delta=0.1, u_outer=30.0, rho=1.2,
)
print(f"Re_tau = {spec.re_tau:.0f}")
i = np.argmax(spec.premultiplied)
print(f"peak f*phi/tau_w^2 = {spec.premultiplied[i]:.2f} at f = {f[i]:.0f} Hz (T+ = {spec.T_plus[i]:.1f})")
print(f"phi_pp at 1 kHz = {np.interp(1e3, f, spec.phi_pp):.3e} Pa^2/Hz")
print(f"<p^2>/tau_w^2 = {wp.variance_plus(spec.f_plus, spec.premultiplied):.2f} "
      f"(Lee & Moser trend: {wp.variance_lee_moser(spec.re_tau):.2f})")

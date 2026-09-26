import numpy as np
import pytest

import wallpressure as wp


def _legacy_model_a(flow, T, re, ratio):
    """Verbatim transcription of the original paper code (paper_scripts/model_A.py)."""
    p = {"boundary_layer": (2.2, 3.9, 20, 1.4, 1.2, 0.82, 7),
         "pipe": (2.9 * (1 - 1000 / re), 4.3, 20, 0.91, 1.0, 0.18, 7),
         "channel": (2.1 * (1 - 100 / re), 4.4, 12, 0.9, 1.0, 0.6, 3)}[flow]
    A1, s1, T1, k2, s2, To, r2 = p
    with np.errstate(all="ignore"):
        rv = np.exp(0.5 * T) / (np.exp(0.5 * r2) + np.exp(0.5 * T))
    rv = np.nan_to_num(rv, nan=1)  # original code's overflow handling
    g1 = A1 * np.exp(-s1 * (np.log10(T) - np.log10(T1)) ** 2)
    g2 = k2 * (np.log10(re) - 2.2) * np.exp(-s2 * (np.log10(T) - np.log10(To * re * ratio)) ** 2)
    return rv * (g1 + g2)


@pytest.mark.parametrize("flow", wp.FLOWS)
@pytest.mark.parametrize("re", [1000.0, 5000.0, 47000.0])
@pytest.mark.filterwarnings("ignore:Re_tau")
def test_model_a_matches_paper_code(flow, re):
    T = np.logspace(0, 4.5, 200)
    ratio = 0.045
    np.testing.assert_allclose(wp.model_a(T, re, flow, ratio), _legacy_model_a(flow, T, re, ratio), rtol=1e-12)


def test_bl_default_ratio():
    T = np.logspace(0, 4, 50)
    re = 4000.0
    np.testing.assert_allclose(wp.model_a(T, re, "boundary_layer"),
                               wp.model_a(T, re, "boundary_layer", np.sqrt(wp.cf_approx(re))))


def test_pipe_needs_ratio():
    with pytest.raises(ValueError):
        wp.model_a(np.array([10.0]), 5000.0, "pipe")


def test_model_b_parameters_and_peak():
    A, ph, q = wp.outer_parameters(47015.0)
    assert 0.9 < A < 1.0 and -2 < ph < -0.5 and 0.2 < q < 0.8
    fp = np.logspace(-6, 1, 400)
    y = wp.model_b(fp, 4794.0, 0.0422)
    assert np.all(np.isfinite(y)) and y.max() > 1.0
    # inner peak of order 1.6 near f+ ~ 0.05-0.1
    assert 0.03 < fp[np.argmax(y)] < 0.2


@pytest.mark.filterwarnings("ignore:Re_tau")
def test_model_b_variance_grows_with_re():
    fp = np.logspace(-9, 2, 2000)
    v = [wp.variance_plus(fp, wp.model_b(fp, re, 0.04)) for re in (5e3, 2e4, 5e4)]
    assert v[0] < v[1] < v[2]
    # close to the Lee & Moser trend that model B was fitted to
    for re, vi in zip((5e3, 2e4, 5e4), v):
        assert abs(vi - wp.variance_lee_moser(re)) < 1.0


def test_predict_spectrum_consistency():
    f = np.logspace(1, 4, 50)
    s = wp.predict_spectrum(f, model="B", flow="pipe", u_tau=0.162, nu=1.5e-5, delta=0.4505,
                            u_outer=3.837, rho=1.2)
    assert s.re_tau == pytest.approx(4863, rel=0.01)
    np.testing.assert_allclose(s.f * s.phi_pp, s.premultiplied * s.tau_w**2)


def test_out_of_range_warns():
    with pytest.warns(UserWarning):
        wp.model_a(np.array([10.0]), 500.0, "pipe", 0.04)
    with pytest.warns(UserWarning):
        wp.model_b(np.array([0.1]), 500.0, 0.04)

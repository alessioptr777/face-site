"""
Test unitari per Near-Miss Exception (det>=0.85, 0.47<=best_score<0.50).
hits_047 = numero di ref_embeddings con score >= 0.47; margin = best_score - second_best.
"""
import pytest

from main import _near_miss_exception_accept, _near_miss_tier2_accept


def test_near_miss_det_86_best_49_hits_3_margin_none_accept():
    """det=0.86 best=0.49 hits=3 margin=None -> ACCEPT (hits_047>=2)."""
    assert _near_miss_exception_accept(0.86, 0.49, 3, None) is True


def test_near_miss_det_86_best_49_hits_0_margin_002_reject():
    """det=0.86 best=0.49 hits=0 margin=0.02 -> REJECT (hits<2 e margin<0.10)."""
    assert _near_miss_exception_accept(0.86, 0.49, 0, 0.02) is False


def test_near_miss_det_91_best_48_hits_3_accept():
    """det=0.91 best=0.48 hits=3 -> ACCEPT (det>=0.90 richiede hits>=3)."""
    assert _near_miss_exception_accept(0.91, 0.48, 3, None) is True


def test_near_miss_det_91_best_48_hits_2_reject():
    """det=0.91 best=0.48 hits=2 -> REJECT (det>=0.90 richiede hits>=3)."""
    assert _near_miss_exception_accept(0.91, 0.48, 2, None) is False


def test_near_miss_det_85_best_49_hits_2_margin_none_accept():
    """det=0.85 best=0.49 hits=2 margin=None -> ACCEPT (hits_047>=2)."""
    assert _near_miss_exception_accept(0.85, 0.49, 2, None) is True


def test_near_miss_det_85_best_49_hits_0_margin_10_accept():
    """det=0.85 best=0.49 hits=0 margin=0.10 -> ACCEPT (margin>=0.10)."""
    assert _near_miss_exception_accept(0.85, 0.49, 0, 0.10) is True


def test_near_miss_best_below_47_reject():
    """best_score < 0.47 -> mai ACCEPT per near-miss."""
    assert _near_miss_exception_accept(0.90, 0.46, 5, None) is False


def test_near_miss_best_above_50_not_near_miss():
    """best_score >= 0.50 -> non è near-miss (soglia normale)."""
    assert _near_miss_exception_accept(0.90, 0.50, 1, None) is False


# --- NEAR_MISS_TIER2 (det>=0.87, 0.40<=best<0.47, margin>=0.30, hits_040>=4 o face_count<=2) ---


def test_tier2_det_872_best_41_margin_393_hits8_accept():
    """det=0.872 best=0.410 margin=0.393 hits_040=8 -> ACCEPT."""
    assert _near_miss_tier2_accept(0.872, 0.410, 0.393, 8, None) is True


def test_tier2_det_872_best_41_margin_low_reject():
    """det=0.872 best=0.410 margin=0.05 hits_040=8 -> REJECT (margin<0.30)."""
    assert _near_miss_tier2_accept(0.872, 0.410, 0.05, 8, None) is False


def test_tier2_det_88_best_42_margin_31_hits2_reject():
    """det=0.88 best=0.42 margin=0.31 hits_040=2 -> REJECT (hits_040<4 e face_count non<=2)."""
    assert _near_miss_tier2_accept(0.88, 0.42, 0.31, 2, 5) is False


def test_tier2_det_86_no_tier2_reject():
    """det=0.86 best=0.42 margin=0.31 hits_040=8 -> REJECT (tier2 solo se det>=0.87)."""
    assert _near_miss_tier2_accept(0.86, 0.42, 0.31, 8, None) is False

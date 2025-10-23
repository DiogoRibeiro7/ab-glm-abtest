from __future__ import annotations

import numpy as np

from ab_glm.pipeline import (
    brier_score,
    run_pipeline,
)


def test_brier_score_basic():
    y = np.array([0, 1, 1, 0], dtype=float)
    p = np.array([0.1, 0.8, 0.6, 0.2], dtype=float)
    bs = brier_score(y, p)
    assert 0.0 <= bs <= 1.0


def test_run_pipeline_ranges():
    res = run_pipeline(link="logit")
    assert 0.0 <= res.p_control <= 1.0
    assert 0.0 <= res.p_treated <= 1.0
    assert res.n_obs > 0 and res.n_users > 0
    # ATE can be negative or positive; RR should be positive
    assert res.rr > 0.0

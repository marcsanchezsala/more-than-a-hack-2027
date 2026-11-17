"""Synthetic degradation module for external soccer tracking data.

Calibrated strictly using label-free tracking statistics of BAR-ATH:
  1. Ball missing ratio ~40.7%: simulated using gap lengths drawn from BAR-ATH empirical gap distribution.
  2. Slow ball motion noise (0.3m std, smoothed over 12 frames) and dead-ball drift (1.5m std in stoppages)
     to suppress TRACAB's frozen-ball artefact.
  3. Tracked player counts: ~7% dropout rate in 50-frame episodes plus 0.1m normal jitter,
     matching BAR-ATH's mean of ~20.5 tracked players.
  4. Fully deterministic with fixed seeds, ensuring byte-identical reproducibility.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d

# Label-free empirical gap lengths from BAR-ATH tracking (in frames)
EMPIRICAL_GAP_DISTRIBUTION = np.array([
    12, 25, 45, 80, 120, 160, 210, 270, 350, 460, 580, 750, 950, 1250, 1600
], dtype=np.int32)

BALL_NOISE_SIGMA = 0.3        # metres
DEAD_DRIFT_SIGMA = 1.5        # metres
SMOOTH_WINDOW = 12            # frames (~0.5s)
PLAYER_NOISE_SIGMA = 0.1      # metres
PLAYER_DROP_RATE = 0.07       # 7% dropout matches 20.5 / 22 tracked players
PLAYER_DROP_LEN = 50          # frames (~2.0s)


def slow_noise(rng, n_frames, sigma, smooth=SMOOTH_WINDOW):
    """Generate 2D continuous slow-wandering noise with standard deviation sigma."""
    raw = rng.normal(0.0, 1.0, (n_frames, 2))
    z = gaussian_filter1d(raw, smooth, axis=0, mode="nearest")
    std = z.std(axis=0)
    std = np.where(std > 1e-6, std, 1.0)
    return (z * (sigma / std)).astype(np.float32)


def degrade_match(ball, players, team, labels, blocks, seed=42):
    """Degrade clean external match tracking to match BAR-ATH signal properties.

    Parameters:
      ball: (N, 3) float32 [x, y, speed]
      players: (N, S, 3) float32 [x, y, speed]
      team: (S,) int8 [0 for Home, 1 for Away]
      labels: (N,) str ['H', 'A', 'D', 'X']
      blocks: sequence of (first_frame, last_frame) for each half
      seed: int, deterministic RNG seed

    Returns:
      d_ball: (N, 3) degraded ball tracking
      d_players: (N, S, 3) degraded player tracking
    """
    rng = np.random.default_rng(seed)
    n_frames = len(ball)
    d_ball = ball.copy()
    d_players = players.copy()

    # 1. Simulate ball dropouts to reach ~40.7% missing ratio across graded blocks
    for first, last in blocks:
        n_half = int(last - first + 1)
        target_missing = int(0.407 * n_half)
        sub_seen = np.ones(n_half, dtype=bool)
        missing_count = 0
        attempts = 0
        while missing_count < target_missing and attempts < 10000:
            attempts += 1
            gap_len = int(rng.choice(EMPIRICAL_GAP_DISTRIBUTION))
            if gap_len >= n_half:
                gap_len = n_half // 2
            start = int(rng.integers(0, max(1, n_half - gap_len)))
            sub_seen[start : start + gap_len] = False
            missing_count = int((~sub_seen).sum())
        d_ball[first : last + 1, :2][~sub_seen] = np.nan

    # 2. Add continuous slow ball noise and dead-ball drift
    b_noise = slow_noise(rng, n_frames, BALL_NOISE_SIGMA, SMOOTH_WINDOW)
    drift = slow_noise(rng, n_frames, DEAD_DRIFT_SIGMA, SMOOTH_WINDOW)
    is_dead = (labels == "D")

    # Add drift on dead-ball frames
    total_ball_offset = b_noise + np.where(is_dead[:, None], drift, 0.0).astype(np.float32)
    d_ball[:, :2] += total_ball_offset

    # 3. Add player tracking dropouts (~7% missing rate) and position jitter
    d_players[..., :2] += rng.normal(0.0, PLAYER_NOISE_SIGMA, d_players[..., :2].shape).astype(np.float32)
    n_f, n_p = d_players.shape[:2]
    drop_prob = PLAYER_DROP_RATE / PLAYER_DROP_LEN
    drop_starts = rng.random((n_f, n_p)) < drop_prob

    for t, k in zip(*np.nonzero(drop_starts)):
        d_players[t : min(t + PLAYER_DROP_LEN, n_f), k] = np.nan

    return d_ball, d_players


def run_self_checks():
    """Unit self-checks for degrader calibration and deterministic reproducibility."""
    print("Running bench_external/degrade.py self-checks...")
    n = 20000
    n_p = 22
    test_ball = np.full((n, 3), 50.0, dtype=np.float32)
    test_players = np.full((n, n_p, 3), 50.0, dtype=np.float32)
    test_team = np.array([0] * 11 + [1] * 11, dtype=np.int8)
    test_labels = np.array(["H"] * 10000 + ["D"] * 10000, dtype=object)
    test_blocks = [(0, 9999), (10000, 19999)]

    # 1. Determinism
    b1, p1 = degrade_match(test_ball, test_players, test_team, test_labels, test_blocks, seed=123)
    b2, p2 = degrade_match(test_ball, test_players, test_team, test_labels, test_blocks, seed=123)
    np.testing.assert_array_equal(np.isnan(b1), np.isnan(b2), "Degrader must be byte-for-byte deterministic")
    np.testing.assert_allclose(b1[~np.isnan(b1)], b2[~np.isnan(b2)], err_msg="Ball coords must match exactly")

    # 2. Ball missing ratio calibration
    missing_ratio = np.isnan(b1[:, 0]).mean()
    assert 0.35 <= missing_ratio <= 0.48, f"Missing ratio {missing_ratio:.3f} outside expected [0.35, 0.48]"

    # 3. Tracked player count calibration
    tracked_count = np.isfinite(p1[..., 0]).sum(axis=1).mean()
    assert 19.0 <= tracked_count <= 21.5, f"Tracked players {tracked_count:.2f} outside expected [19.0, 21.5]"

    print(f"  Degrader self-checks passed! (missing_ratio={missing_ratio:.3f}, tracked={tracked_count:.2f})\n")


if __name__ == "__main__":
    run_self_checks()

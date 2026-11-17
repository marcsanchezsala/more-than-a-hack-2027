"""Label-convention smoothing for external benchmark ground truth.

Provides deterministic label-convention adjustments to test whether low external spellF1
is driven by micro-touch/deflection fragmentation in raw sensor tracking feeds (DFL BallPossession)
versus the match-level possession definition used in the BAR-ATH challenge.
"""

import numpy as np


def smooth_label_convention(labels_in, blocks, min_spell_frames=25, merge_micro_d=True):
    """Optionally smooth external ground truth labels to align with possession conventions.

    Parameters
    ----------
    labels_in : np.ndarray of shape (N,), dtype '<U1'
        Per-frame labels ('H', 'A', 'D', 'X').
    blocks : list of (first, last)
        Boundaries of active halves.
    min_spell_frames : int
        Minimum frame duration (e.g. 25 frames = 1.0s, 50 frames = 2.0s).
    merge_micro_d : bool
        Whether to merge micro-dead ball stretches shorter than min_spell_frames
        sandwiched between the same team (e.g. H - D - H -> H).

    Returns
    -------
    smoothed : np.ndarray of shape (N,), dtype '<U1'
    """
    if min_spell_frames <= 0:
        return labels_in.copy()

    labels = labels_in.copy()
    for first, last in blocks:
        sub = labels[first : last + 1].copy()
        changed = True
        iterations = 0
        while changed and iterations < 50:
            changed = False
            iterations += 1
            breaks = np.flatnonzero(sub[1:] != sub[:-1]) + 1
            bounds = np.concatenate(([0], breaks, [len(sub)]))
            spells = [(bounds[i], bounds[i + 1], sub[bounds[i]]) for i in range(len(bounds) - 1)]

            for i in range(len(spells)):
                start, stop, lbl = spells[i]
                dur = stop - start
                prev_lbl = spells[i - 1][2] if i > 0 else "X"
                next_lbl = spells[i + 1][2] if i < len(spells) - 1 else "X"

                # Opponent deflection sandwich: A -> H (< min_spell_frames) -> A
                if lbl in ("H", "A") and dur < min_spell_frames:
                    opp = "A" if lbl == "H" else "H"
                    if prev_lbl == opp and next_lbl == opp:
                        sub[start:stop] = opp
                        changed = True
                        break

                # Micro dead-ball / pass bridge between same team: H -> D (< min_spell_frames) -> H
                if merge_micro_d and lbl == "D" and dur < min_spell_frames:
                    if prev_lbl in ("H", "A") and prev_lbl == next_lbl:
                        sub[start:stop] = prev_lbl
                        changed = True
                        break

        labels[first : last + 1] = sub
    return labels

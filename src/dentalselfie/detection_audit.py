"""Deterministic one-to-one localization diagnostics; no clinical relabeling."""
from __future__ import annotations

import numpy as np


def box_iou(a, b):
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    intersection = np.maximum(0, np.minimum(a[:, None, 2:], b[None, :, 2:]) -
                              np.maximum(a[:, None, :2], b[None, :, :2])).prod(2)
    area_a = np.maximum(0, a[:, 2:] - a[:, :2]).prod(1)
    area_b = np.maximum(0, b[:, 2:] - b[:, :2]).prod(1)
    union = area_a[:, None] + area_b[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def match_boxes(gt, predicted, scores, threshold=0.5):
    """Confidence-ordered matching; each prediction searches unmatched GT first.

    Non-TP predictions are partitioned into duplicate (IoU >= threshold with
    an already matched GT), localization (0.1 <= best IoU < threshold), and
    unmatched. Unmatched means annotation-relative, not proven healthy tissue.
    """
    gt = np.asarray(gt, float).reshape(-1, 4)
    predicted = np.asarray(predicted, float).reshape(-1, 4)
    scores = np.asarray(scores, float)
    if len(scores) != len(predicted):
        raise ValueError("Score/box count mismatch")
    ious = box_iou(predicted, gt)
    used, events = set(), []
    for i in np.argsort(-scores, kind="stable"):
        candidates = [j for j in range(len(gt)) if j not in used and ious[i, j] >= threshold]
        best = float(ious[i].max()) if len(gt) else 0.0
        if candidates:
            j = max(candidates, key=lambda j: ious[i, j])
            used.add(j)
            kind, overlap = "true_positive", float(ious[i, j])
        else:
            j, overlap = None, best
            kind = "duplicate" if best >= threshold else ("localization" if best >= 0.1 else "unmatched")
        events.append({"prediction_index": int(i), "gt_index": j, "kind": kind,
                       "score": float(scores[i]), "iou": overlap})
    return events, [j for j in range(len(gt)) if j not in used]

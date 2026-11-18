import numpy as np

from dentalselfie.detection_audit import box_iou, match_boxes


def test_empty_and_exact_boxes():
    assert box_iou([], [[0, 0, 10, 10]]).shape == (0, 1)
    events, missed = match_boxes([[0, 0, 10, 10]], [], [])
    assert events == [] and missed == [0]
    events, missed = match_boxes([[0, 0, 10, 10]], [[0, 0, 10, 10]], [0.8])
    assert events[0]["kind"] == "true_positive" and missed == []


def test_matching_uses_available_second_gt_before_duplicate():
    gt = [[0, 0, 10, 10], [2, 0, 12, 10]]
    events, missed = match_boxes(gt, [[0, 0, 10, 10], [0, 0, 10, 10]], [0.9, 0.8])
    assert [e["gt_index"] for e in events] == [0, 1]
    assert missed == []


def test_duplicate_localization_and_unmatched_partition():
    events, missed = match_boxes([[0, 0, 10, 10]],
                                [[0, 0, 10, 10], [0, 0, 10, 10], [5, 0, 15, 10], [30, 30, 40, 40]],
                                [0.9, 0.8, 0.7, 0.6])
    assert [e["kind"] for e in events] == ["true_positive", "duplicate", "localization", "unmatched"]
    assert missed == []
    assert np.isclose(events[2]["iou"], 1 / 3)

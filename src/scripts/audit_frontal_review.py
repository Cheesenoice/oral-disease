"""Validate frontal pilot review files without producing training labels."""
from pathlib import Path
import csv
import math


def audit(queue_path: Path, boxes_path: Path):
    with queue_path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    with boxes_path.open(encoding="utf-8-sig", newline="") as f:
        boxes = list(csv.DictReader(f))
    issues = []
    by_id = {}
    for row in rows:
        key = row['source_image_id']
        if key in by_id:
            issues.append(f'{key}: duplicate image id')
        by_id[key] = row
    counts = {}
    targets = {'upper_midline_diastema': 'diastema_review', 'anterior_crowding': 'crowding_review'}
    for box in boxes:
        key, target = box['source_image_id'], box['target']
        if key not in by_id or target not in targets:
            issues.append(f'{key}: unknown image or target')
            continue
        row = by_id[key]
        if row['view_review'] != 'frontal':
            issues.append(f'{key}: box on a non-frontal or unreviewed image')
        try:
            x1, y1, x2, y2 = (float(box[k]) for k in ('x_min', 'y_min', 'x_max', 'y_max'))
            if not all(map(math.isfinite, (x1, y1, x2, y2))) or not (0 <= x1 < x2 <= int(row['width']) and 0 <= y1 < y2 <= int(row['height'])):
                raise ValueError()
        except (ValueError, TypeError):
            issues.append(f'{key}: invalid box geometry')
        if box['review_state'] != 'approved' or not box['reviewer'].strip() or box['reviewer'].strip().lower() == 'pending':
            issues.append(f'{key}: box is not approved by a named reviewer')
        counts[(key, target)] = counts.get((key, target), 0) + 1
    ready = 0
    for row in rows:
        key = row['source_image_id']
        if row['view_review'] != 'frontal':
            continue
        for target, field in targets.items():
            status, count = row[field], counts.get((key, target), 0)
            if status == 'present' and not count:
                issues.append(f'{key}: {target} present without a box')
            if status == 'absent' and count:
                issues.append(f'{key}: {target} absent but has a box')
        reviewer = row['reviewer'].strip()
        if all(row[field] in ('present', 'absent') for field in targets.values()) and reviewer and reviewer.lower() != 'pending' and not any(issue.startswith(f'{key}:') for issue in issues):
            ready += 1
    return {'images': len(rows), 'boxes': len(boxes), 'reviewed_candidates': ready, 'issues': issues}


if __name__ == '__main__':
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', action='store_true', help='Audit the earlier 36-image pilot')
    args = parser.parse_args()
    base = Path('docs/reports/alignment/frontal_pilot_16_7' if args.pilot else 'data/processed/alignment_omni_frontal_candidates')
    print(json.dumps(audit(base / 'review_queue.csv', base / 'boxes_template.csv'), indent=2))

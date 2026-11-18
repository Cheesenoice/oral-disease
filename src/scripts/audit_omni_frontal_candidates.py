"""Screen exact and perceptual cross-split overlap among local OMNI candidates."""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'data/processed/alignment_omni_frontal_candidates'
REPORT = ROOT / 'docs/reports/alignment/omni_frontal_candidate_audit'


def dhash(path):
    with Image.open(path) as image:
        gray = image.convert('L').resize((9, 8))
        pixels = list(gray.getdata())
    value = 0
    for y in range(8):
        for x in range(8):
            value = (value << 1) | int(pixels[y * 9 + x] > pixels[y * 9 + x + 1])
    return value


def write_csv(path, fields, rows):
    with path.open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    with (BASE / 'review_queue.csv').open(encoding='utf-8-sig', newline='') as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 724
    REPORT.mkdir(parents=True, exist_ok=True)
    by_split = defaultdict(list)
    for row in rows:
        row['dhash'] = dhash(ROOT / row['image_path'])
        by_split[row['split']].append(row)
    train, val = by_split['train'], by_split['val']
    exact = sorted({r['sha256'] for r in train} & {r['sha256'] for r in val})
    group_train = defaultdict(list)
    group_val = defaultdict(list)
    for row in train: group_train[row['source_folder_1']].append(row['source_image_id'])
    for row in val: group_val[row['source_folder_1']].append(row['source_image_id'])
    shared_groups = sorted(set(group_train) & set(group_val))
    group_rows = [dict(source_folder_1=group, split='train', source_image_id=id)
                  for group in shared_groups for id in group_train[group]] + [
                  dict(source_folder_1=group, split='val', source_image_id=id)
                  for group in shared_groups for id in group_val[group]]
    write_csv(REPORT / 'shared_source_folder.csv', ('source_folder_1', 'split', 'source_image_id'), group_rows)
    pairs = []
    for a in train:
        for b in val:
            distance = (a['dhash'] ^ b['dhash']).bit_count()
            if distance <= 4:
                pairs.append(dict(train_id=a['source_image_id'], val_id=b['source_image_id'],
                                  dhash_distance=distance, train_path=a['image_path'], val_path=b['image_path']))
    pairs.sort(key=lambda r: (r['dhash_distance'], r['val_id'], r['train_id']))
    write_csv(REPORT / 'near_duplicate_review.csv', ('train_id', 'val_id', 'dhash_distance', 'train_path', 'val_path'), pairs)
    if pairs:
        thumb_w, thumb_h = 300, 190
        montage = Image.new('RGB', (thumb_w * 2, thumb_h * min(12, len(pairs))), 'white')
        draw = ImageDraw.Draw(montage)
        for idx, pair in enumerate(pairs[:12]):
            for col, key in enumerate(('train_path', 'val_path')):
                with Image.open(ROOT / pair[key]) as image:
                    thumb = image.convert('RGB')
                    thumb.thumbnail((thumb_w, thumb_h - 30))
                montage.paste(thumb, (col * thumb_w, idx * thumb_h + 24))
            draw.text((5, idx * thumb_h + 3), f'{pair["train_id"]} / {pair["val_id"]}  d={pair["dhash_distance"]}', fill='black')
        montage.save(REPORT / 'near_duplicate_first12.jpg', quality=90)
    excluded = sorted({r['source_image_id'] for r in train if r['source_folder_1'] in shared_groups} |
                      {r['train_id'] for r in pairs})
    write_csv(REPORT / 'conservative_train_exclusions.csv', ('source_image_id',),
              [dict(source_image_id=id) for id in excluded])
    summary = dict(images=len(rows), train=len(train), val=len(val), exact_cross_split_hashes=len(exact),
                   shared_first_folder_keys=shared_groups, cross_split_dhash_le4_pairs=len(pairs),
                   conservative_train_exclusions=len(excluded),
                   caution='Folder and dHash candidates are not verified patient identities or confirmed same-scene pairs.')
    (REPORT / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

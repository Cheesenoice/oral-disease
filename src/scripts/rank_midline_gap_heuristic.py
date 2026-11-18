"""Rank OMNI source-train images by central dark-seam evidence for visual review."""
from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'data/processed/alignment_omni_frontal_candidates/review_queue.csv'
OUT = ROOT / 'docs/reports/alignment/midline_heuristic_review'


def seam_score(path: Path):
    with Image.open(path) as image:
        array = np.asarray(image.convert('L'), dtype=np.float32)
    h, w = array.shape
    if (w, h) != (512, 341):
        array = np.asarray(Image.fromarray(array.astype('uint8')).resize((512, 341)), dtype=np.float32)
    best = 0.0
    best_x = None
    for x in range(218, 296):
        left = array[115:220, x - 17:x - 8].mean(axis=1)
        mid = array[115:220, x - 2:x + 3].mean(axis=1)
        right = array[115:220, x + 8:x + 17].mean(axis=1)
        evidence = np.minimum(left, right) - mid
        valid = (left > 120) & (right > 120) & (evidence > 15)
        for start in range(0, len(evidence) - 12):
            band = evidence[start:start+13]
            support = valid[start:start+13]
            score = float((band * support).sum())
            if score > best:
                best, best_x = score, x
    return round(best, 2), best_x


def main():
    with SOURCE.open(encoding='utf-8-sig', newline='') as file:
        rows = [r for r in csv.DictReader(file) if r['split'] == 'train']
    OUT.mkdir(parents=True, exist_ok=True)
    ranking = []
    for row in rows:
        score, x = seam_score(ROOT / row['image_path'])
        ranking.append({'source_image_id': row['source_image_id'], 'image_path': row['image_path'],
                        'seam_score': score, 'seam_x': x})
    ranking.sort(key=lambda r: -r['seam_score'])
    with (OUT / 'ranking.csv').open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=ranking[0].keys())
        writer.writeheader()
        writer.writerows(ranking)
    for page in range(0, 72, 12):
        sheet = Image.new('RGB', (512 * 4, 375 * 3), 'white')
        draw = ImageDraw.Draw(sheet)
        for idx, row in enumerate(ranking[page:page + 12]):
            with Image.open(ROOT / row['image_path']) as image:
                thumb = image.convert('RGB')
                thumb.thumbnail((510, 340))
            x, y = idx % 4 * 512, idx // 4 * 375
            sheet.paste(thumb, (x, y + 25))
            draw.text((x + 5, y + 5), f'{row["source_image_id"]} {row["seam_score"]:.0f}', fill='black')
        sheet.save(OUT / f'top_{page//12+1}.jpg', quality=90)
    low_pool = [r for r in ranking if r['seam_score'] < 650]
    low_sample = random.Random(29).sample(low_pool, min(24, len(low_pool)))
    for page in range(0, len(low_sample), 12):
        sheet = Image.new('RGB', (512 * 4, 375 * 3), 'white')
        draw = ImageDraw.Draw(sheet)
        for idx, row in enumerate(low_sample[page:page+12]):
            with Image.open(ROOT / row['image_path']) as image:
                thumb = image.convert('RGB')
                thumb.thumbnail((510, 340))
            x, y = idx % 4 * 512, idx // 4 * 375
            sheet.paste(thumb, (x, y+25))
            draw.text((x+5, y+5), row['source_image_id'], fill='black')
        sheet.save(OUT / f'low_sample_{page//12+1}.jpg', quality=90)
    lookup = {r['source_image_id']: r['seam_score'] for r in ranking}
    for id in ('201916202112081', '202139202205251', '202212202202071',
               '202362202308141', '201875201810241', '202243202207131', '201883201809131'):
        print(id, lookup[id], next(i+1 for i, r in enumerate(ranking) if r['source_image_id'] == id))


if __name__ == '__main__':
    main()

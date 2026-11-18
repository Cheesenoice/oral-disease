"""Rank local OMNI images for manual review using an occlusal source model."""
from __future__ import annotations

import csv
import random
from pathlib import Path

from PIL import Image, ImageDraw
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'data/processed/alignment_omni_frontal_candidates'
OUT = ROOT / 'docs/reports/alignment/omni_candidate_probe'
MODEL = ROOT / 'data/runs/alignment_occlusal_v1/runs/occlusal_yolo11n/weights/best.pt'


def main():
    with (BASE / 'review_queue.csv').open(encoding='utf-8-sig', newline='') as file:
        rows = list(csv.DictReader(file))
    OUT.mkdir(parents=True, exist_ok=True)
    paths = [str(ROOT / row['image_path']) for row in rows]
    model = YOLO(str(MODEL))
    ranked = []
    for row, result in zip(rows, model.predict(paths, imgsz=640, conf=0.05, device='cpu', stream=True, verbose=False)):
        best = {0: 0.0, 1: 0.0}
        for box in result.boxes:
            cls = int(box.cls[0])
            best[cls] = max(best[cls], float(box.conf[0]))
        ranked.append({'split': row['split'], 'source_image_id': row['source_image_id'],
                       'image_path': row['image_path'], 'diastema_probe_score': round(best[0], 5),
                       'crowding_probe_score': round(best[1], 5)})
    with (OUT / 'ranking.csv').open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=ranked[0].keys())
        writer.writeheader()
        writer.writerows(ranked)
    for target in ('diastema', 'crowding'):
        priority = sorted(ranked, key=lambda r: -r[f'{target}_probe_score'])[:48]
        for page in range(0, len(priority), 12):
            sheet = Image.new('RGB', (512 * 4, 375 * 3), 'white')
            draw = ImageDraw.Draw(sheet)
            for idx, row in enumerate(priority[page:page+12]):
                with Image.open(ROOT / row['image_path']) as image:
                    thumb = image.convert('RGB')
                    thumb.thumbnail((510, 340))
                x, y = (idx % 4) * 512, (idx // 4) * 375
                sheet.paste(thumb, (x, y+25))
                draw.text((x+5, y+5), f'{row["source_image_id"]} {row[f"{target}_probe_score"]:.3f}', fill='black')
            sheet.save(OUT / f'{target}_top{page//12+1}.jpg', quality=90)
    zero = [r for r in ranked if r['split'] == 'train' and r['diastema_probe_score'] == 0 and r['crowding_probe_score'] == 0]
    for page, subset in enumerate((random.Random(17).sample(zero, min(24, len(zero)))[:12],
                                   random.Random(17).sample(zero, min(24, len(zero)))[12:])):
        sheet = Image.new('RGB', (512 * 4, 375 * 3), 'white')
        draw = ImageDraw.Draw(sheet)
        for idx, row in enumerate(subset):
            with Image.open(ROOT / row['image_path']) as image:
                thumb = image.convert('RGB')
                thumb.thumbnail((510, 340))
            x, y = (idx % 4) * 512, (idx // 4) * 375
            sheet.paste(thumb, (x, y+25))
            draw.text((x+5, y+5), row['source_image_id'], fill='black')
        sheet.save(OUT / f'zero_score_sample_{page+1}.jpg', quality=90)
    print('Scored', len(ranked), 'images; ranked review aid:', OUT)


if __name__ == '__main__':
    main()

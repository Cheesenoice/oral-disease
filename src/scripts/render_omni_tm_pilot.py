"""Render existing broad OMNI TM boxes as visual review aids, never target labels."""
import csv
import json
import zipfile
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
PILOT = ROOT / 'data/processed/alignment_omni_pilot'
OUT = ROOT / 'docs/reports/alignment/omni_tm_pilot_review'
OUT.mkdir(parents=True, exist_ok=True)
with (PILOT / 'pilot_provenance.csv').open(newline='', encoding='utf-8-sig') as f:
    rows = list(csv.DictReader(f))
with zipfile.ZipFile(ROOT / 'data/raw/alignment_omni/OMNI_COCO.zip') as z:
    annotations = {}
    for split in ('train', 'val'):
        coco = json.loads(z.read(f'omni_coco/annotations/instances_{split}.json'))
        by_image = defaultdict(list)
        for annotation in coco['annotations']:
            if annotation['category_id'] == 7:
                by_image[str(annotation['image_id'])].append(annotation['bbox'])
        annotations[split] = by_image
    for split in ('train', 'val'):
        selected = [r for r in rows if r['split'] == split]
        for start in range(0, len(selected), 12):
            sheet = Image.new('RGB', (512 * 4, 375 * 3), 'white')
            draw = ImageDraw.Draw(sheet)
            for index, row in enumerate(selected[start:start + 12]):
                image = Image.open(PILOT / 'images' / split / (row['source_image_id'] + '.jpg')).convert('RGB')
                image.thumbnail((510, 340))
                x0, y0 = (index % 4) * 512, (index // 4) * 375
                sheet.paste(image, (x0, y0 + 25))
                for x, y, w, h in annotations[split][row['source_image_id']]:
                    draw.rectangle((x0 + x, y0 + 25 + y, x0 + x + w, y0 + 25 + y + h), outline='yellow', width=3)
                draw.text((x0 + 5, y0 + 5), row['source_image_id'], fill='black')
            sheet.save(OUT / f'{split}_{start // 12 + 1}.jpg', quality=92)
print('Rendered broad TM annotation overlays:', OUT)

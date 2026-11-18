"""Extract source-split-locked OMNI frontal candidates for local review only."""
from __future__ import annotations

import csv
import hashlib
import json
import zipfile
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / 'data/raw/alignment_omni/OMNI_COCO.zip'
SOURCE = ROOT / 'docs/reports/alignment/omni_frontal_review_queue.csv'
OUTPUT = ROOT / 'data/processed/alignment_omni_frontal_candidates'
EXPECTED_ARCHIVE_SHA256 = '3ea84e0b7f28526f80b000dfd9bfdf4677f54d66e24d5a60eedfc93afe87f525'
FIELDS = ('split', 'source_image_id', 'archive_member', 'sha256', 'image_path', 'width', 'height',
          'source_folder_1', 'source_folder_2', 'tm_boxes', 'view_review', 'dentition_review',
          'diastema_review', 'crowding_review', 'reviewer', 'review_notes')


def main():
    assert hashlib.sha256(ARCHIVE.read_bytes()).hexdigest() == EXPECTED_ARCHIVE_SHA256
    with SOURCE.open(encoding='utf-8-sig', newline='') as file:
        source = [r for r in csv.DictReader(file) if r['suffix'] == '1' and r['split'] in ('train', 'val')]
    assert len(source) == 724
    assert {(r['split'], r['source_image_id']) for r in source}.__len__() == len(source)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    with zipfile.ZipFile(ARCHIVE) as archive:
        for item in source:
            member = item['archive_member']
            assert member.startswith(f'omni_coco/{item["split"]}/') and '..' not in Path(member).parts
            content = archive.read(member)
            digest = hashlib.sha256(content).hexdigest()
            assert digest == item['sha256'], f'Source hash mismatch: {member}'
            path = OUTPUT / 'images' / item['split'] / f'{item["source_image_id"]}.jpg'
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, f'Extracted file changed: {path}'
            else:
                path.write_bytes(content)
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                width, height = image.size
            rows.append({
                'split': item['split'], 'source_image_id': item['source_image_id'],
                'archive_member': member, 'sha256': digest,
                'image_path': path.relative_to(ROOT).as_posix(),
                'width': width, 'height': height,
                'source_folder_1': item['folder_1'], 'source_folder_2': item['folder_2'],
                'tm_boxes': item['tm_boxes'], 'view_review': 'pending', 'dentition_review': 'pending',
                'diastema_review': 'pending', 'crowding_review': 'pending', 'reviewer': 'pending',
                'review_notes': 'pending',
            })
    manifest = OUTPUT / 'review_queue.csv'
    if manifest.exists():
        with manifest.open(encoding='utf-8-sig', newline='') as file:
            old = list(csv.DictReader(file))
        assert [(r['split'], r['source_image_id'], r['sha256']) for r in old] == [
            (r['split'], r['source_image_id'], r['sha256']) for r in rows], 'Existing review queue differs from source'
        print('Preserved existing review decisions:', manifest)
    else:
        with manifest.open('w', encoding='utf-8', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
    boxes = OUTPUT / 'boxes_template.csv'
    if not boxes.exists():
        with boxes.open('w', encoding='utf-8', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(('source_image_id', 'target', 'x_min', 'y_min', 'x_max', 'y_max', 'reviewer', 'review_state', 'notes'))
    summary = {'source_archive_sha256': EXPECTED_ARCHIVE_SHA256,
               'candidate_rule': 'source filename suffix 1; not a confirmed view',
               'train': sum(r['split'] == 'train' for r in rows),
               'val': sum(r['split'] == 'val' for r in rows),
               'clinical_label_state': 'pending', 'rights_state': 'redistribution unverified'}
    (OUTPUT / 'preparation.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

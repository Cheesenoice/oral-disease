"""Create self-contained, controlled calculus split and training notebooks."""
from pathlib import Path
import json
import nbformat as nb
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / "notebooks/calculus"


def notebook(title, steps):
    cells = [nb.v4.new_markdown_cell(title)]
    for markdown, code in steps:
        cells.extend([nb.v4.new_markdown_cell(markdown), nb.v4.new_code_cell(code)])
    return nb.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}})


audit = notebook("# 14.8 - Conservative source-image grouping\n\nHistorical calculus validation contains repeated source scenes. Preserve the existing validation membership and remove connected training images. Source test remains unopened. Similarity links are conservative exclusions, not confirmed patient identities.", [
    ("## 1. Inspect the historical manifest\nUse the exact downloaded positive-only baseline inputs. Verify their counts before grouping.", '''from pathlib import Path
import hashlib, json, re
from collections import Counter
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
root = Path.cwd()
while not (root / 'pyproject.toml').exists():
    if root == root.parent: raise RuntimeError('Project root unavailable')
    root = root.parent
source = root / 'data/runs/calculus_v3/calculus_data'
report = root / 'docs/reports/calculus_grouped'
report.mkdir(parents=True, exist_ok=True)
records = []
for split in ['train', 'val']:
    for path in sorted((source / 'images' / split).iterdir()):
        label = source / 'labels' / split / (path.stem + '.txt')
        with Image.open(path) as image:
            gray = np.asarray(ImageOps.grayscale(image).resize((9, 8)))
            width, height = image.size
        records.append(dict(name=path.name, original_split=split,
            source_key=re.split(r'\\.rf\\.', path.stem)[0],
            image_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            label_sha256=hashlib.sha256(label.read_bytes()).hexdigest(),
            dhash=int.from_bytes(np.packbits(gray[:, 1:] > gray[:, :-1]).tobytes(), 'big'),
            width=width, height=height, boxes=len(label.read_text().splitlines())))
assert Counter(r['original_split'] for r in records) == {'train': 846, 'val': 239}
display(pd.DataFrame(records).groupby('original_split').agg(images=('name','size'), boxes=('boxes','sum')))'''),
    ("## 2. Build conservative connected groups\nLink identical bytes, matching pre-Roboflow names and dHash distance <=4 across all retained images. Transitive components touching validation are excluded from training. A dHash match alone is a review candidate. This policy can over-group similar scenes and does not establish patient independence.", '''parent = list(range(len(records)))
def find(index):
    while parent[index] != index:
        parent[index] = parent[parent[index]]
        index = parent[index]
    return index
def union(left, right):
    parent[find(left)] = find(right)
edges = []
for left, a in enumerate(records):
    for right in range(left + 1, len(records)):
        b = records[right]
        reasons = []
        if a['image_sha256'] == b['image_sha256']: reasons.append('exact_bytes')
        if a['source_key'] == b['source_key']: reasons.append('source_filename')
        distance = (a['dhash'] ^ b['dhash']).bit_count()
        if distance <= 4: reasons.append('dhash_candidate')
        if reasons:
            union(left, right)
            edges.append(dict(left=left, right=right, distance=distance, reasons='|'.join(reasons)))
validation_groups = {find(i) for i, r in enumerate(records) if r['original_split'] == 'val'}
for i, row in enumerate(records):
    row['group'] = find(i)
    row['split'] = 'val' if row['original_split'] == 'val' else ('excluded' if find(i) in validation_groups else 'train')
frame = pd.DataFrame(records)
assert set(frame.loc[frame.split == 'train', 'group']).isdisjoint(frame.loc[frame.split == 'val', 'group'])
assert (frame.split == 'train').sum() >= 100, 'Insufficient independent training support'
display(frame.groupby('split').agg(images=('name','size'), boxes=('boxes','sum'), groups=('group','nunique')))
frame.groupby('split').size().plot.bar(title='Conservative grouped split membership', ylabel='Images')
plt.tight_layout(); plt.show()'''),
    ("## 3. Inspect excluded scenes\nShow representative cross-split candidate pairs. Names and measured distances make the exclusions auditable. These examples do not clinically validate every pair.", '''cross = [e for e in edges if records[e['left']]['original_split'] != records[e['right']]['original_split']]
fig, axes = plt.subplots(4, 2, figsize=(12, 12))
for ax in axes.flat: ax.axis('off')
for pair_axes, edge in zip(axes, sorted(cross, key=lambda e: e['distance'])[:4]):
    for ax, key in zip(pair_axes, ['left', 'right']):
        row = records[edge[key]]
        with Image.open(source / 'images' / row['original_split'] / row['name']) as im:
            ax.imshow(im)
        ax.set_title(row['original_split'] + ' | distance ' + str(edge['distance']) + '\\n' + row['source_key'][:50], fontsize=9)
plt.tight_layout(); plt.savefig(report / 'candidate_pairs.png'); plt.show()
display(pd.DataFrame(cross).head(12))'''),
    ("## 4. Freeze provenance and split\nSave all exclusions, graph edges and image/label checksums. The manifest hash is embedded in both training notebooks. This is a validation development split, with positive-only source images and no patient-level guarantee.", '''frame.to_csv(report / 'membership.csv', index=False)
pd.DataFrame(edges).to_csv(report / 'group_edges.csv', index=False)
manifest = {'protocol': 'calculus_conservative_groups_v1', 'dhash_threshold': 4,
    'policy': 'keep historical validation; exclude connected training components',
    'source': 'singh868/multi-class-oral-disease-detection-dataset/1',
    'records': [{k:v for k,v in r.items() if k != 'dhash'} for r in records]}
payload = json.dumps(manifest, sort_keys=True, indent=2).encode('utf-8')
(report / 'manifest.json').write_bytes(payload)
digest = hashlib.sha256(payload).hexdigest()
(report / 'manifest.sha256').write_text(digest + '\\n')
print('Manifest SHA-256:', digest)
print('Training / validation / excluded:', frame.split.value_counts().to_dict())''')])
audit_path = FOLDER / '14_8_calculus_grouped_split.ipynb'
nb.write(audit, audit_path)
NotebookClient(audit, timeout=600, kernel_name='python3', resources={'metadata': {'path': str(ROOT)}}).execute()
nb.write(audit, audit_path)
manifest_path = ROOT / 'docs/reports/calculus_grouped/manifest.json'
payload = manifest_path.read_text(encoding='utf-8')
digest = (manifest_path.with_suffix('.sha256')).read_text().strip()

for suffix in ['n', 's']:
    run_name = 'calculus_grouped_v8' + suffix
    train = notebook(f'# 14.9{suffix} - YOLOv8{suffix} controlled grouped baseline\n\nPositive-only calculus localization on a conservative source-group split. Historical scores use a different split and are context only. One seed, no test evaluation. Every image and label is checked against the locally audited manifest.', [
        ('## 1. Install the shared environment\nBoth arms use identical pinned Ultralytics and training settings.', '%pip install -q ultralytics==8.3.228'),
        ('## 2. Verify GPU and frozen protocol\nUse one T4 worker. Preserve the exact manifest and its checksum in outputs.', f'''from pathlib import Path
import hashlib, json, shutil, time
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import torch
from ultralytics import YOLO
payload = {payload!r}
assert hashlib.sha256(payload.encode('utf-8')).hexdigest() == {digest!r}
manifest = json.loads(payload)
Path('/kaggle/working/grouped_manifest.json').write_text(payload)
print('Manifest:', {digest!r})
print('Visible GPUs:', [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
assert torch.cuda.is_available()
display(pd.DataFrame(manifest['records']).groupby('split').agg(images=('name','size'), boxes=('boxes','sum')))'''),
        ('## 3. Stage verified images and labels\nResolve each original train/valid file and check both SHA-256 values. Retain calculus rows only. Excluded images are never staged. Source test is never read.', '''candidates = list(Path('/kaggle/input').rglob('data.yaml'))
assert len(candidates) == 1, candidates
source = candidates[0].parent
assert '0: calculus' in candidates[0].read_text().lower()
target = Path('/tmp/calculus_grouped_v1')
for row in manifest['records']:
    if row['split'] == 'excluded': continue
    original_split = 'valid' if row['original_split'] == 'val' else 'train'
    image = source / original_split / 'images' / row['name']
    label = source / original_split / 'labels' / (image.stem + '.txt')
    assert hashlib.sha256(image.read_bytes()).hexdigest() == row['image_sha256'], image.name
    rows = [line for line in label.read_text().splitlines() if line.split() and float(line.split()[0]) == 0]
    label_bytes = ('\\n'.join(rows) + '\\n').encode()
    assert hashlib.sha256(label_bytes).hexdigest() == row['label_sha256'], label.name
    for kind in ['images', 'labels']: (target / kind / row['split']).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(image, target / 'images' / row['split'] / image.name)
    (target / 'labels' / row['split'] / label.name).write_bytes(label_bytes)
for split in ['train', 'val']:
    actual = len(list((target / 'images' / split).iterdir()))
    assert actual == sum(r['split'] == split for r in manifest['records'])
    print(split, actual)
(target / 'data.yaml').write_text(f'path: {target}\\ntrain: images/train\\nval: images/val\\nnames:\\n  0: calculus\\n')'''),
        ('## 4. Visualize annotations and image transformation\nShow a retained image, its boxes, and aspect-preserving letterbox geometry at 640. During training YOLO applies its own matched augmentation and converts pixels to floating point in [0,1].', '''from ultralytics.data.augment import LetterBox
import numpy as np
image_path = sorted((target / 'images/train').iterdir())[0]
im = Image.open(image_path).convert('RGB')
draw = ImageDraw.Draw(im)
for line in (target / 'labels/train' / (image_path.stem + '.txt')).read_text().splitlines():
    _, x, y, w, h = map(float, line.split())
    draw.rectangle(((x-w/2)*im.width, (y-h/2)*im.height, (x+w/2)*im.width, (y+h/2)*im.height), outline='red', width=2)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(im); axes[0].set_title('Original with verified boxes')
axes[1].imshow(LetterBox((640,640))(image=np.asarray(im))); axes[1].set_title('Aspect-preserving 640 letterbox')
for ax in axes: ax.axis('off')
plt.tight_layout(); plt.show()'''),
        ('## 5. Train the matched baseline\nOnly model scale differs between the two notebooks. Fixed 60 epochs, patience disabled, AdamW, batch 16, seed 42, matched augmentation and deterministic request. A five-hour wall-time callback saves last.pt and stops at an epoch boundary. Checkpoints are saved every 10 epochs for manual continuation. Kaggle outputs must be collected before a fresh session.', f'''model = YOLO('yolov8{suffix}.pt')
start = time.monotonic()
def bounded_training(trainer):
    if time.monotonic() - start > 5 * 3600:
        trainer.stop = True
        print('Five-hour training guard reached; retain last.pt for continuation')
model.add_callback('on_train_epoch_end', bounded_training)
results = model.train(data=str(target / 'data.yaml'), epochs=60, patience=0,
    imgsz=640, batch=16, device=0, workers=2, seed=42, deterministic=True,
    optimizer='AdamW', lr0=0.001, weight_decay=0.0005,
    mosaic=1.0, close_mosaic=10, mixup=0.0, degrees=0.0, translate=0.1,
    scale=0.5, fliplr=0.5, flipud=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    project='/kaggle/working/runs', name={run_name!r}, exist_ok=False,
    save=True, save_period=10, plots=True, amp=True)
print(results.results_dict)'''),
        ('## 6. Show learning curves and selected-checkpoint evidence\nBest is selected by validation fitness. Save reported metrics and checkpoint hashes. These are source localization results on positive images, with no negative-image specificity estimate.', f'''run = Path(results.save_dir)
history = pd.read_csv(run / 'results.csv')
history.columns = history.columns.str.strip()
display(history.tail())
history.plot(x='epoch', y=['metrics/mAP50(B)', 'metrics/mAP50-95(B)'], figsize=(10,4), grid=True)
plt.title('Grouped validation localization'); plt.tight_layout(); plt.show()
best = run / 'weights/best.pt'
summary = dict(model='yolov8{suffix}', manifest_sha256={digest!r},
    completed_epochs=len(history), best_sha256=hashlib.sha256(best.read_bytes()).hexdigest(),
    metrics=results.results_dict)
(Path('/kaggle/working') / 'summary.json').write_text(json.dumps(summary, indent=2, default=str))
print(json.dumps(summary, indent=2, default=str))
for filename in ['results.png', 'val_batch0_pred.jpg']:
    path = run / filename
    if path.exists():
        display(Image.open(path))''')])
    name = f'14_9{suffix}_calculus_grouped_yolov8{suffix}.ipynb'
    nb.write(train, FOLDER / name)
    stage = ROOT / '.venv/work/kaggle' / run_name
    stage.mkdir(parents=True, exist_ok=True)
    nb.write(train, stage / name)
    meta = dict(id=f'trihuynhviprovcl/dentalselfie-grouped-v8{suffix}-20261117',
        title=f'Dentalselfie Grouped V8{suffix} 20261117', code_file=name,
        language='python', kernel_type='notebook', is_private=False,
        enable_gpu=True, enable_internet=True, machine_shape='NvidiaTeslaT4',
        dataset_sources=['singh868/multi-class-oral-disease-detection-dataset/1'],
        competition_sources=[], kernel_sources=[])
    (stage / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2))
print('Executed split audit and generated both controlled training notebooks.')


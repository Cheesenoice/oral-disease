"""Build technical teaching figures from verified train/validation assets.

Runs existing YOLO weights on ONE validation image. Never trains or opens test.
DINO outputs are collected v8 evidence, not fresh inference. Run with project venv.
"""
from pathlib import Path
import hashlib
import json
import os
import shutil

os.environ['YOLO_AUTOINSTALL'] = 'false'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageOps

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'report'
FIG = OUT / 'figure'
EVIDENCE = {'selection': 'First validation row in raw ROI manifest; first sorted caries validation ID. Not selected for prediction correctness.', 'test_images_opened': 0, 'inputs': {}, 'dino_execution': 'Archived full v8 outputs; no new encoder inference.'}


def source(path):
    path = ROOT / path
    EVIDENCE['inputs'][path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def checked_image(path, expected):
    p = source(path)
    assert hashlib.sha256(p.read_bytes()).hexdigest() == expected, p
    with Image.open(p) as image:
        return image.convert('RGB')


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / (name + '.png'), dpi=170, bbox_inches='tight')
    plt.close(fig)


def show(ax, im, title):
    ax.imshow(im)
    ax.set_title(title, fontsize=11)
    ax.axis('off')


def box(ax, bounds, color, label):
    x1, y1, x2, y2 = bounds
    ax.add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor=color, linewidth=2))
    ax.text(x1, y1, label, color='white', fontsize=9, backgroundcolor=color, va='bottom')


def main():
    FIG.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'figure.facecolor': 'white', 'axes.spines.top': False, 'axes.spines.right': False})
    raw = pd.read_csv(source('data/processed/gum_raw_roi_v1/manifest.csv'))
    paired = pd.read_csv(source('data/processed/gum_roi_v1/manifest.csv'))
    assert set(raw.split) == {'train', 'val'}
    r = raw.loc[raw.split.eq('val')].iloc[0]
    t = raw.loc[raw.split.eq('train')].iloc[0]
    old = paired.set_index('crop_id').loc[r.crop_id]
    assert old.split == r.split == 'val'
    EVIDENCE['gingival_demo'] = {'crop_id': r.crop_id, 'split': r.split, 'class_name': r.class_name, 'bounds': json.loads(r.raw_bounds)}
    parent = checked_image(r.source_image, r.source_sha256)
    crop = checked_image('data/processed/gum_raw_roi_v1/' + r.local_path, r.local_sha256)
    old_crop = checked_image('data/processed/gum_roi_v1/' + old.local_path, old.local_sha256)
    context = checked_image('data/processed/gum_roi_v1/' + old.context_path, old.context_sha256)
    prepared = checked_image('data/processed/gum/dataset/' + old.image_path, old.parent_sha256)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    show(axes[0], parent, f'Source image: {parent.width} x {parent.height}\nValidation; annotation box, not a prediction')
    box(axes[0], json.loads(r.raw_bounds), '#169c5d', 'Selected annotated ROI')
    show(axes[1], crop, 'Original-source crop -> 224 x 224\nSource label: mild inflammation')
    save(fig, 'technical_01_bbox_crop')

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    show(axes[0], prepared, 'Prepared parent: local and context bounds')
    box(axes[0], json.loads(old.context_bounds), '#b86b00', 'Context: 1.5x width / height')
    box(axes[0], json.loads(old.local_bounds), '#169c5d', 'Local')
    show(axes[1], old_crop, 'Recorded local view: 224 x 224')
    show(axes[2], context, 'Recorded context view: 224 x 224')
    save(fig, 'technical_02_context')

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    show(axes[0], old_crop, f'Old crop: box ~{old.box_width:.0f} x {old.box_height:.0f} px\nPrepared parent -> crop -> 224')
    show(axes[1], crop, f'Raw crop: box ~{r.raw_box_width:.0f} x {r.raw_box_height:.0f} px\nOriginal source -> crop -> 224')
    save(fig, 'technical_03_resolution')

    train = checked_image('data/processed/gum_raw_roi_v1/' + t.local_path, t.local_sha256)
    variants = [train, ImageOps.mirror(train), train.rotate(7, resample=Image.Resampling.BILINEAR, fillcolor=(128,128,128)), train.rotate(-7, resample=Image.Resampling.BILINEAR, fillcolor=(128,128,128)), ImageEnhance.Brightness(train).enhance(1.10), ImageEnhance.Contrast(train).enhance(.90)]
    labels = ['Unmodified training ROI', 'Horizontal flip', 'Rotate +7 degrees', 'Rotate -7 degrees', 'Brightness x1.10', 'Contrast x0.90']
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    for ax, im, label in zip(axes.flat, variants, labels):
        show(ax, im, label)
    fig.suptitle('Deterministic illustrations of training transforms; not extra validation cases')
    save(fig, 'technical_04_augmentation')

    pixels = np.asarray(train).astype(np.float32) / 255
    normalized = (pixels - np.array([.485,.456,.406])) / np.array([.229,.224,.225])
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    for i, name in enumerate(['Red', 'Green', 'Blue']):
        axes[i].imshow(pixels[:,:,i], cmap='gray', vmin=0, vmax=1)
        axes[i].set_title(name + ' channel [0, 1]'); axes[i].axis('off')
        axes[3].hist(normalized[:,:,i].ravel(), bins=50, histtype='step', color=['r','g','b'][i], label=name)
    axes[3].set_title('After ImageNet normalization'); axes[3].set_xlabel('Normalized value'); axes[3].legend()
    save(fig, 'technical_05_tensor')

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    show(axes[0], crop, '224 x 224 image = 14 x 14 patches')
    for v in range(0, 225, 16):
        axes[0].axvline(v-.5, color='cyan', linewidth=.5); axes[0].axhline(v-.5, color='cyan', linewidth=.5)
    box(axes[0], [96,96,112,112], '#ff7f0e', 'One patch')
    show(axes[1], crop.crop((96,96,112,112)).resize((224,224), Image.Resampling.NEAREST), 'One 16 x 16 RGB patch, enlarged\nLearned projection -> 384-dimensional token')
    save(fig, 'technical_06_patches')
    diagnostic = source('data/runs/ROI_11_1_full_v8/__results___files/__results___18_0.png')
    shutil.copyfile(diagnostic, FIG / 'technical_07_recorded_patch_norm.png')

    predictions = []
    paths = [('Frozen CLS / logistic', 'ROI_11_1_full_v8/dentalselfie-11-1/CLS_validation_predictions.csv'), ('CLS + patch / logistic', 'ROI_11_1_full_v8/dentalselfie-11-1/CLS_plus_patch_mean_validation_predictions.csv'), ('Last 2 blocks / CE', 'ROI_11_2_full_v8/dentalselfie-11-2/last2_blocks_norm_CE/validation_predictions.csv')]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    show(axes[0], crop, 'Same validation ROI\nSource label: mild inflammation')
    for j, (name, path) in enumerate(paths):
        frame = pd.read_csv(source('data/runs/' + path))
        id_col = 'crop_id' if 'crop_id' in frame else 'sample_id'
        row = frame.loc[frame[id_col].eq(r.crop_id)].iloc[0]
        pcols = [c for c in frame if c.startswith('prob_')]
        if not pcols:
            pcols = [c for c in frame if c.startswith('p') and c[-1:].isdigit()]
        values = np.asarray(json.loads(row['probabilities']), dtype=float) if 'probabilities' in frame else row[pcols].to_numpy(dtype=float)
        assert len(values) == 3, frame.columns
        assert abs(values.sum()-1) < 1e-5
        axes[1].bar(np.arange(3)+(j-1)*.25, values, width=.25, label=name)
        predictions.append({'model': name, 'probabilities': values.tolist()})
    axes[1].set_xticks(range(3), ['No visible', 'Mild', 'Marked']); axes[1].set_ylim(0,1)
    axes[1].set_ylabel('Model class probability (not clinical risk)'); axes[1].legend(fontsize=8)
    axes[1].set_title('Actual saved full-v8 predictions; not rerun here')
    EVIDENCE['gingival_demo']['recorded_predictions'] = predictions
    save(fig, 'technical_08_dino_predictions')

    fig, ax = plt.subplots(figsize=(13, 3.8))
    ax.set_xlim(-.5,13.5); ax.set_ylim(-.6,2); ax.axis('off')
    for i in range(12):
        color = '#e7a14b' if i >= 10 else '#99b6d1'
        ax.add_patch(Rectangle((i,.5),.85,.65,facecolor=color))
        ax.text(i+.425,.825,str(i),ha='center',va='center')
    ax.add_patch(Rectangle((12,.5),1.3,.65,facecolor='#e7a14b'))
    ax.text(12.65,.825,'Norm\n+ head',ha='center',va='center')
    ax.text(4.5,1.5,'Transformer blocks 0-9: frozen',ha='center')
    ax.text(11.3,1.5,'Blocks 10-11: adapted',ha='center')
    ax.text(6.5,0,'Frozen CE: 1,155 trainable parameters | Adapted: 3,551,619 / 21,597,699 total',ha='center')
    ax.text(6.5,-.35,'Encoder learning rate 1e-5; head learning rate 3e-4. This is supervised adaptation.',ha='center')
    save(fig, 'technical_09_adaptation')

    fig, axes = plt.subplots(1,2,figsize=(11,4))
    for arm in ['frozen_CE', 'last2_blocks_norm_CE']:
        history = pd.read_csv(source(f'data/runs/ROI_11_2_full_v8/dentalselfie-11-2/{arm}/history.csv'))
        axes[0].plot(history.epoch,history.train_loss,label=arm)
        axes[1].plot(history.epoch,history.macro_f1*100,label=arm)
    for ax in axes:
        ax.set_xlabel('Epoch'); ax.legend(fontsize=8); ax.grid(alpha=.2)
    axes[0].set_title('Actual training cross-entropy'); axes[1].set_title('Actual validation macro-F1 (%)')
    save(fig,'technical_10_learning_curves')

    lock=json.loads(source('data/manifests/caries.lock.json').read_text())
    split_path=source('data/manifests/caries.splits.csv')
    assert hashlib.sha256(split_path.read_bytes()).hexdigest()==lock['splits_sha256']
    splits=pd.read_csv(split_path)
    c=splits.loc[splits.split.eq('val')].sort_values('sample_id').iloc[0]
    caries=checked_image(c.image,c.sha256)
    label_path=source(c.label)
    assert hashlib.sha256(label_path.read_bytes()).hexdigest()==c.label_sha256
    from ultralytics import YOLO
    import ultralytics
    fig,axes=plt.subplots(1,3,figsize=(16,5))
    show(axes[0],caries,'Validation source annotation')
    for line in label_path.read_text().splitlines():
        cls,cx,cy,w,h=map(float,line.split())
        bounds=[(cx-w/2)*caries.width,(cy-h/2)*caries.height,(cx+w/2)*caries.width,(cy+h/2)*caries.height]
        box(axes[0],bounds,'#169c5d','Source caries box')
    EVIDENCE['fresh_yolo']={'sample_id':c.sample_id,'split':'val','ultralytics':ultralytics.__version__,'settings':{'device':'cpu','imgsz':640,'conf':.25,'iou':.7,'max_det':300,'augment':False},'models':[]}
    for ax,name,run in zip(axes[1:],['YOLOv8n','YOLO26n'],['E0','E1']):
        receipt=json.loads(source(f'docs/reports/post09/caries/{name}_clean/audit_receipt.json').read_text())
        weight=source(f'data/runs/{run}_full_v4/experiments/{run}_caries_seed42/weights/best.pt')
        assert hashlib.sha256(weight.read_bytes()).hexdigest()==receipt['identity']['weight_sha256']
        model=YOLO(str(weight))
        result=model.predict(caries,**EVIDENCE['fresh_yolo']['settings'],verbose=False)[0]
        show(ax,caries,name+' fresh CPU inference')
        boxes=[]
        for xyxy,conf in zip(result.boxes.xyxy.cpu().tolist(),result.boxes.conf.cpu().tolist()):
            box(ax,xyxy,'#1869ba',f'caries {conf:.2f}')
            boxes.append({'xyxy':xyxy,'confidence':conf})
        EVIDENCE['fresh_yolo']['models'].append({'model':name,'boxes':boxes,'speed_ms':result.speed})
    save(fig,'technical_11_fresh_yolo')
    (OUT/'technical_demo_evidence.json').write_text(json.dumps(EVIDENCE,indent=2)+'\n',encoding='utf-8')
    print('Built 11 technical figures and verified demo evidence. Test images opened: 0.')


if __name__ == '__main__':
    main()

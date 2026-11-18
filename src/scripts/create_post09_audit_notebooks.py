"""Build teaching notebooks; actual evidence is produced by notebook execution."""
from pathlib import Path
import textwrap
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]


def build(name, title, sections):
    path = ROOT / "notebooks" / name
    if path.exists():
        raise FileExistsError(path)
    flow = ('flowchart LR\n    A[Locked validation evidence] --> B[Verify identities and processing]\n'
            '    B --> C[Recompute metrics and inspect errors]\n    C --> D[Save evidence and decide next experiment]')
    cells = [nbf.v4.new_markdown_cell(title + '\n\n```mermaid\n' + flow + '\n```')]
    for markdown, code in sections:
        cells += [nbf.v4.new_markdown_cell(textwrap.dedent(markdown).strip()),
                  nbf.v4.new_code_cell(textwrap.dedent(code).strip())]
    notebook = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {
        "display_name": "DentalSelfie (Python 3.11)", "language": "python", "name": "dentalselfie-local"},
        "language_info": {"name": "python", "version": "3.11"}})
    nbf.validate(notebook)
    nbf.write(notebook, path)
    print("Created", path.name, len(cells), "cells")


COMMON = '''
from pathlib import Path
import hashlib, json, platform, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
from IPython.display import display, Markdown
ROOT = Path.cwd().resolve()
if not (ROOT / 'pyproject.toml').exists():
    ROOT = ROOT.parent
assert (ROOT / 'src/dentalselfie').exists(), 'Run from project root or notebooks directory'
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding='utf-8')
plt.rcParams.update({'figure.dpi': 110, 'axes.spines.top': False, 'axes.spines.right': False})
print('Root:', ROOT, '| Python:', platform.python_version())
'''


def gingival():
    sections = [
    ("## 1. Scope and reproducibility\nUse existing validation predictions. Test pixels remain unopened; clinical labels remain unchanged.", COMMON + '''
OUT = ROOT / 'docs/reports/post09/gingival'
OUT.mkdir(parents=True, exist_ok=True)
RAW = ROOT / 'data/processed/gum_raw_roi_v1'
OLD = ROOT / 'data/processed/gum_roi_v1'
raw_receipt = json.loads((RAW / 'receipt.json').read_text())
assert sha(RAW / 'manifest.csv') == raw_receipt['manifest_sha256']
assert sha(OLD / 'manifest.csv') == raw_receipt['signature']['roi_v1_manifest_sha256']
manifest = pd.read_csv(RAW / 'manifest.csv')
old = pd.read_csv(OLD / 'manifest.csv')
keys = ['crop_id','parent_id','group_id','split','class_id']
pd.testing.assert_frame_equal(manifest[keys].sort_values('crop_id').reset_index(drop=True),
                             old[keys].sort_values('crop_id').reset_index(drop=True))
for task in ['gum', 'caries']:
    lock = json.loads((ROOT / f'data/manifests/{task}.lock.json').read_text())
    assert sha(ROOT / f'data/manifests/{task}.splits.csv') == lock['splits_sha256']
val = manifest[manifest.split.eq('val')].copy().set_index('crop_id')
assert len(val) == 1920 and manifest.groupby('group_id').split.nunique().max() == 1
print('Verified manifest identities and both split locks; validation crops:', len(val))
'''),
    ("## 2. Original label semantics and independent support\nClasses are source-defined appearance groups. Jaw boxes 0/1 are excluded. Group counts describe observable source constraints, not verified patients. Count source annotation IDs without opening test files.", '''
mapping = json.loads((ROOT / 'src/configs/label_mappings.json').read_text())['gum']
assert mapping['classes'] == {'0':None,'1':None,'2':0,'3':1,'4':1,'5':2,'6':2}
display(pd.DataFrame([{'source_id': k, 'mapped_class': v} for k,v in mapping['classes'].items()]))
display(Markdown(mapping['evidence']))
support = manifest.groupby(['split','class_id','class_name']).agg(
    crops=('crop_id','size'), parents=('parent_id','nunique'), groups=('group_id','nunique')).reset_index()
support.to_csv(OUT / 'class_group_support.csv', index=False)
display(support)
parents = pd.read_csv(ROOT / 'data/manifests/gum.splits.csv').set_index('sample_id')
source_ids = {}
for pid, group in manifest.groupby('parent_id'):
    parent = parents.loc[pid]
    label_path = ROOT / parent.source_label
    assert parent.split in ['train','val'] and sha(label_path) == parent.source_label_sha256
    lines = label_path.read_text().splitlines()
    for row in group.itertuples():
        source_id = lines[int(row.source_annotation_line)].split()[0]
        assert mapping['classes'][source_id] == row.class_id
        source_ids[row.crop_id] = source_id
manifest['original_source_class'] = manifest.crop_id.map(source_ids)
display(pd.crosstab([manifest.split, manifest.original_source_class], manifest.class_name))
fig, ax = plt.subplots(figsize=(7,3))
for split, group in support.groupby('split'):
    ax.plot(group.class_id, group.crops, marker='o', label=split)
ax.set(xticks=[0,1,2], xlabel='Appearance class', ylabel='ROI count', title='Class imbalance before any model')
ax.legend(); fig.tight_layout(); fig.savefig(OUT/'class_support.png'); plt.show()
'''),
    ("## 3. Recompute every available full-run prediction table\nExclude smoke runs and the invalid scratch initialization version. Some 6.x artifacts contain only aggregate metrics: report that limitation rather than invent per-crop predictions. All comparisons here remain exploratory.", '''
def matrix_metrics(cm):
    recall = np.divide(cm.diagonal(), cm.sum(1), out=np.zeros(3), where=cm.sum(1)>0)
    precision = np.divide(cm.diagonal(), cm.sum(0), out=np.zeros(3), where=cm.sum(0)>0)
    f1 = np.divide(2*precision*recall, precision+recall, out=np.zeros(3), where=(precision+recall)>0)
    return {'accuracy': float(cm.trace()/cm.sum()), 'macro_f1': float(f1.mean()),
            'balanced_accuracy': float(recall.mean())}
prediction_paths = []
for run in ['ROI_07_1_v1','ROI_07_2_v1','ROI_09_1_v1','ROI_09_2_v1','ROI_09_3_v3']:
    prediction_paths += [p for p in (ROOT/'data/runs'/run).rglob('validation_predictions.csv') if 'smoke' not in p.parts]
joined = val.copy()
rows, evidence = [], []
for path in sorted(prediction_paths):
    pred = pd.read_csv(path)
    assert pred.crop_id.is_unique and set(pred.crop_id) == set(val.index)
    pred = pred.set_index('crop_id').loc[val.index]
    assert np.array_equal(pred.true_class, val.class_id)
    assert pred.predicted_class.isin([0,1,2]).all()
    metrics = json.loads(path.with_name('metrics.json').read_text())
    cm = np.zeros((3,3), dtype=int)
    np.add.at(cm, (pred.true_class.to_numpy(), pred.predicted_class.to_numpy()), 1)
    assert cm.tolist() == metrics['confusion_matrix']
    measured = matrix_metrics(cm)
    for k, v in measured.items(): assert np.isclose(v, metrics[k]), (path,k)
    probabilities = np.array(pred.probabilities.map(json.loads).tolist())
    ordinal = bool(metrics.get('ordinal', False))
    if ordinal:
        assert probabilities.shape[1] == 2 and (probabilities[:,0] >= probabilities[:,1]).all()
        decoded = (probabilities > .5).sum(1)
    else:
        assert np.allclose(probabilities.sum(1), 1, atol=.002)
        decoded = probabilities.argmax(1)
    assert np.array_equal(decoded, pred.predicted_class)
    history_path = path.with_name('history.csv')
    if history_path.exists():
        history = pd.read_csv(history_path)
        best = history.loc[history.macro_f1.idxmax()]
        assert int(best.epoch) == int(metrics['selected_epoch'])
        assert np.isclose(best.macro_f1, metrics['macro_f1'])
    name = path.parent.name
    joined['pred_'+name] = pred.predicted_class
    rows.append({'experiment':name, **measured, 'no_visible_recall':cm[0,0]/cm[0].sum()})
    evidence.append({'experiment':name,'predictions_sha256':sha(path),'metrics_sha256':sha(path.with_name('metrics.json'))})
results = pd.DataFrame(rows).sort_values('macro_f1', ascending=False)
results.to_csv(OUT/'verified_model_metrics.csv',index=False)
display(results.style.format({k:'{:.2%}' for k in ['accuracy','macro_f1','balanced_accuracy','no_visible_recall']}))
print('Verified per-crop runs:',len(results),'| Historical 6.x aggregate-only results remain in their original report.')
'''),
    ("## 4. Shared failures versus model disagreement\nAgreement does not establish label correctness. These are correlated models trained on the same source. Identify hard examples without relabelling them.", '''
pred_cols = [c for c in joined if c.startswith('pred_')]
pred_array = joined[pred_cols].to_numpy()
joined['wrong_count'] = (pred_array != joined.class_id.to_numpy()[:,None]).sum(1)
joined['all_wrong'] = joined.wrong_count.eq(len(pred_cols))
joined['all_correct'] = joined.wrong_count.eq(0)
joined['unanimous'] = np.array([len(set(row)) == 1 for row in pred_array])
joined['unanimous_wrong'] = joined.unanimous & joined.all_wrong
display(joined.groupby('class_name').agg(crops=('class_id','size'),all_wrong=('all_wrong','sum'),
                                       unanimous_wrong=('unanimous_wrong','sum'),all_correct=('all_correct','sum')))
error_matrix = (pred_array != joined.class_id.to_numpy()[:,None]).T.astype(float)
intersection = error_matrix @ error_matrix.T
union = error_matrix.sum(1)[:,None] + error_matrix.sum(1)[None,:] - intersection
jaccard = np.divide(intersection, union, out=np.zeros_like(intersection), where=union>0)
fig,ax=plt.subplots(figsize=(9,7)); im=ax.imshow(jaccard,vmin=0,vmax=1,cmap='magma')
labels=[c.removeprefix('pred_') for c in pred_cols]
ax.set(xticks=range(len(labels)),yticks=range(len(labels)),xticklabels=labels,yticklabels=labels,title='Jaccard overlap of error sets')
plt.setp(ax.get_xticklabels(),rotation=65,ha='right'); fig.colorbar(im,ax=ax); fig.tight_layout()
fig.savefig(OUT/'error_overlap.png'); plt.show()
'''),
    ("## 5. Verify validation crop bytes and measure quality proxies\nLaplacian variance and grayscale brightness are descriptive image statistics, not clinical quality judgments. Fixed bins are reported on the complete validation set; no image is excluded.", '''
import cv2
quality = []
for crop_id, row in val.iterrows():
    path = RAW / row.local_path
    assert sha(path) == row.local_sha256
    with Image.open(path) as image:
        assert image.size == (224,224) and image.mode == 'RGB'
        gray = np.asarray(image.convert('L'),dtype=np.uint8)
    quality.append({'crop_id':crop_id,'brightness':float(gray.mean()),
                    'laplacian_variance':float(cv2.Laplacian(gray,cv2.CV_64F).var())})
joined = joined.join(pd.DataFrame(quality).set_index('crop_id'))
joined['source_min_edge'] = joined[['raw_box_width','raw_box_height']].min(axis=1)
joined['size_bin'] = pd.cut(joined.source_min_edge, [0,224,448,np.inf], labels=['<=224','225-448','>448'])
joined['brightness_bin'] = pd.cut(joined.brightness, [-1,64,128,192,256], labels=['0-64','65-128','129-192','193-255'])
joined['sharpness_bin'] = pd.cut(joined.laplacian_variance, [-1,100,500,np.inf], labels=['<=100','101-500','>500'])
strata=[]
for variable in ['size_bin','brightness_bin','sharpness_bin']:
    for name, subset in joined.groupby(variable,observed=True):
        cm=np.zeros((3,3),int)
        np.add.at(cm,(subset.class_id,subset.pred_R1_raw_CE),1)
        strata.append({'stratum':variable,'bin':str(name),'crops':len(subset),
                       'source_groups':subset.group_id.nunique(),'class_support':cm.sum(1).tolist(),
                       **matrix_metrics(cm)})
strata=pd.DataFrame(strata); strata.to_csv(OUT/'quality_strata.csv',index=False); display(strata)
print('Macro-F1 uses the fixed three classes; absent classes in small strata receive zero F1.')
'''),
    ("## 6. Blinded clinical review sheet and representative galleries\nInclude every no-visible ROI, a fixed random sample of the other classes, and a separately identified hard-case sample. The blinded sheet omits model predictions; technical statistics remain in a separate file. Empty review fields are intentional and require a qualified reviewer.", '''
selected = joined[joined.class_id.eq(0)].copy()
selected['selection_reason']='all_no_visible'
for cls in [1,2]:
    sample=joined[joined.class_id.eq(cls)].sample(n=24,random_state=42).copy()
    sample['selection_reason']='seeded_random_class_'+str(cls)
    selected=pd.concat([selected,sample])
hard=joined.sort_values(['wrong_count'],ascending=False,kind='stable').head(24).copy()
hard['selection_reason']='hard_example_nonrandom'
selected=pd.concat([selected,hard]).reset_index().drop_duplicates('crop_id').set_index('crop_id')
review=selected[['parent_id','class_id','class_name','source_image','local_path','selection_reason']].copy()
for field in ['reviewer_id','reviewed_appearance','confidence','visibility_issue','comment']:
    review[field]=''
review.to_csv(OUT/'reviewer_sheet.csv')
joined.to_csv(OUT/'all_validation_diagnostics.csv')
for page, start in enumerate(range(0,len(selected),16),1):
    subset=selected.iloc[start:start+16]
    fig,axes=plt.subplots(4,4,figsize=(12,12))
    for ax in axes.flat: ax.axis('off')
    for ax,(crop_id,row) in zip(axes.flat,subset.iterrows()):
        with Image.open(RAW/row.local_path) as image: ax.imshow(image)
        ax.set_title(f'{crop_id}\\nsource label={row.class_id}; {row.selection_reason}',fontsize=7)
    fig.suptitle('Review gallery: original labels, no automatic relabelling');fig.tight_layout()
    fig.savefig(OUT/f'review_page_{page:02d}.png');plt.show()
print('Review sheet rows:',len(review),'| All no-visible included:',int((review.class_id==0).sum()))
'''),
    ("## 7. Source-parent and crop geometry panels\nShow the full source photograph with the selected ROI rectangle, followed by its raw-source crop. These panels support technical inspection; box coverage is not a tissue segmentation mask.", '''
representatives=selected.reset_index().groupby('class_id',group_keys=False).head(2)
fig,axes=plt.subplots(len(representatives),2,figsize=(12,3*len(representatives)))
for index,row in enumerate(representatives.itertuples()):
    path=ROOT/row.source_image
    assert sha(path)==row.source_sha256
    with Image.open(path) as image:
        original_size=image.size
        preview=image.convert('RGB');preview.thumbnail((1200,800))
        bounds=json.loads(row.raw_bounds)
        scale=np.array([preview.width/original_size[0],preview.height/original_size[1]]*2)
        ImageDraw.Draw(preview).rectangle(tuple(np.array(bounds)*scale),outline='red',width=4)
        axes[index,0].imshow(preview)
    with Image.open(RAW/row.local_path) as image: axes[index,1].imshow(image)
    axes[index,0].set_title('Source photograph with audited box')
    axes[index,1].set_title(f'{row.crop_id}: source class {row.class_id}')
    for ax in axes[index]:ax.axis('off')
fig.tight_layout();fig.savefig(OUT/'parent_crop_geometry.png');plt.show()
'''),
    ("## 8. Technical verdict and next experiment gate\nPrediction joins, probabilities, checkpoint selection, source-label mapping, crop bytes and split locks are checked above. This does not certify clinical label validity, patient independence, or that every original-source crop contains sufficient gingival tissue.", '''
summary={'validation_crops':len(joined),'validation_groups':joined.group_id.nunique(),
         'verified_prediction_runs':len(results),'all_models_wrong':int(joined.all_wrong.sum()),
         'unanimous_wrong':int(joined.unanimous_wrong.sum()),
         'no_visible_validation_groups':joined[joined.class_id.eq(0)].group_id.nunique(),
         'technical_checks':'passed for the explicitly executed assertions',
         'clinical_review':'pending; no labels changed','test_images_opened':0,
         'raw_manifest_sha256':sha(RAW/'manifest.csv'),'prediction_evidence':evidence}
save_json(OUT/'summary.json',summary)
display(pd.Series({k:v for k,v in summary.items() if k!='prediction_evidence'}).to_frame('value'))
display(Markdown('**Decision:** no failed technical assertion in this audit. Clinical ambiguity remains unverified. '
                 'A bounded patch-readout/adaptation study is eligible for later implementation; '
                 'do not claim a data-label ceiling or a solved clinical task from model disagreement.'))
''')]
    build('10_1_gingival_failure_audit.ipynb', '# 10.1 Gingival failure audit\n\nEvidence-first diagnosis of the completed model comparisons. No training and no test evaluation.', sections)


def caries():
    helper=(ROOT/'src/dentalselfie/detection_audit.py').read_text()
    sections=[
    ("## 1. Runtime, scope and fixed protocol\nReuse existing weights. Evaluate full validation photographs at 640 input. CPU timing is a desktop measurement, not phone performance. Corruptions are fixed synthetic stress tests, not real phone-photo validation.",COMMON+'''
import os, io, shutil, yaml, torch, ultralytics
os.environ['YOLO_AUTOINSTALL']='false'
from ultralytics import YOLO
torch.set_num_threads(min(8,os.cpu_count() or 1))
OUT=ROOT/'docs/reports/post09/caries';OUT.mkdir(parents=True,exist_ok=True)
WORK=ROOT/'.venv/work/audit10/caries';WORK.mkdir(parents=True,exist_ok=True)
assert ultralytics.__version__=='8.4.183'
protocol={'imgsz':640,'batch':8,'workers':0,'device':'cpu','conf':0.001,'iou':0.7,
          'max_det':300,'half':False,'augment':False,'split':'val',
          'diagnostic_confidence':0.25,'diagnostic_iou':0.5,
          'variants':['clean','blur_r1_5','dark_0_65','jpeg_q35']}
display(pd.Series(protocol).to_frame('fixed setting'))
print('Torch',torch.__version__,'Ultralytics',ultralytics.__version__)
'''),
    ("## 2. Lock validation membership and verify input bytes\nRead the split table, then open only validation image/label paths. Missing annotations remain an uncertainty, not healthy negatives.",'''
lock=json.loads((ROOT/'data/manifests/caries.lock.json').read_text())
assert sha(ROOT/'data/manifests/caries.splits.csv')==lock['splits_sha256']
manifest=pd.read_csv(ROOT/'data/manifests/caries.splits.csv')
val=manifest[manifest.split.eq('val')].copy()
assert len(val)==327 and val.sample_id.is_unique
gt={}; labels=[]
for row in val.itertuples():
    assert sha(ROOT/row.image)==row.sha256 and sha(ROOT/row.label)==row.label_sha256
    boxes=[]
    for line in (ROOT/row.label).read_text().splitlines():
        cls,cx,cy,w,h=map(float,line.split());assert cls==0 and w>0 and h>0
        box=[(cx-w/2)*row.width,(cy-h/2)*row.height,(cx+w/2)*row.width,(cy+h/2)*row.height]
        boxes.append(box)
        area=(w*row.width)*(h*row.height)*(640/max(row.width,row.height))**2
        labels.append({'sample_id':row.sample_id,'gt_index':len(boxes)-1,'area_640':area,
                       'size':'small' if area<32**2 else ('medium' if area<96**2 else 'large')})
    gt[row.sample_id]=boxes
lesions=pd.DataFrame(labels);lesions.to_csv(OUT/'lesion_size_membership.csv',index=False)
display(val.groupby(['source_collection','view']).agg(images=('sample_id','size'),boxes=('label_count','sum')))
display(lesions.groupby('size').size().rename('lesions').to_frame())
print('Size bins use box area after isotropic scaling to maximum edge 640: <32^2, <96^2, otherwise large.')
'''),
    ("## 3. Deterministic complete-image stress inputs\nTransform the full photograph before model resizing. Keep box coordinates and image dimensions fixed. JPEG q35 uses one re-encode from the prepared image; blur/dark variants use high-quality PNG to avoid extra JPEG confounding.",'''
variant_paths={}
for variant in protocol['variants']:
    paths=[]
    for row in val.itertuples():
        if variant=='clean':
            path=ROOT/row.image
        else:
            folder=WORK/variant
            extension='.jpg' if variant=='jpeg_q35' else '.png'
            path=folder/'images'/(row.sample_id+extension)
            path.parent.mkdir(parents=True,exist_ok=True)
            label=folder/'labels'/(row.sample_id+'.txt');label.parent.mkdir(parents=True,exist_ok=True)
            if not path.exists():
                with Image.open(ROOT/row.image) as opened:
                    image=opened.convert('RGB')
                    if variant=='blur_r1_5':image=image.filter(ImageFilter.GaussianBlur(1.5))
                    if variant=='dark_0_65':image=ImageEnhance.Brightness(image).enhance(.65)
                    if variant=='jpeg_q35':image.save(path,quality=35,subsampling=2)
                    else:image.save(path)
            shutil.copyfile(ROOT/row.label,label)
            with Image.open(path) as image:assert image.size==(row.width,row.height)
        paths.append(str(path.resolve()))
    listing=WORK/(variant+'.txt');listing.write_text('\\n'.join(paths)+'\\n',encoding='utf-8')
    config=WORK/(variant+'.yaml')
    config.write_text(yaml.safe_dump({'path':str(ROOT),'train':str(listing),'val':str(listing),'names':{0:'visible_caries'}}))
    variant_paths[variant]=config
    print('Prepared',variant,len(paths),'full images',flush=True)
fig,axes=plt.subplots(1,4,figsize=(15,4))
for ax,(variant,config) in zip(axes,variant_paths.items()):
    path=Path((WORK/(variant+'.txt')).read_text().splitlines()[0])
    with Image.open(path) as image:ax.imshow(image)
    ax.set_title(variant);ax.axis('off')
fig.tight_layout();fig.savefig(OUT/'stress_examples.png');plt.show()
print('Prepared all four fixed validation variants:',len(val),'images each')
'''),
    ("## 4. Load existing checkpoints and evaluate all full images\nThe detector's native postprocessing is retained: YOLO26 end-to-end behavior is not forced into YOLOv8 NMS. Save confidence curves, JSON detections and model-native AP metrics. Reuse a completed audit only if its checkpoint/input/protocol/runtime fingerprint matches.",'''
weights={'YOLOv8n':ROOT/'data/runs/E0_full_v4/experiments/E0_caries_seed42/weights/best.pt',
         'YOLO26n':ROOT/'data/runs/E1_full_v4/experiments/E1_caries_seed42/weights/best.pt'}
summaries=[]
for name,weight in weights.items():
    assert weight.exists()
    model=YOLO(str(weight))
    print('Loaded',name,'parameters:',sum(p.numel() for p in model.model.parameters()),flush=True)
    for variant,config in variant_paths.items():
        run=OUT/(name+'_'+variant);receipt=run/'audit_receipt.json'
        identity={'weight_sha256':sha(weight),'splits_sha256':lock['splits_sha256'],
                  'input_files':[{ 'id':r.sample_id,'image_sha256':sha(Path(p)),'label_sha256':r.label_sha256}
                                 for r,p in zip(val.itertuples(),(WORK/(variant+'.txt')).read_text().splitlines())],
                  'protocol':protocol,'variant':variant,'torch':torch.__version__,
                  'ultralytics':ultralytics.__version__}
        fingerprint=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        if receipt.exists():
            result=json.loads(receipt.read_text());assert result['fingerprint']==fingerprint
            assert sha(run/'predictions.json')==result['predictions_sha256']
            print('Reused verified audit',name,variant,flush=True)
        else:
            start=time.perf_counter()
            metrics=model.val(data=str(config),imgsz=640,batch=8,workers=0,device='cpu',
                              conf=.001,iou=.7,max_det=300,half=False,augment=False,split='val',
                              save_json=True,plots=True,project=str(OUT),name=run.name,exist_ok=True,verbose=False)
            result={'model':name,'variant':variant,'fingerprint':fingerprint,
                    'map50':float(metrics.box.map50),'map50_95':float(metrics.box.map),
                    'precision_native_curve':float(metrics.box.mp),'recall_native_curve':float(metrics.box.mr),
                    'speed_ms_per_image':{k:float(v) for k,v in metrics.speed.items()},
                    'wall_seconds':time.perf_counter()-start,'test_images_opened':0,
                    'predictions_sha256':sha(run/'predictions.json'),'identity':identity}
            save_json(receipt,result)
        summaries.append({k:v for k,v in result.items() if k not in ['identity','speed_ms_per_image']})
        print(name,variant,'mAP50:',result['map50'],'mAP50-95:',result['map50_95'],flush=True)
summary=pd.DataFrame(summaries);summary.to_csv(OUT/'native_detection_metrics.csv',index=False)
display(summary[['model','variant','map50','map50_95','wall_seconds']])
'''),
    ("## 5. Auditable one-to-one matching implementation\nAt confidence >=0.25 and IoU >=0.5, match predictions in descending score order to unmatched GT boxes. Count duplicates separately from localization errors. An unmatched prediction is an apparent annotation-relative false positive; it may need annotation review.",helper.replace('from __future__ import annotations\n','')+"\nprint('Matching helpers defined; fixed confidence=0.25, IoU=0.5')"),
    ("## 6. Full-image failure decomposition and lesion-size recall\nNative AP above integrates confidence thresholds. The following fixed operating-point diagnostics are different quantities. Misses are retained, including images with no returned predictions. Size-stratified recall is not size-stratified AP.",'''
image_rows=[];event_rows=[];lesion_rows=[];prediction_cache={}
# Verified in pinned DetectionValidator.init_metrics: non-COCO JSON IDs are
# range(1, len(model.names)+1). Exported category 1 is model class 0.
from ultralytics.models.yolo.detect.val import DetectionValidator
import inspect
assert 'list(range(1, len(model.names) + 1))' in inspect.getsource(DetectionValidator.init_metrics)
exported_categories={1:0}
print('Verified native JSON mapping:',exported_categories,'(exported category -> model class)')
for name in weights:
    for variant in protocol['variants']:
        detections=json.loads((OUT/(name+'_'+variant)/'predictions.json').read_text())
        by_image={sid:[] for sid in val.sample_id}
        for det in detections:
            sid=str(det['image_id']);assert sid in by_image
            assert exported_categories[int(det['category_id'])]==0
            if det['score']>=.25:by_image[sid].append(det)
        prediction_cache[(name,variant)]=by_image
        for row in val.itertuples():
            predictions=by_image[row.sample_id]
            boxes=[[d['bbox'][0],d['bbox'][1],d['bbox'][0]+d['bbox'][2],d['bbox'][1]+d['bbox'][3]] for d in predictions]
            events,missed=match_boxes(gt[row.sample_id],boxes,[d['score'] for d in predictions])
            counts={kind:sum(e['kind']==kind for e in events) for kind in ['true_positive','duplicate','localization','unmatched']}
            image_rows.append({'model':name,'variant':variant,'sample_id':row.sample_id,
                               'ground_truth':len(gt[row.sample_id]),'missed':len(missed),**counts})
            for event in events:event_rows.append({'model':name,'variant':variant,'sample_id':row.sample_id,**event})
            for j in range(len(gt[row.sample_id])):
                lesion_rows.append({'model':name,'variant':variant,'sample_id':row.sample_id,'gt_index':j,'detected':j not in missed})
images=pd.DataFrame(image_rows);events=pd.DataFrame(event_rows)
lesion_results=pd.DataFrame(lesion_rows).merge(lesions,on=['sample_id','gt_index'],validate='many_to_one')
images.to_csv(OUT/'per_image_errors.csv',index=False);events.to_csv(OUT/'prediction_events.csv',index=False)
lesion_results.to_csv(OUT/'per_lesion_recall.csv',index=False)
totals=images.groupby(['model','variant'])[['ground_truth','true_positive','missed','duplicate','localization','unmatched']].sum()
totals['precision_at_025']=totals.true_positive/(totals.true_positive+totals.duplicate+totals.localization+totals.unmatched)
totals['recall_at_025']=totals.true_positive/totals.ground_truth
display(totals)
size_recall=lesion_results.groupby(['model','variant','size']).agg(lesions=('detected','size'),recall=('detected','mean'))
display(size_recall);size_recall.to_csv(OUT/'size_recall.csv')
'''),
    ("## 7. Error galleries on complete validation photographs\nChoose the highest missed/unmatched counts for inspection, rather than claiming these are a random sample. Green boxes are GT, red boxes are detector outputs at the fixed confidence threshold. Source annotation completeness remains unverified.",'''
for name in weights:
    failures=images[(images.model==name)&images.variant.eq('clean')].copy()
    failures['error_count']=failures[['missed','duplicate','localization','unmatched']].sum(axis=1)
    selected=failures.sort_values(['error_count','sample_id'],ascending=[False,True]).head(8)
    fig,axes=plt.subplots(4,2,figsize=(14,16))
    for ax,item in zip(axes.flat,selected.itertuples()):
        row=val.set_index('sample_id').loc[item.sample_id]
        with Image.open(ROOT/row.image) as image:
            image=image.convert('RGB');draw=ImageDraw.Draw(image)
            for box in gt[item.sample_id]:draw.rectangle(tuple(box),outline='lime',width=4)
            for d in prediction_cache[(name,'clean')][item.sample_id]:
                x,y,w,h=d['bbox'];draw.rectangle((x,y,x+w,y+h),outline='red',width=3)
            image.thumbnail((1000,700));ax.imshow(image)
        ax.set_title(f'{name}: {item.sample_id}\\nmissed={item.missed}; duplicate={item.duplicate}; localization={item.localization}; unmatched={item.unmatched}',fontsize=8);ax.axis('off')
    fig.tight_layout();fig.savefig(OUT/(name+'_full_image_errors.png'));plt.show()
print('Green = source annotation; red = prediction. These are annotation-relative error examples.')
'''),
    ("## 8. Robustness, timing and evidence-based scope\nCompare fixed synthetic perturbations against the same model's clean performance. Curves and raw detections are saved in each native validation directory. No phone accuracy or clinical usability is established here.",'''
fig,axes=plt.subplots(1,2,figsize=(12,4))
for name,group in summary.groupby('model'):
    for ax,metric in zip(axes,['map50','map50_95']):
        ax.plot(group.variant,group[metric],marker='o',label=name);ax.set(ylabel=metric,ylim=(0,1))
        ax.tick_params(axis='x',rotation=25);ax.legend()
fig.tight_layout();fig.savefig(OUT/'robustness_comparison.png');plt.show()
runtime=[]
for name in weights:
    item=json.loads((OUT/(name+'_clean')/'audit_receipt.json').read_text())
    runtime.append({'model':name,**item['speed_ms_per_image'],'wall_seconds':item['wall_seconds']})
display(pd.DataFrame(runtime));print('Timing includes validator batch/runtime behavior; not per-request phone latency.')
save_json(OUT/'summary.json',{'protocol':protocol,'validation_images':len(val),'validation_lesions':len(lesions),
                            'test_images_opened':0,'split_sha256':lock['splits_sha256'],
                            'models':list(weights),'runs':len(summary),
                            'limitations':['source-domain validation only','observable groups are not verified patients',
                                           'synthetic corruptions are not external phone images',
                                           'unmatched detections require annotation review']})
display(Markdown('**Next decision:** use full-image errors and corruption gaps to specify a realistic capture feasibility check. '
                 'Keep personal photographs local and report no accuracy without independent reference labels.'))
''')]
    build('10_2_caries_full_image_audit.ipynb','# 10.2 Caries full-image audit\n\nExisting YOLO checkpoints, locked validation images, full-image diagnostics and fixed robustness tests. No training.',sections)


if __name__=='__main__':
    gingival()
    caries()

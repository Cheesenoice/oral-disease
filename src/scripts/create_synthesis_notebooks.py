"""Build local evidence synthesis and private photo-feasibility teaching notebooks."""
from pathlib import Path
import textwrap
import nbformat as nb

ROOT = Path(__file__).resolve().parents[2]


def save(name, title, sections, private=False):
    path = ROOT / "notebooks" / name
    if path.exists():
        raise FileExistsError(f"Preserve existing notebook: {path}")
    cells = [nb.v4.new_markdown_cell(title)]
    for markdown, code in sections:
        cells += [nb.v4.new_markdown_cell(textwrap.dedent(markdown).strip()),
                  nb.v4.new_code_cell(textwrap.dedent(code).strip())]
    book = nb.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "DentalSelfie (Python 3.11)", "language": "python", "name": "dentalselfie-local"},
        "language_info": {"name": "python", "version": "3.11"},
        "dentalselfie": {"local_only": private, "stage": "evidence_and_capture_feasibility"}})
    nb.validate(book)
    nb.write(book, path)
    print("Created", name, len(cells), "cells")


COMMON = '''
from pathlib import Path
import hashlib, json, platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import display, Markdown, Code
ROOT = Path.cwd().resolve()
if not (ROOT/'pyproject.toml').exists(): ROOT = ROOT.parent
assert (ROOT/'src/dentalselfie').is_dir(), 'Run from the project root or notebooks directory'
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
plt.rcParams.update({'figure.dpi':110, 'axes.spines.top':False, 'axes.spines.right':False})
print('Root:', ROOT, '| Python:', platform.python_version(), '| No training or test evaluation')
'''


def synthesis():
    sections = [
    ('''## 1. Scope and provenance
    Consolidate completed experiments only. Classification is measured on 1,920 validation ROIs;
    detection is measured on 327 full images. Their F1 and AP are different endpoints.
    The earlier proposed notebook 12 finalist repeats is deferred; this notebook performs no repeats.
    ''', COMMON + '''
OUT = ROOT/'docs/reports/synthesis'; OUT.mkdir(parents=True,exist_ok=True)
TRAINING = ROOT/'docs/reports/training'
provenance = []
def load(relative):
    path = ROOT/relative
    value = json.loads(path.read_text(encoding='utf-8'))
    provenance.append({'file':relative, 'sha256':sha(path)})
    return value
manifest = ROOT/'data/processed/gum_raw_roi_v1/manifest.csv'
raw = pd.read_csv(manifest)
assert raw.query("split == 'val'").shape[0] == 1920
assert sha(manifest) == 'b9feaaceb4b9ea94ad540b553927ffb7f67e731098f7a78c770941804cbb4f06'
display(raw.groupby(['split','class_name']).size().rename('ROIs').to_frame())
'''),
    ('''## 2. Consolidate scratch, pretrained and combined-method evidence
    Read the previously collected 6.x and 07.x summaries. No new metrics or clinical labels are generated.
    Context, learned gating and CORAL are separate hypotheses; the study did not show monotonic improvement.
    ''','''
method = load('docs/reports/training/roi_method_comparison.json')
assert method['validation_rows'] == 1920
rows=[]
for item in method['summary']:
    name=item['experiment']; detail=method['details'][name]
    rows.append({'stage':'6.x' if name.startswith('C') else '07.x', 'experiment':name,
                 'accuracy':item['accuracy'],'macro_f1':item['macro_f1'],
                 'no_visible_recall':detail['class_recall'][0], 'crop_version':'old_224',
                 'source':'roi_method_comparison.json'})
display(pd.DataFrame(rows))
'''),
    ('''## 3. Add source-resolution, class-imbalance and frozen-feature studies
    R0/R1 isolate crop preparation, R1/R2/R3 compare objectives. P0/P1 are frozen probes.
    These interventions do not share a single matched control, so a leaderboard alone cannot explain causality.
    ''','''
recovery=load('docs/reports/training/roi_recovery_comparison.json')
for item in recovery:
    rows.append({'stage':'09.1/09.2','experiment':item['trial']['id'],
                 'accuracy':item['accuracy'],'macro_f1':item['macro_f1'],
                 'no_visible_recall':item['class_recall'][0],
                 'crop_version':item['trial']['data'],'source':'roi_recovery_comparison.json'})
probe=load('docs/reports/training/roi_foundation_probe_comparison.json')
assert probe['version']==3 and probe['status']=='COMPLETE'
for item in probe['probes']:
    rows.append({'stage':'09.3','experiment':item['trial']['id'],
                 'accuracy':item['accuracy'],'macro_f1':item['macro_f1'],
                 'no_visible_recall':item['class_recall'][0],
                 'crop_version':'raw','source':'roi_foundation_probe_comparison.json'})
display(pd.DataFrame(rows).tail(6))
'''),
    ('''## 4. Add verified full DINOv3 results
    11.1 changes frozen readout, while 11.2 changes trainable layers under CE.
    CLS logistic regression is repeated from 09.3: list it for its matched comparison, not as independent confirmation.
    ''','''
for track in ['11_1','11_2']:
    receipt=load(f'docs/reports/training/dinov3_publication/{track}_full_v8/verification.json')
    assert receipt['state']=='COMPLETE' and receipt['test_images_opened']==0
    assert receipt['manifest_sha256']==sha(manifest)
    for item in receipt['verified_results']:
        name=item.get('feature',item.get('trial'))
        rows.append({'stage':track.replace('_','.'),'experiment':name,
                     'accuracy':item['accuracy'],'macro_f1':item['macro_f1'],
                     'no_visible_recall':item['class_recall'][0],
                     'crop_version':'raw','source':f'{track}_full_v8/verification.json'})
summary=pd.DataFrame(rows)
assert summary.experiment.notna().all() and len(summary)==17
summary.to_csv(OUT/'roi_experiment_matrix.csv',index=False)
display(summary.style.format({'accuracy':'{:.2%}','macro_f1':'{:.2%}','no_visible_recall':'{:.2%}'}))
'''),
    ('''## 5. Compare performance and minority-class failure
    Plot accuracy beside macro-F1 and no-visible recall. Accuracy can rise while the rare class remains unrecognized.
    All values are extensively reused validation scores, mostly one seed. The 75% line is a project aspiration.
    ''','''
fig,axes=plt.subplots(1,2,figsize=(14,7))
labels=summary.stage+' '+summary.experiment
y=np.arange(len(summary))
axes[0].barh(y,summary.macro_f1*100,label='Macro-F1')
axes[0].scatter(summary.accuracy*100,y,color='orange',label='Accuracy',s=18)
axes[0].set_yticks(y,labels,fontsize=8); axes[0].invert_yaxis()
axes[0].axvline(75,ls='--',color='gray',label='Project aspiration'); axes[0].legend(fontsize=8)
axes[0].set_xlabel('Validation score (%)')
axes[1].barh(y,summary.no_visible_recall*100,color='teal')
axes[1].set_yticks(y,labels,fontsize=8); axes[1].invert_yaxis()
axes[1].set_xlabel('No-visible recall (%)'); axes[1].set_xlim(0,100)
fig.tight_layout(); fig.savefig(OUT/'roi_research_progression.png'); plt.show()
'''),
    ('''## 6. Keep the detector evidence separate
    Reuse the executed 10.2 audit, including fixed blur, darkness and JPEG stress.
    Calculate annotation-relative precision/recall at confidence 0.25 from saved counts.
    These are not the native AP curve's precision/recall values and not phone-photo accuracy.
    ''','''
caries=load('docs/reports/post09/caries/summary.json')
assert caries['validation_images']==327 and caries['test_images_opened']==0
native=pd.read_csv(ROOT/'docs/reports/post09/caries/native_detection_metrics.csv')
errors=pd.read_csv(ROOT/'docs/reports/post09/caries/per_image_errors.csv')
counts=errors.groupby(['model','variant'])[['ground_truth','true_positive','missed','duplicate','localization','unmatched']].sum()
counts['apparent_fp']=counts[['duplicate','localization','unmatched']].sum(axis=1)
counts['precision_at_025']=counts.true_positive/(counts.true_positive+counts.apparent_fp)
counts['recall_at_025']=counts.true_positive/counts.ground_truth
detectors=native.merge(counts.reset_index(),on=['model','variant'],validate='one_to_one')
detectors.to_csv(OUT/'caries_detector_matrix.csv',index=False)
display(detectors[['model','variant','map50','map50_95','precision_at_025','recall_at_025']])
fig,ax=plt.subplots(figsize=(7,3))
for name,group in detectors.groupby('model'): ax.plot(group.variant,group.map50_95*100,marker='o',label=name)
ax.set(ylabel='Validation mAP50-95 (%)',title='Fixed synthetic stress: not real phone validation')
ax.legend(); fig.tight_layout();fig.savefig(OUT/'caries_stress_summary.png');plt.show()
'''),
    ('''## 7. Presentation narrative and evidence gaps
    Explain hypotheses and measured outcomes honestly. A failed combination still contributes an interpretable ablation.
    Annotation review is pending; no class regrouping or relabelling is performed here.
    ''','''
story=pd.DataFrame([
 ['6.x','Can a learned CNN and residual learning establish a reference?','Scratch and ImageNet controls completed.'],
 ['07.x','Do wider context, gating or ordinal loss help?','Controlled variants did not surpass C2.'],
 ['08-09.x','Are source detail and imbalance limiting performance?','Small observed gains; R1/R2 remain near-tied.'],
 ['10.x','Are errors shared, and does caries detection localize lesions?','Technical audit passed; clinical and capture-domain review pending.'],
 ['11.x','Do spatial foundation features or limited adaptation help?','Matched gains, but below ResNet18; minority failures persist.'],
 ['12.x','Can ordinary phone captures support the intended workflow?','Local feasibility study prepared; real captures required.']],
 columns=['stage','research_question','measured_conclusion'])
story.to_csv(OUT/'presentation_story.csv',index=False);display(story)
review=pd.read_csv(ROOT/'docs/reports/post09/gingival/reviewer_sheet.csv')
display(Markdown(f'**Existing reviewer sheet:** {len(review)} rows in `docs/reports/post09/gingival/reviewer_sheet.csv`. '
                 'Keep original labels unchanged. Qualified review remains pending.'))
gaps=pd.DataFrame([
 ['Independent phone capture domain','Pending real images; no accuracy without reference labels'],
 ['Gingival no-visible recognition','Unresolved; do not describe no-visible as verified healthy'],
 ['11.2 inference checkpoint','Not collected: Kaggle temp checkpoints were not exported'],
 ['Repeated seeds and final test','Not completed; test stays locked'],
 ['Android inference','Deferred; desktop CPU timing is not phone latency']],columns=['gap','status'])
display(gaps);gaps.to_csv(OUT/'evidence_gaps.csv',index=False)
'''),
    ('''## 8. Save the evidence receipt and next action
    No additional training, uploads or test evaluation occur. The paired photo notebook is local-only.
    Freeze the present comparison and collect feasible ordinary captures before choosing another model experiment.
    ''','''
receipt={'roi_rows':len(summary),'roi_validation_population':1920,'detector_rows':len(detectors),
         'detector_validation_images':327,'test_images_opened':0,'training_started':False,
         'sources':provenance,'next_action':'Collect local phone captures and run notebook 12.2',
         'personal_photos_uploaded':False}
(OUT/'receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
display(pd.Series({k:v for k,v in receipt.items() if k!='sources'}).to_frame('evidence'))
print('Saved evidence matrix, figures and source hashes:', OUT)
''')]
    save('12_1_research_evidence_synthesis.ipynb', '''# 12.1 Research evidence synthesis

An executed presentation companion for the completed study. No new model fitting.

```mermaid
flowchart LR
    A[Collected 6.x to 11.x evidence] --> B[Check source and population]
    B --> C[ROI classification comparisons]
    B --> D[Full-image detector comparisons]
    C --> E[Explain failures and scope]
    D --> E
    E --> F[Local phone capture feasibility]
```
''',sections)


def phone():
    sections=[
    ('''## 1. Local-only scope and fixed protocol
    This notebook may contain personal images in outputs. It is excluded from public code packages.
    Photos, predictions and galleries stay under ignored `data/pilot/` directories. No network or training is used.
    Detection confidence is not a disease probability. A zero-box image is not a healthy diagnosis.
    ''',COMMON+'''
from dentalselfie.phone_feasibility import inspect_photos, infer_images, VIEWS
PILOT=ROOT/'data/pilot/phone_feasibility'; INBOX=PILOT/'images'
INBOX.mkdir(parents=True,exist_ok=True)
META=PILOT/'capture_manifest.csv'
if not META.exists(): pd.DataFrame(columns=['file','view','consent']).to_csv(META,index=False)
PROTOCOL={'imgsz':640,'conf':0.25,'iou':0.7,'max_det':300}
display(pd.Series(PROTOCOL).to_frame('fixed desktop inference setting'))
print('Photo inbox:',INBOX, '| Capture manifest:',META)
'''),
    ('''## 2. Capture procedure and manifest
    Use ordinary accessible views; do not force deep intraoral angles. Prepare 15-30 JPEG/PNG/WebP images
    in total, including repeats with realistic differences in framing and light. Avoid names, location or other identifiers.
    Record permission for any participant; metadata value `yes` means permission has been recorded locally.
    HEIC is not supported by this runtime: export JPEG first. Original images are never modified.
    ''','''
display(pd.DataFrame([
 ['front_teeth','Teeth visible while smiling; close framing'],
 ['open_mouth','Open mouth with teeth and accessible surfaces visible'],
 ['left_accessible / right_accessible','Comfortable side view without accessories'],
 ['upper_accessible / lower_accessible','Only surfaces visible with an ordinary phone camera']],
 columns=['view','capture_instruction']))
display(pd.DataFrame([{'file':'example_001.jpg','view':'front_teeth','consent':'yes'}]))
print('Copy images into the inbox, add their exact filenames/views/consent to capture_manifest.csv, then rerun all cells.')
print('These example rows are instructions only; no synthetic participant or disease label is created.')
'''),
    ('''## 3. Inspect image bytes, orientation, quality and research-data overlap
    Apply EXIF orientation, inspect RGB dimensions and descriptive brightness/sharpness values.
    Do not discard a capture because of an unvalidated quality threshold. Exact known research-file bytes and
    duplicate captures are excluded from this independent feasibility input. This cannot prove subject independence.
    ''','''
known=set()
for task in ['caries','gum']:
    table=pd.read_csv(ROOT/f'data/manifests/{task}.splits.csv')
    for column in ['sha256','source_sha256']:
        if column in table: known.update(table[column].dropna().astype(str))
inventory=inspect_photos(INBOX,META,known)
inventory.to_csv(PILOT/'capture_inventory.csv',index=False)
display(inventory)
print('Ready captures:',int(inventory.status.eq('ready').sum()),'| Pending/invalid:',int(inventory.status.ne('ready').sum()))
if inventory.empty: print('AWAITING_PHONE_PHOTOS: the private inbox is empty.')
'''),
    ('''## 4. Verify inference plumbing on three locked validation images
    This deterministic source-validation smoke is separate from the phone study and has no accuracy claim.
    Check input hashes and both existing checkpoint hashes, run CPU inference, and keep zero-box results.
    It confirms loading/preprocessing/postprocessing/output plumbing; it cannot prove phone-domain performance.
    ''','''
lock=json.loads((ROOT/'data/manifests/caries.lock.json').read_text())
assert sha(ROOT/'data/manifests/caries.splits.csv')==lock['splits_sha256']
val=pd.read_csv(ROOT/'data/manifests/caries.splits.csv').query("split == 'val'").sort_values('sample_id').head(3)
assert len(val)==3
for row in val.itertuples(): assert sha(ROOT/row.image)==row.sha256
smoke_out=PILOT/'source_validation_smoke'
smoke_images,smoke_boxes=infer_images(ROOT,[ROOT/r.image for r in val.itertuples()],smoke_out,PROTOCOL)
assert len(smoke_images)==6
display(smoke_images[['model','file','detections','elapsed_ms']])
print('SOURCE VALIDATION SMOKE ONLY; phone captures evaluated: 0 in this cell.')
'''),
    ('''## 5. Run existing detectors on approved phone captures
    Keep the same inference settings and native detector postprocessing for both models.
    All ready captures remain in the image table, including those with no candidates. No ground-truth labels exist here.
    ''','''
ready=inventory[inventory.status.eq('ready')]
phone_out=PILOT/'phone_results'
phone_images,phone_boxes=infer_images(ROOT,[INBOX/name for name in ready.file],phone_out,PROTOCOL)
if phone_images.empty:
    print('NOT_EVALUATED: add approved real captures; no phone metrics or detections fabricated.')
else:
    assert len(phone_images)==2*len(ready)
    display(phone_images[['model','file','detections','elapsed_ms']]);display(phone_boxes)
'''),
    ('''## 6. Visualize outputs and desktop runtime
    Show original-size annotated detections with model names. In the empty-inbox state, display clearly identified
    source-validation smoke examples instead. Candidate boxes can be wrong; visual inspection is not a clinical label.
    Latency excludes model loading, warm-up, image decode and annotation rendering; it is desktop CPU timing only.
    ''','''
shown=phone_images if not phone_images.empty else smoke_images
base=phone_out if not phone_images.empty else smoke_out
print('Gallery population:', 'PHONE CAPTURES' if not phone_images.empty else 'SOURCE VALIDATION SMOKE, NOT PHONE IMAGES')
fig,axes=plt.subplots(2,min(3,len(shown)//2),figsize=(12,7),squeeze=False)
for ax,row in zip(axes.flat,shown.groupby('model',sort=False).head(3).itertuples()):
    with Image.open(base/row.annotated) as im: ax.imshow(im)
    ax.set_title(f'{row.model}: {row.detections} candidates',fontsize=9);ax.axis('off')
fig.tight_layout();plt.show()
display(shown.groupby('model').agg(images=('file','size'),zero_box_images=('detections',lambda s:int(s.eq(0).sum())),
                                  median_desktop_ms=('elapsed_ms','median')))
'''),
    ('''## 7. Create a qualitative review sheet and readiness receipt
    Record tooth visibility, framing/light/focus observations, candidate plausibility and suspected misses for review.
    Do not fill clinical correctness or calculate accuracy without an independent reference and defined annotation protocol.
    Existing reviewer entries are preserved across reruns. No phone image is added to train, validation or test.
    ''','''
review_path=PILOT/'phone_review.csv'
previous=pd.read_csv(review_path,keep_default_na=False) if review_path.exists() else pd.DataFrame()
review=ready[['file','view','sha256']].copy()
for column in ['teeth_visible','focus_observation','lighting_observation','candidate_observation','suspected_miss_for_review','reviewer_notes']:
    review[column]=''
if not previous.empty:
    old=previous.set_index('sha256')
    for index,row in review.iterrows():
        if row.sha256 in old.index:
            for col in review.columns.difference(['file','view','sha256']): review.loc[index,col]=old.loc[row.sha256,col]
if not review.empty: review.to_csv(review_path,index=False)
status={'state':'AWAITING_PHONE_PHOTOS' if ready.empty else 'QUALITATIVE_PHONE_INFERENCE_COMPLETED',
        'ready_phone_images':len(ready),'source_smoke_images':3,'source_smoke_model_image_rows':len(smoke_images),
        'phone_model_image_rows':len(phone_images),'accuracy_computed':False,'clinical_labels_available':False,
        'training_started':False,'test_images_opened':0,'uploaded':False,'device_measurement':'desktop_cpu'}
(PILOT/'readiness.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
display(review);display(pd.Series(status).to_frame('actual status'))
'''),
    ('''## 8. Implementation appendix
    The exact shared implementation below performs file checks and inference. It is displayed for presentation;
    it is not a second implementation. Inspect how zero detections are retained and EXIF metadata is not copied to outputs.
    ''','''
display(Code((ROOT/'src/dentalselfie/phone_feasibility.py').read_text(encoding='utf-8'),language='python'))
print('Implementation SHA256:',sha(ROOT/'src/dentalselfie/phone_feasibility.py'))
''')]
    save('12_2_phone_photo_feasibility.ipynb', '''# 12.2 Local phone-photo feasibility

Private, local-only capture inspection and existing-weight caries inference. No training and no clinical accuracy claim.

```mermaid
flowchart TD
    A[Local photo inbox] --> B[Consent and view manifest]
    B --> C[Inspect orientation and descriptive quality]
    C --> D[Existing YOLOv8n and YOLO26n checkpoints]
    D --> E[Boxes and zero-detection cases]
    E --> F[Qualitative capture review]
    G[Three source-validation images] --> H[Separate inference plumbing smoke]
```
''',sections,private=True)


if __name__=='__main__':
    synthesis()
    phone()

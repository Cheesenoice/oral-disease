from pathlib import Path
import hashlib, json
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
M=ROOT/'docs/reports/alignment/teeth_gap_v1/manifest.csv'
OUT=ROOT/'docs/reports/alignment/teeth_gap_v1/image_level_review_named'
EXPECTED='9c413bc2f2e811c22fb3e31f46f7ce584840fca75c5d6b99372ce17f1cb94036'
# Conservative first-pass visual flags from the named, box-overlaid contact sheets.
# These are dataset-quality candidates, not clinical relabels. They do not alter a run.
EXCLUDE={
 0:'before/after composite: one box is attached to one panel while the other panel is unlabeled',
 3:'distant face/portrait: target occupies only a few pixels',
 6:'educational graphic/text overlay rather than a standalone intraoral image',
 9:'before/after graphic with incompatible unlabeled comparison panel',
 10:'multi-panel before/after teaching image',
 11:'promotional infographic; box is on face region, not a tooth-space ROI',
 13:'distant face portrait; dental target has very low pixel support',
 15:'orthodontic/braces promotion with multiple boxes on tooth/contact areas',
 16:'educational graphic rather than a standalone clinical photo',
 17:'multi-panel educational comparison with text and inconsistent panel labels',
 21:'text/diagram graphic about closure rather than a standalone intraoral photo',
 23:'before/after comparison graphic; panels carry conflicting visual states',
 32:'orthodontic braces image with box on appliance/gingiva rather than a spacing target',
 37:'before/after composite, target box covers a panel-level gap',
 39:'promotional image/text overlay for space-between-teeth product',
 40:'hospital infographic with small embedded photos and text',
 42:'before/after graphic with paired panels',
 43:'diastema promotional poster with text and arrow overlay',
 52:'distant child portrait; low target resolution',
 56:'braces promotion / crowded anterior teeth; target definition does not match boxes well',
 65:'multi-panel clinical comparison; boxes depend on panel-specific context',
 87:'before/after educational panel with text',
 90:'lecture slide / composite image rather than standalone photo',
 94:'presentation slide with embedded clinical image and presenter/text',
 114:'orthodontic promotional image with boxes across crowded teeth',
 117:'distant portrait; dental target is too small for stable detector supervision',
}
# Rows with clear single-image views and visible box placement are only retain candidates,
# not ground-truth approvals. Everything else remains review-needed.
REVIEW={1,5,7,8,12,14,18,19,20,22,24,26,28,33,36,38,41,44,45,46,47,48,49,50,51,53,54,55,57,58,59,60,61,62,63,64,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,88,89,91,92,93,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,115,116,118}
assert hashlib.sha256(M.read_bytes()).hexdigest()==EXPECTED
rows=pd.read_csv(M); rows=rows[rows.split.eq('train')].sort_values('output_image').reset_index(drop=True)
assert len(rows)==119 and set(EXCLUDE).isdisjoint(REVIEW)
records=[]
for i,row in enumerate(rows.itertuples(index=False)):
    if i in EXCLUDE: status='exclude_candidate'; reason=EXCLUDE[i]
    elif i in REVIEW: status='manual_review'; reason='Single-view candidate or uncertain target/box alignment; retain only after explicit review.'
    else: status='retain_candidate'; reason='Single photo with visibly plausible spacing-box placement; not clinically adjudicated.'
    records.append({'atlas_row':i,'output_image':row.output_image,'source_stem':row.source_stem,'visual_group':row.group,'boxes':row.boxes,'image_sha256':row.image_sha256,'label_sha256':row.label_sha256,'visual_triage':status,'reason':reason,'used_in_current_kaggle_run':True,'source_test_opened':False})
out=pd.DataFrame(records); out.to_csv(OUT/'train_image_triage_v1.csv',index=False)
summary={'manifest_sha256':EXPECTED,'images_reviewed':len(out),'boxes':int(out.boxes.sum()),'visual_triage_counts':out.visual_triage.value_counts().to_dict(),'exclude_candidate_images':int((out.visual_triage=='exclude_candidate').sum()),'manual_review_images':int((out.visual_triage=='manual_review').sum()),'retain_candidate_images':int((out.visual_triage=='retain_candidate').sum()),'current_run_changed':False,'source_test_opened':False,'caveat':'Visual technical triage only; no clinical adjudication, relabeling, validation selection or current-run data change.'}
(OUT/'triage_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))

"""Build Kaggle notebook 16.70 from the pinned 16.67 NWD experiment."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat


ROOT = Path(__file__).resolve().parents[2]
SOURCE_NOTEBOOK = ROOT / "notebooks/alignment/16_67_spacing_train_addon_ablation.ipynb"
TARGET_NOTEBOOK = ROOT / "notebooks/alignment/16_70_spacing_context_crop_ablation.ipynb"
KAGGLE_STAGE = ROOT / ".venv/work/kaggle/spacing_context_crop_ablation"


def md(text: str):
    return nbformat.v4.new_markdown_cell(dedent(text).strip())


def code(text: str):
    return nbformat.v4.new_code_cell(dedent(text).strip())


def main() -> None:
    nb = nbformat.read(SOURCE_NOTEBOOK, as_version=4)
    nb.cells[0] = md(r"""
# 16.70 - Scale-preserving context crops for diastema detection

## Research question and fixed comparison

Does restricting mixed-panel source images to a target-centered local context window help the spacing detector, while preserving the original pixel scale and annotation geometry?

| Arm | Training data | Model and method |
|---|---|---|
| Control | Locked project train split | YOLO26s + common tooth-anatomy initialization + CIoU-NWD |
| Context-crop arm | Same project train split plus one context crop per eligible external train image | Identical |

The external export contains many collages. Each source image is 640 x 640; the transformation takes a 320 x 320 square around its largest diastema box and places that crop at native pixel scale on a black 640 x 640 canvas. Box coordinates are translated only. No zoom, relabeling, source validation or source test is used. Crops with a second target box partially cut by the crop boundary are excluded. Duplicate review combines seven visually confirmed mirror-invariant pHash <= 2 pairs with 16 additional train-only pairs found among 31 pHash 3-8 candidates. The latter were manually inspected and required SIFT RANSAC inliers >= 30, inlier ratio >= 0.80 and aligned grayscale correlation >= 0.98. Only confirmed repeated-photo variants are grouped; the lexicographically first filename is retained. Ambiguous candidates remain in training.

The paired primary evaluation remains the unchanged 477-image project validation, with the predeclared 473-image overlap-excluded sensitivity cohort reported separately. The validation is reused development data, not an independent clinical test. This is a source-data/method ablation, not a claim of clinical readiness or guaranteed improvement.

**Source attribution:** AI WideTech Dental Disease v2, Roboflow Universe, CC BY 4.0. Only a train-only candidate subset is attached; it contains no source validation or test files. See the local 16.69 audit report for its visual and label limitations.
""")

    nb.cells[2].source = nb.cells[2].source.replace("ultralytics==8.4.183", "ultralytics==8.4.183 imagehash==4.3.2")
    nb.cells[3].source = nb.cells[3].source.replace("from PIL import Image\n", "from PIL import Image, ImageDraw, ImageOps\nimport imagehash\n")
    nb.cells[3].source = nb.cells[3].source.replace(
        "assert __import__('ultralytics').__version__=='8.4.183'",
        "assert __import__('ultralytics').__version__=='8.4.183'\nassert imagehash.__version__=='4.3.2'",
    )
    nb.cells[3].source = nb.cells[3].source.replace("OUT=Path('/kaggle/working/spacing_addon')", "OUT=Path('/kaggle/working/spacing_context_crop')")
    nb.cells[3].source = nb.cells[3].source.replace("TMP=Path('/tmp/spacing_addon')", "TMP=Path('/tmp/spacing_context_crop')")

    nb.cells[5] = code(r"""
EXPECTED_MANIFEST='38cba07194707131b629b020ea9078d487dd397e13221d88075ca8d9cf7a3e9b'
def sha256_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
manifest_paths=[p for p in INPUT.rglob('manifest.csv') if sha256_file(p)==EXPECTED_MANIFEST]
assert len(manifest_paths)==1,manifest_paths
SOURCE=manifest_paths[0].parent; manifest=pd.read_csv(manifest_paths[0],dtype={'sample_id':str})
assert len(manifest)==15511 and manifest.sample_id.is_unique
assert manifest.split.value_counts().to_dict()=={'train':15034,'valid':477}
assert not (SOURCE/'test').exists()
TRAIN_OVERLAP_IDS={'d7f3e2f640c5c219cf0486d4','cdd4842bb15ff4a10286b1d2','2d317bf73290741503a7b773','84558726296547622c33c6cf'}
VALID_OVERLAP_IDS={'ebb582d897abd9e1f55f4618','1a279792c0f591cf6dabe546','f8c55f96a0c5d7e0fe735d32','5b52ca77ecd22069f79c356d'}
assert TRAIN_OVERLAP_IDS.issubset(set(manifest.loc[manifest.split.eq('train'),'sample_id']))
assert VALID_OVERLAP_IDS.issubset(set(manifest.loc[manifest.split.eq('valid'),'sample_id']))
train_rows=manifest[manifest.split.eq('train')&~manifest.sample_id.isin(TRAIN_OVERLAP_IDS)].copy()
valid_rows=manifest[manifest.split.eq('valid')].copy()
sensitivity_rows=valid_rows[~valid_rows.sample_id.isin(VALID_OVERLAP_IDS)].copy()
assert (len(train_rows),len(valid_rows),len(sensitivity_rows))==(15030,477,473)
print({'manifest_sha256':EXPECTED_MANIFEST,'train_images':len(train_rows),'primary_valid':len(valid_rows),'sensitivity_valid':len(sensitivity_rows),'source_test_opened':False})

GHQ_PACKAGE_SHA='becfefe26a9fd7080d86048430897aee38eac30f3740272110c3bf7836f4a299'
GHQ_SOURCE_SHA='5f4d9a6c974b6a43521119b1edd6f3c17cf67c24aa62e1eebb6c5b0b9b95521a'
candidate_packages=[p for p in INPUT.rglob('*.zip') if sha256_file(p)==GHQ_PACKAGE_SHA]
assert len(candidate_packages)==1,candidate_packages
GHQ_ZIP=candidate_packages[0]
GHQ_EXTRACTED=TMP/'ghqwdo_train_only'
with __import__('zipfile').ZipFile(GHQ_ZIP) as zf:
    names=zf.namelist()
    assert not any(n.startswith(('valid/','test/')) for n in names)
    zf.extractall(GHQ_EXTRACTED)
ghq_manifest=pd.read_csv(GHQ_EXTRACTED/'manifest.csv')
assert len(ghq_manifest)==572 and ghq_manifest.source_group.is_unique
assert int(ghq_manifest.diastema_boxes.sum())==917
assert sha256_file(GHQ_ZIP)==GHQ_PACKAGE_SHA
print({'candidate_package_sha256':GHQ_PACKAGE_SHA,'source_archive_sha256':GHQ_SOURCE_SHA,
       'train_only_positive_images':len(ghq_manifest),'source_boxes':int(ghq_manifest.diastema_boxes.sum()),
       'source_valid_or_test_files_attached':False})
""")

    nb.cells[6] = md(r"""
## 3. Prepare paired project splits and make target-centered context crops

Both arms receive byte-identical project train, validation, and sensitivity cohorts. The external arm adds one image per deduplicated source train group. The 320-pixel crop is translated onto a 640-pixel canvas at original pixel scale; target box width/height are unchanged and only center coordinates move. A second box that intersects the crop but cannot be represented in full causes that crop to be quarantined, avoiding truncated positives. This limits the field of view but does not guarantee removal of every neighboring panel or label inconsistency.

After crop generation, compute pHash on foreground-normalized crops and horizontal mirrors. Review all distance <= 2 and 3-8 candidates using contact sheets. The second band is only a candidate generator: keep an exclusion only after same-photo visual confirmation and strong SIFT/pixel alignment evidence. The fixed project validation is never used for crop selection or deduplication.
""")

    nb.cells[7] = code(r"""
BASE=TMP/'base'; CROP_RUN=TMP/'context_crop'; counts={}
def prepare_project_split(root, split, rows):
    image_dir=root/split/'images'; label_dir=root/split/'labels'
    image_dir.mkdir(parents=True,exist_ok=True); label_dir.mkdir(parents=True,exist_ok=True)
    boxes=positives=0
    for row in rows.itertuples(index=False):
        image=SOURCE/row.image; label=SOURCE/row.label
        assert image.is_file() and label.is_file()
        raw=label.read_bytes(); assert hashlib.sha256(raw).hexdigest()==row.label_sha256
        kept=[]
        for line in raw.decode('utf-8').splitlines():
            fields=line.split(); assert len(fields)==5 and fields[0] in ('0','1')
            if fields[0]=='0': kept.append(line)
        assert len(kept)==int(row.gap_or_spacing_boxes)
        target=image_dir/image.name
        if not target.exists(): target.symlink_to(image)
        (label_dir/(image.stem+'.txt')).write_text(('\n'.join(kept)+'\n') if kept else '',encoding='utf-8')
        boxes+=len(kept); positives+=bool(kept)
    return {'images':len(rows),'positive_images':int(positives),'boxes':int(boxes)}

for root in [BASE,CROP_RUN]:
    counts[root.name]={}
    for split,rows in [('train',train_rows),('valid',valid_rows),('valid_sensitivity',sensitivity_rows)]:
        counts[root.name][split]=prepare_project_split(root,split,rows)

def parse_boxes(label_text):
    boxes=[]
    for line in label_text.splitlines():
        f=line.split()
        if not f: continue
        assert len(f)==5 and f[0]=='0'
        boxes.append(tuple(map(float,f[1:])))
    assert boxes
    return boxes

def context_crop(image, boxes, side=320, canvas_side=640):
    assert image.size==(canvas_side,canvas_side)
    # Center the context window on the largest annotated target.
    selected=max(boxes,key=lambda b:b[2]*b[3])
    x0=round(selected[0]*canvas_side-side/2); y0=round(selected[1]*canvas_side-side/2)
    crop=Image.new('RGB',(side,side),'black')
    sx0=max(0,x0); sy0=max(0,y0); sx1=min(canvas_side,x0+side); sy1=min(canvas_side,y0+side)
    crop.paste(image.crop((sx0,sy0,sx1,sy1)),(sx0-x0,sy0-y0))
    output=Image.new('RGB',(canvas_side,canvas_side),'black'); offset=(canvas_side-side)//2
    output.paste(crop,(offset,offset)); transformed=[]; partial=False
    for xc,yc,bw,bh in boxes:
        px=xc*canvas_side; py=yc*canvas_side
        left,right=px-bw*canvas_side/2,px+bw*canvas_side/2
        top,bottom=py-bh*canvas_side/2,py+bh*canvas_side/2
        fully=(left>=x0 and right<=x0+side and top>=y0 and bottom<=y0+side)
        intersects=(min(right,x0+side)>max(left,x0) and min(bottom,y0+side)>max(top,y0))
        if not fully:
            partial |= intersects
            continue
        transformed.append(f'0 {(offset+px-x0)/canvas_side:.8f} {(offset+py-y0)/canvas_side:.8f} {bw:.8f} {bh:.8f}')
    assert transformed
    return output,transformed,partial

crop_images=TMP/'generated_context_crops'/'images'; crop_labels=TMP/'generated_context_crops'/'labels'
crop_images.mkdir(parents=True,exist_ok=True); crop_labels.mkdir(parents=True,exist_ok=True)
crop_records=[]; partial_exclusions=[]; gallery=[]
for row in ghq_manifest.itertuples(index=False):
    image_path=GHQ_EXTRACTED/'train/images'/row.output_image
    label_path=GHQ_EXTRACTED/'train/labels'/row.output_label
    assert image_path.is_file() and label_path.is_file()
    assert sha256_file(image_path)==row.image_sha256
    assert sha256_file(label_path)==row.output_target_label_sha256
    with Image.open(image_path) as opened: image=opened.convert('RGB')
    boxes=parse_boxes(label_path.read_text(encoding='utf-8'))
    crop,new_labels,partial=context_crop(image,boxes)
    if partial:
        partial_exclusions.append(row.output_image)
        continue
    stem=Path(row.output_image).stem+'_context'
    out_image=crop_images/(stem+'.jpg'); out_label=crop_labels/(stem+'.txt')
    crop.save(out_image,quality=95,subsampling=0)
    out_label.write_text('\n'.join(new_labels)+'\n',encoding='utf-8')
    crop_records.append({'source_image':row.output_image,'source_group':row.source_group,
        'crop_image':str(out_image.relative_to(TMP)),'crop_label':str(out_label.relative_to(TMP)),'source_boxes':len(boxes),
        'crop_boxes':len(new_labels),'crop_side_px':320,'canvas_side_px':640,
        'pixel_scale_preserved':True,'image_sha256':sha256_file(out_image),'label_sha256':sha256_file(out_label)})
    if len(gallery)<12:
        shown=crop.copy(); draw=ImageDraw.Draw(shown)
        for line in new_labels:
            _,xc,yc,bw,bh=line.split();xc,yc,bw,bh=map(float,(xc,yc,bw,bh))
            draw.rectangle(((xc-bw/2)*640,(yc-bh/2)*640,(xc+bw/2)*640,(yc+bh/2)*640),outline='red',width=3)
        gallery.append(shown)

crop_manifest=pd.DataFrame(crop_records)
assert crop_manifest.source_group.is_unique
assert (crop_manifest.crop_boxes>0).all() and crop_manifest.pixel_scale_preserved.all()
assert len(partial_exclusions)==13 and len(crop_manifest)==559
outside_crop_boxes=int(crop_manifest.source_boxes.sum()-crop_manifest.crop_boxes.sum())
assert int(crop_manifest.source_boxes.sum())==874 and outside_crop_boxes==184
def normalized_mirror_phashes(path):
    with Image.open(path) as opened:
        image=opened.convert('RGB')
        pixels=np.asarray(image)
    yy,xx=np.where(pixels.max(axis=2)>18)
    if len(xx): pixels=pixels[yy.min():yy.max()+1,xx.min():xx.max()+1]
    content=Image.fromarray(pixels); content.thumbnail((128,128))
    canvas=Image.new('RGB',(128,128),'black')
    canvas.paste(content,((128-content.width)//2,(128-content.height)//2))
    return (int(str(imagehash.phash(canvas)),16),
            int(str(imagehash.phash(ImageOps.mirror(canvas))),16))

names=crop_manifest.source_image.tolist(); hashes={}
for row in crop_manifest.itertuples(index=False):
    hashes[row.source_image]=normalized_mirror_phashes(TMP/row.crop_image)
parents={name:name for name in names}
def find(name):
    while parents[name]!=name:
        parents[name]=parents[parents[name]]; name=parents[name]
    return name
def union(a,b):
    ra,rb=find(a),find(b)
    if ra!=rb: parents[max(ra,rb)]=min(ra,rb)
duplicate_edges=[]
near_duplicate_candidates=[]
for i,a in enumerate(names):
    for b in names[i+1:]:
        distance=min((ha^hb).bit_count() for ha in hashes[a] for hb in hashes[b])
        if distance<=2:
            duplicate_edges.append({'image_a':a,'image_b':b,'mirror_invariant_phash_distance':distance})
            union(a,b)
        elif distance<=8:
            near_duplicate_candidates.append({'image_a':a,'image_b':b,'mirror_invariant_phash_distance':distance})
EXPECTED_REVIEWED_PAIRS={
 ('375_jpg.rf.b2179fc3a449256b639d51200295de2b.jpg','gap-in-teeth_208_jpeg_jpg.rf.e3a4820f448b0cf4f05e52b38bbedca6.jpg'):0,
 ('Screenshot-2021-11-29-at-3-00-54-PM_png_jpg.rf.46bbc0cc92d6077fe297d280702f78fd.jpg','space-between-premolars_132_jpeg_jpg.rf.39817d87c26f086decf5cd6e172ddddf.jpg'):0,
 ('images-11-_jpg.rf.c759eebfb28375631604208510ad4786.jpg','images11-Copy_jpg.rf.789452bd1f9dea5aa444546dd9700ca2.jpg'):0,
 ('AdobeStock_75846210-1200x800_jpeg_jpg.rf.c0b18672721003d7be7eeb432bdb1ee0.jpg','images-17-_jpg.rf.8f4cacb60f86b9c2802ce531cf3ec602.jpg'):2,
 ('IMG_3332_JPG_jpg.rf.a86d6b9f4e040f63e8f0c2ea9257ab88.jpg','Screenshot-2021-10-31-at-2-28-04-PM_png_jpg.rf.bf66b6e5e1020e753688689e4cb2997a.jpg'):2,
 ('case-II-before-scaled-e1607388085941_jpg.rf.6c736c30a8f0d12f769a86a1ba8376e3.jpg','download-8-_jpg.rf.435a756888cae8b5ed10facb012e4bf1.jpg'):2,
 ('diastema-teeth_136_jpeg_jpg.rf.31852607e60b9f1f590138319652cd81.jpg','diastema-teeth_136_jpeg_jpg.rf.e0c29691624a8fb7c1013c51701ff18f.jpg'):2,
}
observed_pairs={tuple(sorted((edge['image_a'],edge['image_b']))):edge['mirror_invariant_phash_distance'] for edge in duplicate_edges}
assert observed_pairs==EXPECTED_REVIEWED_PAIRS,observed_pairs
assert len(near_duplicate_candidates)==31, len(near_duplicate_candidates)
SIFT_REVIEWED_PAIRS=[
 ('0219_Skramstad_01_jpg.rf.6d5d45cb587e104a5fe1567ae44d043f.jpg','diastema-teeth_112_jpeg_jpg.rf.d61934a6d3ef15d46a49ab262ac66ddc.jpg',4,188,.979167,.997360),
 ('B1-2_jpg.rf.f6b7f83d9dbd899c4c0fbfaf6730d8e9.jpg','diastema-teeth_114_jpeg_jpg.rf.eec26b61a23272c556ec956be5304789.jpg',6,185,.948718,.997089),
 ('IMG_3612_JPG_jpg.rf.36cda9e8dbb9b0494c830feff4a17458.jpg','IMG_5408-scaled_jpg.rf.3f97348497895a9e6c377c15912a8459.jpg',8,160,.930233,.985407),
 ('download-9-_jpg.rf.8e59376855eecd07df07c4e0213db47f.jpg','download-9-_jpg.rf.c32b0fa9ada124883b80c58aa363a308.jpg',4,145,.973154,.998727),
 ('27_jpg.rf.248fdeb42b8f25a7188bf75c2c46b2d6.jpg','gap-in-teeth-from-inside-mouth_288_jpeg_jpg.rf.3d55fe87152b1f9b6df8ff6281cc8802.jpg',6,119,.915385,.985265),
 ('misaligned-teeth_122_jpeg_jpg.rf.677f9e5993fa5f332611482b37d8c378.jpg','misaligned-teeth_122_jpeg_jpg.rf.c91e7bc05d0c763ae2a57d07cdb8365c.jpg',4,100,.934579,.996650),
 ('diastema-teeth_143_jpeg_jpg.rf.20e07a979607675acb3f316372553afe.jpg','gap-between-molars_12_jpeg_jpg.rf.c42f2f3504d10d9d3144f331e38289dc.jpg',4,99,.831933,.985283),
 ('c1a_jpg.rf.b62a5d0f778c091b76991a932ac0bf82.jpg','download-5-_jpg.rf.41d3e8b7be2386b80083b795bf133cc4.jpg',6,75,.872093,.990070),
 ('images205_jpg.rf.32c5e72c40ec821ff857266468d4693d.jpg','overcrowded-teeth_147_jpeg_jpg.rf.4f323530cadb0ca479619e51dc3141b5.jpg',4,72,.960000,.998913),
 ('singleimplant-before1_jpg.rf.34baf7519c254cf40c3ae2ecb94c853c.jpg','unhealthy_teeth_292_jpg.rf.ca7c93eba11c0f7b14e328f57e4f8a9b.jpg',6,64,.888889,.998125),
 ('images40_jpg.rf.3e9a9985d38b1ee9b7ec37d79ceb5a32.jpg','unhealthy_teeth_258_jpg.rf.cff5f3875d35f7b3fb84f6d4612fc736.jpg',6,63,.807692,.999027),
 ('Before-implant1_jpg.rf.dd174ace43617db6f32800f3e18d3003.jpg','diastema-teeth_101_jpeg_jpg.rf.0781dbbbfdbf59bff15ae80b6a993a61.jpg',6,59,.951613,.991271),
 ('gap-in-teeth_105_jpeg_jpg.rf.04d25d96346dbe88210caedd8f935009.jpg','image28-Copy_jpeg_jpg.rf.2fd716dc4cf1b95172c6c362cecefd3b.jpg',6,57,.904762,.995907),
 ('diastema-teeth_155_png_jpg.rf.3f51d2bc3afc141fbda253861cf6b181.jpg','diastema-teeth_155_png_jpg.rf.a40933a1dae9e8cef0fbb07ece7b4e18.jpg',6,47,.810345,.995732),
 ('diastema-teeth_140_jpeg_jpg.rf.f34f2dd6b506fe46d0d957f60743d8ad.jpg','diastema-teeth_140_jpeg_jpg.rf.f99e8894be8aa899aae7f21a182990d1.jpg',4,41,.911111,.996691),
 ('3-Figure5-1_png_jpg.rf.4e1f215debfe365ca0877c7d649504bd.jpg','images-25-_jpg.rf.37c073abbfbd049a68c1196e9337965d.jpg',6,40,.888889,.994738),
]
candidate_distance={(edge['image_a'],edge['image_b']):edge['mirror_invariant_phash_distance'] for edge in near_duplicate_candidates}
assert all(candidate_distance.get((a,b))==distance for a,b,distance,*_ in SIFT_REVIEWED_PAIRS)
for a,b,distance,inliers,inlier_ratio,pixel_corr in SIFT_REVIEWED_PAIRS:
    assert inliers>=30 and inlier_ratio>=.80 and pixel_corr>=.98
    duplicate_edges.append({'image_a':a,'image_b':b,'mirror_invariant_phash_distance':distance,
        'review_tier':'visual_plus_sift_pixel','sift_ransac_inliers':inliers,
        'sift_inlier_ratio':inlier_ratio,'aligned_grayscale_correlation':pixel_corr})
    union(a,b)
components={}
for name in names: components.setdefault(find(name),[]).append(name)
duplicate_groups=[sorted(group) for group in components.values() if len(group)>1]
retained_names={min(group) for group in components.values()}
crop_manifest['crop_duplicate_group']=[find(name) for name in names]
crop_manifest['retained_after_crop_dedup']=[name in retained_names for name in names]
crop_manifest.to_csv(OUT/'context_crop_manifest_all.csv',index=False)
crop_duplicate_edges=pd.DataFrame(duplicate_edges)
crop_duplicate_edges.to_csv(OUT/'crop_duplicate_edges_reviewed.csv',index=False)
deduplicated_crop_manifest=crop_manifest[crop_manifest.retained_after_crop_dedup].copy()
excluded_duplicate_crops=crop_manifest[~crop_manifest.retained_after_crop_dedup].copy()
duplicate_source_boxes_removed=int(excluded_duplicate_crops.source_boxes.sum())
duplicate_crop_boxes_removed=int(excluded_duplicate_crops.crop_boxes.sum())
final_source_boxes=int(deduplicated_crop_manifest.source_boxes.sum())
final_outside_crop_boxes=final_source_boxes-int(deduplicated_crop_manifest.crop_boxes.sum())
excluded_duplicate_crops.to_csv(OUT/'crop_duplicate_exclusions.csv',index=False)
assert len(duplicate_edges)==23 and len(duplicate_groups)==23 and len(excluded_duplicate_crops)==23
assert len(deduplicated_crop_manifest)==536
assert duplicate_source_boxes_removed==32 and duplicate_crop_boxes_removed==28
display(crop_duplicate_edges)
duplicate_gallery=Image.new('RGB',(2*330,len(duplicate_edges)*220),'white'); duplicate_draw=ImageDraw.Draw(duplicate_gallery)
crop_rows=crop_manifest.set_index('source_image')
for i,edge in crop_duplicate_edges.iterrows():
    for column,key in enumerate(['image_a','image_b']):
        row=crop_rows.loc[edge[key]]
        with Image.open(TMP/row.crop_image) as opened: shown=opened.convert('RGB')
        label_text=(TMP/row.crop_label).read_text(encoding='utf-8')
        for line in label_text.splitlines():
            _,xc,yc,bw,bh=map(float,line.split())
            x0=(xc-bw/2)*640; y0=(yc-bh/2)*640; x1=(xc+bw/2)*640; y1=(yc+bh/2)*640
            ImageDraw.Draw(shown).rectangle((x0,y0,x1,y1),outline='red',width=3)
        shown.thumbnail((310,180))
        x=column*330+(310-shown.width)//2; y=i*220+30
        duplicate_gallery.paste(shown,(x,y))
        duplicate_draw.text((column*330+4,i*220+4),edge[key][:40],fill='black')
duplicate_gallery.save(OUT/'crop_duplicate_review_gallery.jpg',quality=92)
display(duplicate_gallery)
(OUT/'context_crop_manifest.csv').write_text(deduplicated_crop_manifest.to_csv(index=False),encoding='utf-8')
(OUT/'partial_target_crop_exclusions.json').write_text(json.dumps(partial_exclusions,indent=2))
crop_addon_images=0; crop_addon_boxes=0
training_image_dir=CROP_RUN/'train'/'images'; training_label_dir=CROP_RUN/'train'/'labels'
for row in deduplicated_crop_manifest.itertuples(index=False):
    image=TMP/row.crop_image; label=TMP/row.crop_label
    target=training_image_dir/image.name
    if not target.exists(): target.symlink_to(image)
    shutil.copyfile(label,training_label_dir/label.name)
    crop_addon_images+=1; crop_addon_boxes+=len(label.read_text(encoding='utf-8').splitlines())

# Build an interpretable 12-example overlay output for the method presentation.
sheet=Image.new('RGB',(4*250,3*270),'white'); draw=ImageDraw.Draw(sheet)
for i,im in enumerate(gallery):
    x=(i%4)*250; y=(i//4)*270; im.thumbnail((240,240)); sheet.paste(im,(x+(250-im.width)//2,y+20)); draw.text((x+5,y+4),f'crop example {i+1}',fill='black')
sheet.save(OUT/'context_crop_overlay_gallery.jpg',quality=92)

base_train_count=len(list((BASE/'train/images').iterdir()))
base_box_count=sum(len(p.read_text(encoding='utf-8').splitlines()) for p in (BASE/'train/labels').glob('*.txt'))
crop_box_count=base_box_count+crop_addon_boxes
assert base_train_count==15030 and base_box_count==8223
assert crop_addon_images==536 and crop_addon_boxes==662
assert crop_box_count==8885
for root in [BASE,CROP_RUN]:
    (root/'data.yaml').write_text(yaml.safe_dump({'path':str(root),'train':'train/images','val':'valid/images','names':{0:'gap_or_spacing'}},sort_keys=False))
    (root/'sensitivity.yaml').write_text(yaml.safe_dump({'path':str(root),'train':'train/images','val':'valid_sensitivity/images','names':{0:'gap_or_spacing'}},sort_keys=False))
assert counts['base']['valid']==counts['context_crop']['valid']
assert counts['base']['valid_sensitivity']==counts['context_crop']['valid_sensitivity']
display(pd.DataFrame({'control_project_split':counts['base'],'context_arm_project_portion':counts['context_crop']}))
display(pd.DataFrame([{'external_candidate_images':572,'partial_target_crops_excluded':len(partial_exclusions),
    'pre_dedup_crops':len(crop_manifest),'phash_le2_pairs':len(EXPECTED_REVIEWED_PAIRS),
    'phash_3_to_8_candidates':len(near_duplicate_candidates),'sift_visual_confirmed_pairs':len(SIFT_REVIEWED_PAIRS),
    'duplicate_crops_removed':len(excluded_duplicate_crops),
    'duplicate_source_boxes_removed':duplicate_source_boxes_removed,
    'duplicate_crop_boxes_removed':duplicate_crop_boxes_removed,
    'retained_source_target_boxes':final_source_boxes,
    'target_boxes_outside_selected_crop':final_outside_crop_boxes,
    'context_crops_added':crop_addon_images,'crop_target_boxes':crop_addon_boxes,
    'control_train_images':base_train_count,'context_crop_train_images':base_train_count+crop_addon_images,
    'control_train_boxes':base_box_count,'context_crop_train_boxes':crop_box_count,
    'primary_validation_images':477,'sensitivity_validation_images':473}]))
display(Image.open(OUT/'context_crop_overlay_gallery.jpg'))
print({'partial_crop_exclusion_reasons':'unrepresented target would be partially visible',
       'phash_le2_pairs':len(EXPECTED_REVIEWED_PAIRS),'phash_3_to_8_candidates':len(near_duplicate_candidates),
       'sift_visual_confirmed_pairs':len(SIFT_REVIEWED_PAIRS),'duplicate_crops_removed':len(excluded_duplicate_crops),
       'source_validation_used':False,'source_test_opened':False,'project_validation_changed':False,
       'source_crop_pixel_scale_preserved':True})
""")

    nb.cells[15].source = nb.cells[15].source.replace("ADDON_RUN/'data.yaml'", "CROP_RUN/'data.yaml'")
    nb.cells[15].source = nb.cells[15].source.replace("external_train_4x", "context_crop")
    old_guard = "def guard(trainer):\n if time.monotonic()-start>5*3600:trainer.stop=True;print('Five-hour epoch-boundary guard reached',flush=True)"
    new_guard = "def guard(trainer):\n elapsed=time.monotonic()-start\n print({'completed_epoch':trainer.epoch+1,'worker_elapsed_seconds':round(elapsed,1)},flush=True)\n if elapsed>5*3600:trainer.stop=True;print('Five-hour epoch-boundary guard reached',flush=True)"
    assert old_guard in nb.cells[13].source
    nb.cells[13].source = nb.cells[13].source.replace(old_guard,new_guard)
    assert "cmd=[sys.executable,str(worker_path)" in nb.cells[15].source
    nb.cells[15].source = nb.cells[15].source.replace("cmd=[sys.executable,str(worker_path)","cmd=[sys.executable,'-u',str(worker_path)")
    old_monitor = """deadline=time.monotonic()+6.25*3600
while any(p.poll() is None for p in processes.values()):
    states={k:p.poll() for k,p in processes.items()}
    print({'worker_exit_codes':states,'elapsed_hours':round((6.25*3600-(deadline-time.monotonic()))/3600,2)},flush=True)
    if time.monotonic()>deadline:
        for p in processes.values():
            if p.poll() is None:p.terminate()
        raise TimeoutError('Safety guard reached. Preserve checkpoints and logs for continuation.')
    time.sleep(300)"""
    new_monitor = """deadline=time.monotonic()+6.25*3600
no_progress_limit=15*60
last_log_change={arm:time.monotonic() for arm in processes}
last_log_mtime={arm:(OUT/f'{arm}.log').stat().st_mtime for arm in processes}
last_status_print=0.0
while any(p.poll() is None for p in processes.values()):
    now=time.monotonic(); states={k:p.poll() for k,p in processes.items()}; stale={}
    for arm in processes:
        current_mtime=(OUT/f'{arm}.log').stat().st_mtime
        if current_mtime>last_log_mtime[arm]:last_log_mtime[arm]=current_mtime;last_log_change[arm]=now
        if processes[arm].poll() is None and now-last_log_change[arm]>no_progress_limit:
            stale[arm]=round(now-last_log_change[arm])
    if now-last_status_print>=300:
        print({'worker_exit_codes':states,'no_progress_seconds':{k:round(now-last_log_change[k]) for k in processes},
               'elapsed_hours':round((6.25*3600-(deadline-now))/3600,2)},flush=True)
        last_status_print=now
    if stale or now>deadline:
        for p in processes.values():
            if p.poll() is None:p.terminate()
        for p in processes.values():
            try:p.wait(timeout=30)
            except subprocess.TimeoutExpired:p.kill()
        for handle in handles.values():handle.close()
        if stale:raise TimeoutError(f'No-progress watchdog stopped paired workers after {stale}; checkpoints/logs retained.')
        raise TimeoutError('6.25-hour safety guard reached; checkpoints/logs retained.')
    time.sleep(60)"""
    assert old_monitor in nb.cells[15].source
    nb.cells[15].source = nb.cells[15].source.replace(old_monitor,new_monitor)
    nb.cells[12].source = nb.cells[12].source.replace(
        "the declared 4x repetition of the external train partition",
        "adding each eligible external crop once",
    )

    nb.cells[17] = code(r"""
arms={}
for arm,data_root,gpu in [('project_control',BASE,0),('context_crop',CROP_RUN,1)]:
    run=OUT/arm/arm; best=run/'weights'/'best.pt'; assert best.is_file()
    history=pd.read_csv(run/'results.csv'); history.columns=history.columns.str.strip()
    primary=YOLO(str(best)).val(data=str(data_root/'data.yaml'),imgsz=640,batch=16,device=gpu,workers=2,plots=True)
    sensitivity=YOLO(str(best)).val(data=str(data_root/'sensitivity.yaml'),imgsz=640,batch=16,device=gpu,workers=2,plots=False)
    arms[arm]={'epochs':len(history),'checkpoint_sha256':sha256_file(best),'primary_477':primary.results_dict,'sensitivity_473':sensitivity.results_dict}
    shutil.copy2(best,OUT/f'{arm}_best.pt'); shutil.copy2(run/'results.csv',OUT/f'{arm}_results.csv')
summary={'experiment':'scale_preserving_context_crop_external_data_ablation','spacing_manifest_sha256':EXPECTED_MANIFEST,
 'candidate_package_sha256':GHQ_PACKAGE_SHA,'source_archive_sha256':GHQ_SOURCE_SHA,
 'mask_pretrain_checkpoint_sha256':EXPECTED_MASK_CKPT,'common_initialization_sha256':sha256_file(COMMON_INIT),
 'project_train_images':15030,'project_train_boxes':8223,'external_source_candidate_images':572,
    'pre_dedup_context_crops':len(crop_manifest),'phash_le2_pairs':len(EXPECTED_REVIEWED_PAIRS),
    'phash_3_to_8_candidates':len(near_duplicate_candidates),'sift_visual_confirmed_pairs':len(SIFT_REVIEWED_PAIRS),
 'duplicate_crops_removed':len(excluded_duplicate_crops),
 'duplicate_source_boxes_removed':duplicate_source_boxes_removed,
 'duplicate_crop_boxes_removed':duplicate_crop_boxes_removed,
 'external_source_boxes_before_crop_dedup':int(crop_manifest.source_boxes.sum()),
 'external_target_boxes_outside_selected_crop_before_dedup':outside_crop_boxes,
 'external_source_boxes_in_final_crops':final_source_boxes,
 'external_target_boxes_outside_selected_crop_after_dedup':final_outside_crop_boxes,
 'partial_target_crop_exclusions':len(partial_exclusions),'external_context_crops':crop_addon_images,
 'external_context_crop_boxes':crop_addon_boxes,'crop_side_px':320,'canvas_side_px':640,
 'pixel_scale_preserved':True,'context_crop_arm_train_images':base_train_count+crop_addon_images,
 'context_crop_arm_train_boxes':crop_box_count,'NWD_C_input_pixels':C_NWD,'NWD_mix_weight':NWD_LAMBDA,
 'primary_validation_images':477,'sensitivity_validation_images':473,'source_validation_used':False,
 'source_test_opened':False,'validation_reused':True,'patient_independence_verified':False,'arms':arms}
(OUT/'comparison.json').write_text(json.dumps(summary,indent=2,default=str),encoding='utf-8')
display(pd.DataFrame({k:{'AP50':v['primary_477'].get('metrics/mAP50(B)'),'AP50-95':v['primary_477'].get('metrics/mAP50-95(B)'),'precision':v['primary_477'].get('metrics/precision(B)'),'recall':v['primary_477'].get('metrics/recall(B)'),'epochs':v['epochs']} for k,v in arms.items()}).T)
print(json.dumps(summary,indent=2,default=str))
""")

    nb.cells[18] = md(r"""
## Interpretation and promotion rule

The context-crop arm is promoted only if it improves primary AP50 by at least 1 percentage point, does not reduce AP50-95 on either fixed project cohort, and does not reduce recall by more than 1.5 points. Otherwise keep the control and document the negative result. The external source's own validation and test partitions are excluded from this comparison.

Interpret the result as a paired development ablation on a reused project validation. The intervention combines the selected external train images and the declared crop transform; it does not isolate a clinical effect. Do not tune or filter validation cases after seeing predictions.
""")

    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "dentalselfie": {"stage": "spacing_context_crop_ablation", "source_test_opened": False},
    }
    TARGET_NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, TARGET_NOTEBOOK)

    KAGGLE_STAGE.mkdir(parents=True, exist_ok=True)
    kaggle_nb = KAGGLE_STAGE / "16_70_spacing_context_crop_ablation.ipynb"
    nbformat.write(nb, kaggle_nb)
    metadata = {
        "id": "hngphc73/ds-spacing-context-crop-ablation",
        "title": "DS Spacing Context Crop Ablation",
        "code_file": kaggle_nb.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": False,
        "enable_gpu": True,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [
            "trihuynhviprovcl/dentalselfie-visible-alignment-v2/1",
            "hngphc73/ghqwdo-v2-diastema-train-only/1",
        ],
        "competition_sources": [],
        "kernel_sources": ["hngphc73/ds-spacing-iop-anatomy-transfer/2"],
    }
    (KAGGLE_STAGE / "kernel-metadata.json").write_text(__import__("json").dumps(metadata, indent=2), encoding="utf-8")
    print(TARGET_NOTEBOOK)
    print(kaggle_nb)


if __name__ == "__main__":
    main()

from pathlib import Path
import json,hashlib,zipfile,ast
import pandas as pd
import nbformat as nb
root=Path.cwd(); stage=root/'.venv/work/kaggle/gum_binary_publication';stage.mkdir(parents=True,exist_ok=True)
source=root/'data/processed/gum_binary_v1'; report=root/'docs/reports/gingivitis_binary'
manifest=report/'manifest.csv'; digest=hashlib.sha256(manifest.read_bytes()).hexdigest()
f=pd.read_csv(manifest); hashes={}
with zipfile.ZipFile(stage/'gum_binary_v1.zip','w',compression=zipfile.ZIP_STORED) as archive:
 for row in f.itertuples():
  for kind,ext,key in [('images','.jpg','image_sha256'),('labels','.txt','binary_label_sha256')]:
   path=source/kind/row.split/(row.sample_id+ext)
   sha=hashlib.sha256(path.read_bytes()).hexdigest();assert sha==getattr(row,key)
   name=path.relative_to(source).as_posix(); hashes[name]=sha;archive.write(path,name)
 archive.write(manifest,'manifest.csv')
 archive.writestr('checksums.json',json.dumps(hashes,sort_keys=True))
(stage/'dataset-metadata.json').write_text(json.dumps(dict(title='DentalSelfie Binary Gingival Inflammation v1',id='trihuynhviprovcl/dentalselfie-gum-binary-v1',licenses=[{'name':'CC-BY-4.0'}],description='Derived from Mendeley 3253gj88rr v1, CC BY 4.0. https://data.mendeley.com/datasets/3253gj88rr/1 . Merges audited mild and marked gingival regions. 747 train /160 validation images; 8759 /1871 boxes. Original locked splits preserved. No test images included. All images contain positive source annotations; independent phone and negative-image validation pending. Patient independence unverified.'),indent=2))
for scale in ['n','s']:
 p=root/f'notebooks/gingivitis/15_2{scale}_binary_gingival_yolov8{scale}.ipynb'; book=nb.read(p,4)
 cell=next(c for c in book.cells if c.cell_type=='code' and 'files=list' in c.source)
 cell.source=f'''from pathlib import Path
import torch, json, hashlib, zipfile, time
from ultralytics import YOLO
import pandas as pd
import matplotlib.pyplot as plt
# Kaggle expands uploaded ZIP datasets.
manifests=list(Path('/kaggle/input').rglob('manifest.csv'))
assert len(manifests)==1, manifests
source=manifests[0].parent
target=Path('/tmp/gum_binary_v1')
import shutil
shutil.copytree(source,target,dirs_exist_ok=True)
assert hashlib.sha256((target/'manifest.csv').read_bytes()).hexdigest()=={digest!r}
checksums=json.loads((target/'checksums.json').read_text())
for name, expected in checksums.items():
    assert hashlib.sha256((target/name).read_bytes()).hexdigest()==expected, name
assert len(checksums)==1814
(target/'data.yaml').write_text(f'path: {{target}}\\ntrain: images/train\\nval: images/val\\nnames:\\n  0: visible_gingival_inflammation\\n')
data=target/'data.yaml'
Path('/kaggle/working/input_manifest.csv').write_bytes((target/'manifest.csv').read_bytes())
print('Manifest SHA256:',{digest!r})
display(pd.read_csv(target/'manifest.csv').groupby('split').agg(images=('sample_id','size'), boxes=('positives','sum')))
assert torch.cuda.is_available()
print('Visible GPUs:',[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])'''
 index=book.cells.index(cell)
 book.cells.insert(index+1,nb.v4.new_markdown_cell('## Inspect annotations and letterboxing\nShow a seeded training image with target boxes, then the 640 aspect-preserving input. YOLO converts input pixels to floating-point values in [0,1]. Input staging verified every image and label checksum.'))
 book.cells.insert(index+2,nb.v4.new_code_cell('''from PIL import Image, ImageDraw
from ultralytics.data.augment import LetterBox
import numpy as np
path=sorted((target/'images/train').iterdir())[0]
im=Image.open(path).convert('RGB');draw=ImageDraw.Draw(im)
for line in (target/'labels/train'/(path.stem+'.txt')).read_text().splitlines():
    _,x,y,w,h=map(float,line.split())
    draw.rectangle(((x-w/2)*im.width,(y-h/2)*im.height,(x+w/2)*im.width,(y+h/2)*im.height),outline='lime',width=2)
fig,axes=plt.subplots(1,2,figsize=(12,5))
axes[0].imshow(im);axes[0].set_title('Source annotation')
axes[1].imshow(LetterBox((640,640))(image=np.asarray(im)));axes[1].set_title('640 letterbox geometry')
for ax in axes: ax.axis('off')
plt.tight_layout();plt.show()'''))
 train=next(c for c in book.cells if c.cell_type=='code' and 'model.train' in c.source)
 train.source=train.source.replace('results=model.train', '''started=time.monotonic()
def time_guard(trainer):
    if time.monotonic()-started > 5*3600:
        trainer.stop=True
        print('Five-hour guard reached; retain periodic checkpoints for continuation')
model.add_callback('on_train_epoch_end',time_guard)
results=model.train''')
 for c in book.cells:
  if c.cell_type=='markdown' and 'periodic checkpoints.' in c.source: c.source+=' A five-hour epoch-boundary guard requests a normal stop. Periodic checkpoints retain resumable training state; collect them before session expiry.'
 nb.validate(book)
 for c in book.cells:
  if c.cell_type=='code' and not c.source.startswith('%'):ast.parse(c.source)
 nb.write(book,p)
 kernel=root/f'.venv/work/kaggle/gum_binary_v8{scale}';kernel.mkdir(exist_ok=True)
 nb.write(book,kernel/p.name)
 (kernel/'kernel-metadata.json').write_text(json.dumps(dict(id=f'trihuynhviprovcl/dentalselfie-gum-binary-v8{scale}-20261117',title=f'DentalSelfie Gum Binary V8{scale} 20261117',code_file=p.name,language='python',kernel_type='notebook',is_private=False,enable_gpu=True,enable_internet=True,machine_shape='NvidiaTeslaT4',dataset_sources=['trihuynhviprovcl/dentalselfie-gum-binary-v1/1'],competition_sources=[],kernel_sources=[]),indent=2))
(report/'publication_preflight.json').write_text(json.dumps(dict(manifest_sha256=digest,files=len(hashes),archive_sha256=hashlib.sha256((stage/'gum_binary_v1.zip').read_bytes()).hexdigest()),indent=2))
print('Verified',len(hashes),'files; archive MB',(stage/'gum_binary_v1.zip').stat().st_size/1e6)

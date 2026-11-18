"""Create two self-contained, presentation-ready DINOv3 research notebooks."""
from pathlib import Path
import textwrap

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = 'facebook/dinov3-vits16-pretrain-lvd1689m'
MODEL_REVISION = '114c1379950215c8b35dfcd4e90a5c251dde0d32'
MANIFEST_SHA = 'b9feaaceb4b9ea94ad540b553927ffb7f67e731098f7a78c770941804cbb4f06'

RUNTIME_SETUP = '''
# Install only the HF stack into a session-local directory, not global site-packages.
RUNTIME_DIR=Path('/kaggle/temp/dentalselfie_hf_runtime')
RUNTIME_PACKAGES=['transformers==4.57.1','huggingface-hub==0.36.2',
                  'tokenizers==0.22.1','safetensors==0.6.2']
if RUN_EXPERIMENT:
    import importlib
    subprocess.run([sys.executable,'-m','pip','install','--quiet','--no-deps','--upgrade',
                    '--target',str(RUNTIME_DIR),*RUNTIME_PACKAGES],check=True)
    sys.path.insert(0,str(RUNTIME_DIR))
    os.environ['PYTHONPATH']=str(RUNTIME_DIR)+os.pathsep+os.environ.get('PYTHONPATH','')
    importlib.invalidate_caches()
    import transformers,huggingface_hub,tokenizers,safetensors
    for module in [transformers,huggingface_hub,tokenizers,safetensors]:
        assert Path(module.__file__).resolve().is_relative_to(RUNTIME_DIR.resolve()), module.__name__
    assert transformers.__version__=='4.57.1' and huggingface_hub.__version__=='0.36.2'
    CONFIG['runtime_packages']=RUNTIME_PACKAGES
    display(pd.DataFrame([{'package':m.__name__,'version':m.__version__,'path':m.__file__}
                          for m in [transformers,huggingface_hub,tokenizers,safetensors]]))
else:
    print('HF runtime installation is disabled in the source preview.')
'''


def cell_pair(cells, explanation, code):
    cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(explanation).strip()))
    cells.append(nbf.v4.new_code_cell(textwrap.dedent(code).strip()))


def save(name, title, cells):
    path = ROOT / 'notebooks' / name
    if path.exists():
        previous=nbf.read(path,as_version=4)
        if any(c.get('outputs') or c.get('execution_count') is not None
               for c in previous.cells if c.cell_type=='code'):
            raise FileExistsError(f'Refusing to overwrite executed notebook {path}')
    flow = ('```mermaid\nflowchart LR\n'
            'A[Locked raw ROI data] --> B[Official DINOv3 encoder]\n'
            'B --> C[Declared representation / adaptation]\n'
            'C --> D[Train-only fitting and validation selection]\n'
            'D --> E[Per-class metrics and saved evidence]\n```')
    book = nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell(title + '\n\n' + flow), *cells],
                               metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                                         'language_info': {'name': 'python', 'version': '3.11'}})
    nbf.validate(book)
    nbf.write(book, path)
    print('Created', path.name, 'with', len(book.cells), 'cells')


def notebook_111():
    cells = []
    cell_pair(cells, '''
## 0. Load the experiment tools

These imports support image inspection, feature extraction, train-only scaling, validation metrics, plots, and output checks. The notebook runs on Kaggle T4 x2; the local machine only prepares the source.
''','''
from pathlib import Path
import os,sys,subprocess,hashlib,json,time,platform
import numpy as np,pandas as pd,matplotlib.pyplot as plt
from PIL import Image
from IPython.display import display,Markdown
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,precision_recall_fscore_support
import joblib,torch
def save_json(path,value):Path(path).write_text(json.dumps(value,indent=2),encoding='utf-8')
''')
    cell_pair(cells, '''
## 1. Define a reproducible, smoke-gated run

The same frozen encoder produces two readouts. `CLS` summarizes the crop; `CLS + mean patch` appends the average of spatial patch tokens. No encoder weights are updated. Kaggle runs use the predeclared Hugging Face revision and raw ROI dataset version 1. Test data is excluded.
''', f'''
RUN_EXPERIMENT = False  # The Kaggle staging script enables this only for execution.
PHASE = 'smoke'         # Staging selects smoke first, then full after review.
SEED = 42
MODEL_ID = {MODEL_ID!r}
MODEL_REVISION = {MODEL_REVISION!r}
EXPECTED_MANIFEST_SHA256 = {MANIFEST_SHA!r}
DATASET_VERSION = 'trihuynhviprovcl/dentalselfie-gum-raw-roi-v1/1'
OUTPUT = Path('/kaggle/working/dentalselfie-11-1')
TEMP = Path('/kaggle/temp/dentalselfie-11-1')
CONFIG = {{'image_size':224, 'patch_size':16, 'batch_size':64,
          'c_grid':[0.01,0.1,1.0,10.0], 'seed':SEED, 'phase':PHASE,
          'model_id':MODEL_ID, 'model_revision':MODEL_REVISION,
          'manifest_sha256':EXPECTED_MANIFEST_SHA256}}
display(pd.Series(CONFIG).to_frame('fixed setting'))
print('Run enabled:', RUN_EXPERIMENT)
''')
    cell_pair(cells, '''
## 2. Install and record the DINOv3 interface

Use the Transformers implementation already used for the successful probe. The four pinned Hugging Face packages are installed without dependencies into a session-local directory. Existing Kaggle Torch, NumPy and scientific libraries are reused. This avoids changing the global Hub version required by unrelated Gradio/Diffusers packages. Display and assert the actual import paths; no access token is involved in installation.
''', RUNTIME_SETUP + '''
import torch, torchvision
print({'python':platform.python_version(),'torch':torch.__version__,
       'torchvision':torchvision.__version__,'cuda':torch.version.cuda,
       'gpu_count':torch.cuda.device_count(),
       'gpu_names':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]})
''')
    cell_pair(cells, '''
## 3. Find and verify the immutable input manifest

The raw-source manifest locks the 10,889 crop identities and train/validation assignments. Hash the manifest and every requested image before using it. Do not discover a test directory or include test rows.
''', '''
def sha256_file(path):
    digest=hashlib.sha256()
    with open(path,'rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()

manifests=list(Path('/kaggle/input').rglob('manifest.csv')) if Path('/kaggle/input').exists() else []
matched=[path for path in manifests if sha256_file(path)==EXPECTED_MANIFEST_SHA256]
if RUN_EXPERIMENT and len(matched)!=1:
    raise RuntimeError(f'Expected exactly one pinned raw ROI manifest, found {len(matched)}')
if matched:
    DATA_ROOT=matched[0].parent
    frame=pd.read_csv(matched[0])
    assert set(frame.split)=={'train','val'} and frame.crop_id.is_unique
    assert frame.groupby('group_id').split.nunique().max()==1
    assert (frame.groupby(['split','class_id']).size().unstack(fill_value=0).index==['train','val']).all()
    assert frame.class_id.isin([0,1,2]).all()
    print('Manifest:',len(frame),'rows | hash:',sha256_file(matched[0]),'| test rows: 0')
    display(frame.groupby(['split','class_name']).agg(crops=('crop_id','size'),groups=('group_id','nunique')))
else:
    DATA_ROOT=None
    print('Local preview only. Attach the pinned raw ROI input in Kaggle to execute.')
''')
    cell_pair(cells, '''
## 4. Inspect the exact 224-pixel crops

This panel shows the model input as stored. The same deterministic resize and ImageNet normalization are applied to both representations. No additional center crop or context image is introduced.
''', '''
if DATA_ROOT is not None:
    examples=frame.groupby('class_id',group_keys=False).head(2)
    fig,axes=plt.subplots(3,2,figsize=(7,9))
    for ax,row in zip(axes.flat,examples.itertuples()):
        with Image.open(DATA_ROOT/row.local_path) as image:ax.imshow(image.convert('RGB'))
        ax.set_title(f'{row.class_name} | {row.crop_id}',fontsize=8);ax.axis('off')
    fig.tight_layout();plt.show()
else:
    display(Markdown('Preview appears after the Kaggle input is attached.'))
''')
    cell_pair(cells, '''
## 5. Load official weights without exposing the secret

The repository is gated. Retrieve `HF_TOKEN` through Kaggle Secrets; never print, save, or embed it in a notebook. If the secret is not attached or access is not granted, stop with an actionable error. Record only the immutable model revision and model configuration.
''', '''
def get_hf_token():
    token=os.environ.get('HF_TOKEN')
    if token:return token
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret('HF_TOKEN')
    except Exception as exc:
        raise RuntimeError('Attach the HF_TOKEN secret to this Kaggle notebook and accept official DINOv3 access.') from exc

encoder=None
if RUN_EXPERIMENT:
    from transformers import AutoModel
    token=get_hf_token()
    encoder=AutoModel.from_pretrained(MODEL_ID,revision=MODEL_REVISION,token=token,
                                      use_safetensors=True,trust_remote_code=False).eval().to('cuda:0')
    del token
    assert encoder.config.patch_size==16 and encoder.config.hidden_size==384
    print({'resolved_revision':encoder.config._commit_hash,'hidden_size':encoder.config.hidden_size,
           'patch_size':encoder.config.patch_size,'register_tokens':encoder.config.num_register_tokens,
           'transformer_blocks':encoder.config.num_hidden_layers})
    assert encoder.config._commit_hash==MODEL_REVISION
''')
    cell_pair(cells, '''
## 6. Recreate the image preprocessing and patch pooling

DINOv3 outputs one CLS token, register tokens, then spatial patch tokens. Exclude the register tokens before averaging patches. At 224 pixels and patch size 16, the expected grid is 14 x 14. The second readout concatenates the 384-dimensional CLS and 384-dimensional mean patch vector.
''', '''
transform=transforms.Compose([
    transforms.Resize((224,224),interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])
def pool_dinov3_tokens(outputs,model,pixel_values):
    tokens=outputs.last_hidden_state
    batch,_,height,width=pixel_values.shape
    patch_size=model.config.patch_size
    patch_h,patch_w=height//patch_size,width//patch_size
    prefix=1+model.config.num_register_tokens
    expected=prefix+patch_h*patch_w
    assert tokens.shape==(batch,expected,model.config.hidden_size),tokens.shape
    cls=tokens[:,0,:]
    patches=tokens[:,prefix:,:]
    assert patches.shape[1]==patch_h*patch_w
    patch_grid=patches.unflatten(1,(patch_h,patch_w))
    patch_mean=patch_grid.mean(dim=(1,2))
    combined=torch.cat([cls,patch_mean],dim=1)
    return cls,patch_mean,combined,patch_grid
print('224 / 16 = 14 x 14 spatial patches; register tokens excluded from pooling.')
''')
    cell_pair(cells, '''
## 7. Extract deterministic features from train and validation only

Full runs verify each crop checksum, keep source groups inside their locked split, and cache feature arrays under `/kaggle/temp`. Smoke uses a balanced, deterministic sample to check every class and code path; smoke metrics are diagnostic only.
''', '''
def selected_rows(split):
    rows=frame[frame.split.eq(split)].copy()
    if PHASE=='smoke':
        rows=rows.groupby('class_id',group_keys=False).head(16).copy()
    return rows.reset_index(drop=True)

class CropDataset(Dataset):
    def __init__(self,rows,verify):self.rows=rows;self.verify=verify
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        row=self.rows.iloc[index];path=DATA_ROOT/row.local_path
        if self.verify and sha256_file(path)!=row.local_sha256:raise ValueError(f'Checksum mismatch: {row.crop_id}')
        with Image.open(path) as image:x=transform(image.convert('RGB'))
        return x,int(row.class_id),str(row.crop_id)

def extract_split(split):
    rows=selected_rows(split)
    loader=DataLoader(CropDataset(rows,verify=(PHASE=='full')),batch_size=CONFIG['batch_size'],
                      shuffle=False,num_workers=2,pin_memory=True)
    outputs=[];labels=[];ids=[];patch_example=None
    started=time.monotonic()
    with torch.inference_mode():
        for step,(images,y,crop_ids) in enumerate(loader):
            images=images.cuda(non_blocking=True)
            with torch.autocast('cuda',dtype=torch.float16):encoded=encoder(pixel_values=images)
            cls,patch_mean,combined,patch_grid=pool_dinov3_tokens(encoded,encoder,images)
            outputs.append(torch.cat([cls.float(),patch_mean.float(),combined.float()],dim=1).cpu().numpy())
            labels.extend(y.tolist());ids.extend(crop_ids)
            if patch_example is None:patch_example=(images[0].cpu(),patch_grid[0].float().norm(dim=-1).cpu())
            if step%20==0:print(f'{split}: {step+1}/{len(loader)} batches',flush=True)
    features=np.concatenate(outputs)
    assert len(features)==len(rows) and set(labels)=={0,1,2}
    print(split,'rows',len(rows),'feature extraction seconds',round(time.monotonic()-started,1))
    return rows,features,np.asarray(labels),ids,patch_example

if RUN_EXPERIMENT:
    TEMP.mkdir(parents=True,exist_ok=True)
    extracted={split:extract_split(split) for split in ['train','val']}
    train_rows,train_all,y_train,id_train,_=extracted['train']
    val_rows,val_all,y_val,id_val,patch_example=extracted['val']
    print('Output blocks: CLS 384 + patch mean 384; concatenated 768.')
''')
    cell_pair(cells, '''
## 8. Visualize patch features as a representation diagnostic

The patch norm map shows feature magnitude only. It is not an attention map, saliency map, segmentation, or clinical explanation. It is included to make the 14 x 14 spatial feature grid tangible.
''', '''
if RUN_EXPERIMENT:
    image_tensor,feature_map=patch_example
    image=(image_tensor*torch.tensor([.229,.224,.225])[:,None,None]+
           torch.tensor([.485,.456,.406])[:,None,None]).clamp(0,1).permute(1,2,0).numpy()
    fig,axes=plt.subplots(1,2,figsize=(9,4))
    axes[0].imshow(image);axes[0].set_title('224 x 224 crop');axes[0].axis('off')
    axes[1].imshow(image);axes[1].imshow(feature_map.numpy(),alpha=.55,cmap='magma',
        extent=(0,224,224,0),interpolation='nearest');axes[1].set_title('Patch feature norm (not saliency)');axes[1].axis('off')
    fig.tight_layout();plt.show()
''')
    cell_pair(cells, '''
## 9. Fit matched linear probes and select regularization on validation

Both readouts use the same train-fitted StandardScaler, LogisticRegression implementation, four-value C grid, seed, rows, and validation selection metric. Validation is repeatedly reused elsewhere in the project, so these results are exploratory. Test data stays unopened.
''', '''
def metric_table(y,pred):
    cm=confusion_matrix(y,pred,labels=[0,1,2])
    precision,recall,f1,support=precision_recall_fscore_support(y,pred,labels=[0,1,2],zero_division=0)
    return {'accuracy':float(accuracy_score(y,pred)),'macro_f1':float(f1.mean()),
            'balanced_accuracy':float(recall.mean()),'class_precision':precision.tolist(),
            'class_recall':recall.tolist(),'class_f1':f1.tolist(),'support':support.tolist(),
            'confusion_matrix':cm.tolist()}

if RUN_EXPERIMENT:
    OUTPUT.mkdir(parents=True,exist_ok=True)
    feature_specs={'CLS':(0,384),'CLS_plus_patch_mean':(768,1536)}
    results=[];predictions={};grid_rows=[]
    for name,(start,end) in feature_specs.items():
        train_features=train_all[:,start:end];val_features=val_all[:,start:end]
        scaler=StandardScaler().fit(train_features)
        train_x=scaler.transform(train_features);val_x=scaler.transform(val_features)
        grid=[];best=None
        for c in CONFIG['c_grid']:
            classifier=LogisticRegression(C=c,max_iter=1000,solver='lbfgs',random_state=SEED)
            classifier.fit(train_x,y_train);pred=classifier.predict(val_x)
            metrics=metric_table(y_val,pred);grid.append({'feature':name,'C':c,**metrics})
            grid_rows.append({'feature':name,'C':c,**metrics})
            if best is None or metrics['macro_f1']>best['metrics']['macro_f1']:
                best={'classifier':classifier,'scaler':scaler,'C':c,'metrics':metrics,'pred':pred}
        best['metrics'].update(feature=name,selected_C=best['C'],feature_dim=end-start,
            model_id=MODEL_ID,model_revision=MODEL_REVISION,manifest_sha256=EXPECTED_MANIFEST_SHA256,
            seed=SEED,phase=PHASE,test_images_opened=0)
        results.append(best['metrics']);predictions[name]=best['pred']
        pd.DataFrame({'crop_id':id_val,'true_class':y_val,'predicted_class':best['pred'],
          'probabilities':[json.dumps(p.tolist()) for p in best['classifier'].predict_proba(val_x)]}).to_csv(
          OUTPUT/f'{name}_validation_predictions.csv',index=False)
        save_json(OUTPUT/f'{name}_metrics.json',best['metrics'])
        joblib.dump({'scaler':best['scaler'],'classifier':best['classifier']},OUTPUT/f'{name}_probe.joblib')
    grid_frame=pd.DataFrame(grid_rows);grid_frame.to_csv(OUTPUT/'probe_grid.csv',index=False)
    result_frame=pd.DataFrame([{k:v for k,v in row.items() if not isinstance(v,list)} for row in results])
    result_frame.to_csv(OUTPUT/'comparison.csv',index=False)
    display(result_frame.style.format({'accuracy':'{:.2%}','macro_f1':'{:.2%}','balanced_accuracy':'{:.2%}'}))
    fig,ax=plt.subplots(figsize=(8,4))
    for name,group in grid_frame.groupby('feature'):
        ax.plot(group.C,group.macro_f1,marker='o',label=name)
    ax.set_xscale('log');ax.set(xlabel='C (log scale)',ylabel='Validation macro-F1',ylim=(0,1),title='Matched frozen-feature probe');ax.legend();fig.tight_layout();plt.show()
''')
    cell_pair(cells, '''
## 10. Compare per-class behavior and save the experiment record

Read the confusion matrices and recall of each class alongside accuracy and macro-F1. A higher aggregate score is not enough if `no-visible` behavior collapses. The probe grid, predictions, head, configuration, and resolved DINO revision are saved for collection.
''', '''
if RUN_EXPERIMENT:
    for result in results:
        display(Markdown('**'+result['feature']+' | validation confusion matrix**'))
        cm=np.asarray(result['confusion_matrix'])
        fig,ax=plt.subplots(figsize=(4,3));display_matrix=ax.imshow(cm,cmap='Blues')
        ax.set(xticks=[0,1,2],yticks=[0,1,2],xlabel='Predicted',ylabel='Reference',title=result['feature'])
        for (i,j),value in np.ndenumerate(cm):ax.text(j,i,int(value),ha='center',va='center')
        fig.colorbar(display_matrix,ax=ax);fig.tight_layout();plt.show()
        display(pd.DataFrame({'class':['no-visible','mild','marked'],
                              'precision':result['class_precision'],'recall':result['class_recall'],
                              'F1':result['class_f1']}))
    save_json(OUTPUT/'run_config.json',CONFIG)
    print('Saved:',sorted(p.name for p in OUTPUT.iterdir()))
else:
    display(Markdown('Execution is disabled in the published source; the Kaggle runner enables a smoke pass before the full feature extraction.'))
''')
    save('11_1_dinov3_patch_probe.ipynb',
         '# 11.1 DINOv3 spatial patch pooling\n\nFrozen representation ablation: CLS token versus CLS concatenated with mean spatial patch features.\n\nMethod references: [DINOv3 paper](https://arxiv.org/abs/2508.10104), [official implementation](https://github.com/facebookresearch/dinov3), and [Transformers DINOv3 interface](https://huggingface.co/docs/transformers/en/model_doc/dinov3). Patch mean pooling is our declared readout, not a reproduced dental method or a saliency explanation.',cells)


def notebook_112():
    cells=[]
    cell_pair(cells, '''
## 0. Load the experiment tools

These imports support the fixed split, DINOv3 classification head, two GPU-isolated workers, validation metrics, checkpoints, and plots. The model weights and gated credentials stay out of collected outputs.
''','''
from pathlib import Path
import os,sys,subprocess,hashlib,json,time,platform,random
import numpy as np,pandas as pd,matplotlib.pyplot as plt
from PIL import Image
from IPython.display import display,Markdown
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score,confusion_matrix,precision_recall_fscore_support,balanced_accuracy_score
import torch
def save_json(path,value):Path(path).write_text(json.dumps(value,indent=2),encoding='utf-8')
''')
    cell_pair(cells, '''
## 1. State the matched adaptation experiment

Both arms use the same official DINOv3 ViT-S/16 CLS representation and the same randomly initialized linear 3-class head. The control freezes the entire backbone. The adaptation arm also trains the final two transformer blocks and final LayerNorm. It does not alter the embeddings or other blocks.
''',f'''
RUN_EXPERIMENT = False
PHASE = 'smoke'
SEED = 42
MODEL_ID = {MODEL_ID!r}
MODEL_REVISION = {MODEL_REVISION!r}
EXPECTED_MANIFEST_SHA256 = {MANIFEST_SHA!r}
OUTPUT = Path('/kaggle/working/dentalselfie-11-2')
TEMP = Path('/kaggle/temp/dentalselfie-11-2')
CONFIG = {{'image_size':224,'patch_size':16,'batch_size':32,'max_epochs':30,'patience':8,
          'head_lr':3e-4,'encoder_lr':1e-5,'weight_decay':5e-4,'seed':SEED,
          'worker_seconds':3600,'phase':PHASE,'model_id':MODEL_ID,
          'model_revision':MODEL_REVISION,'manifest_sha256':EXPECTED_MANIFEST_SHA256,
          'trials':['frozen_CE','last2_blocks_norm_CE']}}
display(pd.Series(CONFIG).to_frame('fixed setting'))
''')
    cell_pair(cells, '''
## 2. Install and inspect the fixed GPU environment

Use the same pinned HF stack and DINOv3 revision in both arms. Install only the four HF packages into a session-local directory and inherit that import path in each GPU worker. Global Gradio/Diffusers dependencies stay untouched. Each worker sees one assigned T4, with isolated RNG and outputs.
''',RUNTIME_SETUP + '''
import torch,torchvision
if RUN_EXPERIMENT:
    assert torch.cuda.is_available() and torch.cuda.device_count()==2
    subprocess.run(['nvidia-smi'],check=True)
print({'python':platform.python_version(),'torch':torch.__version__,'torchvision':torchvision.__version__,
       'gpu_count':torch.cuda.device_count(),'gpus':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]})
''')
    cell_pair(cells, '''
## 3. Verify the pinned raw crop dataset and its class distribution

Only the fixed train and validation split are loaded. Every original source group stays wholly in one split. The smoke sample is balanced and deterministic; the full experiment uses every training crop and every validation crop.
''','''
def sha256_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
paths=list(Path('/kaggle/input').rglob('manifest.csv')) if Path('/kaggle/input').exists() else []
matches=[p for p in paths if sha256_file(p)==EXPECTED_MANIFEST_SHA256]
if RUN_EXPERIMENT and len(matches)!=1:raise RuntimeError(f'Expected one pinned raw ROI manifest, got {len(matches)}')
if matches:
    DATA_ROOT=matches[0].parent;frame=pd.read_csv(matches[0])
    assert set(frame.split)=={'train','val'} and frame.crop_id.is_unique
    assert frame.groupby('group_id').split.nunique().max()==1
    assert frame.class_id.isin([0,1,2]).all()
    display(frame.groupby(['split','class_name']).agg(crops=('crop_id','size'),groups=('group_id','nunique')))
    print('Pinned manifest verified:',sha256_file(matches[0]),'test rows: 0')
    if RUN_EXPERIMENT and PHASE=='full':
        started=time.monotonic()
        for row in frame.itertuples():
            if sha256_file(DATA_ROOT/row.local_path)!=row.local_sha256:raise ValueError(f'Crop hash mismatch: {row.crop_id}')
        print('Verified all',len(frame),'crop files in',round(time.monotonic()-started,1),'seconds')
else:
    DATA_ROOT=None;print('Local preview only; attach raw ROI dataset v1 to run on Kaggle.')
''')
    cell_pair(cells, '''
## 4. Show the crops and match the baseline preprocessing

Both arms see RGB crops resized to 224 x 224 with bilinear interpolation and ImageNet normalization. Training adds the same mild horizontal flip and +/-7 degree rotation in both arms. No context branch or class weighting is mixed in.
''','''
train_transform=transforms.Compose([transforms.Resize((224,224),interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.RandomHorizontalFlip(),transforms.RandomRotation(7,interpolation=transforms.InterpolationMode.BILINEAR,fill=(128,128,128)),
    transforms.ToTensor(),transforms.Normalize((.485,.456,.406),(.229,.224,.225))])
eval_transform=transforms.Compose([transforms.Resize((224,224),interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),transforms.Normalize((.485,.456,.406),(.229,.224,.225))])
if DATA_ROOT is not None:
    sample=frame.groupby('class_id',group_keys=False).head(2);fig,axes=plt.subplots(3,2,figsize=(7,9))
    for ax,row in zip(axes.flat,sample.itertuples()):
        with Image.open(DATA_ROOT/row.local_path) as image:ax.imshow(image.convert('RGB'))
        ax.set_title(row.class_name,fontsize=8);ax.axis('off')
    fig.tight_layout();plt.show()
''')
    cell_pair(cells, '''
## 5. Load DINOv3 using the existing secret channel

Read `HF_TOKEN` once in the notebook process and fail before launching either worker if the secret is unavailable. Workers inherit it only through their in-memory process environment; it is never written to configuration, source or outputs. Model loading is pinned to the same repository revision in both workers. Publish using QUICK_SAVE, then launch through the Kaggle editor with the secret attached.
''','''
def get_hf_token():
    token=os.environ.get('HF_TOKEN')
    if token:return token
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret('HF_TOKEN')
    except Exception as exc:raise RuntimeError('Attach HF_TOKEN to this Kaggle notebook and accept official DINOv3 access.') from exc

HF_RUNTIME_TOKEN=get_hf_token() if RUN_EXPERIMENT else None
if RUN_EXPERIMENT:
    assert HF_RUNTIME_TOKEN, 'HF_TOKEN is empty; attach it in the Kaggle editor.'
    print('HF_TOKEN preflight passed; token value is never logged.')
print('Official model:',MODEL_ID,'| pinned revision:',MODEL_REVISION)
''')
    cell_pair(cells, '''
## 6. Define the classifier and verify trainable parameters

The model is the CLS token from the final normalized DINOv3 sequence followed by one linear layer. In Transformers 4.57.1 the ViT blocks are `backbone.layer`; the adapted arm unfreezes block indices 10 and 11 plus final normalization. Parameter counts, gradient routing, and exact names are checked before interpreting a run.
''','''
import torch.nn as nn
try:
    from transformers import AutoModel
except ImportError:
    AutoModel=None  # Source-only version; the bounded run installs the pinned package first.

class DINOv3Classifier(nn.Module):
    def __init__(self,backbone):
        super().__init__();self.backbone=backbone
        self.head=nn.Linear(backbone.config.hidden_size,3)
    def forward(self,pixel_values,backbone_grad):
        if backbone_grad:outputs=self.backbone(pixel_values=pixel_values)
        else:
            with torch.no_grad():outputs=self.backbone(pixel_values=pixel_values)
        return self.head(outputs.last_hidden_state[:,0,:].float())

def configure_trainable(model,adapt):
    for parameter in model.parameters():parameter.requires_grad=False
    if adapt:
        blocks=model.backbone.layer
        assert len(blocks)==model.backbone.config.num_hidden_layers==12
        for block in blocks[-2:]:
            for parameter in block.parameters():parameter.requires_grad=True
        for parameter in model.backbone.norm.parameters():parameter.requires_grad=True
    for parameter in model.head.parameters():parameter.requires_grad=True
    names=[name for name,p in model.named_parameters() if p.requires_grad]
    if adapt:
        assert any('backbone.layer.10.' in n for n in names) and any('backbone.layer.11.' in n for n in names)
        assert any(n.startswith('backbone.norm.') for n in names)
        assert all(('backbone.layer.10.' in n or 'backbone.layer.11.' in n or n.startswith('backbone.norm.') or n.startswith('head.')) for n in names)
    else:assert all(n.startswith('head.') for n in names)
    return names

print('Classifier head:',384,'-> 3 logits; patched backbone structure will be checked against the downloaded model.')
''')
    # Worker source cells are real, readable Python sections executed in this notebook and
    # reconstructed verbatim into a subprocess script to isolate GPU RNG/state.
    worker_sections = [
('imports and deterministic data pipeline', '''# WORKER_SECTION: imports_and_data
import os, json, time, random, hashlib, argparse
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, balanced_accuracy_score
try:
    from transformers import AutoModel
except ImportError:
    AutoModel=None  # Source-only version; the bounded run installs the pinned package first.

MODEL_ID='facebook/dinov3-vits16-pretrain-lvd1689m'
MODEL_REVISION='114c1379950215c8b35dfcd4e90a5c251dde0d32'
MANIFEST_SHA='b9feaaceb4b9ea94ad540b553927ffb7f67e731098f7a78c770941804cbb4f06'
IMAGE_TRANSFORM=transforms.Compose([transforms.Resize((224,224),interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),transforms.Normalize((.485,.456,.406),(.229,.224,.225))])
TRAIN_TRANSFORM=transforms.Compose([transforms.Resize((224,224),interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.RandomHorizontalFlip(),transforms.RandomRotation(7,interpolation=transforms.InterpolationMode.BILINEAR,fill=(128,128,128)),
    transforms.ToTensor(),transforms.Normalize((.485,.456,.406),(.229,.224,.225))])
class CropDataset(Dataset):
    def __init__(self,frame,root,training,verify):self.frame=frame.reset_index(drop=True);self.root=Path(root);self.transform=TRAIN_TRANSFORM if training else IMAGE_TRANSFORM;self.verify=verify
    def __len__(self):return len(self.frame)
    def __getitem__(self,index):
        row=self.frame.iloc[index];path=self.root/row.local_path
        if self.verify and hashlib.sha256(path.read_bytes()).hexdigest()!=row.local_sha256:raise ValueError(f'Image hash mismatch: {row.crop_id}')
        with Image.open(path) as image:x=self.transform(image.convert('RGB'))
        return x,int(row.class_id),str(row.crop_id)

class Classifier(nn.Module):
    def __init__(self,backbone):super().__init__();self.backbone=backbone;self.head=nn.Linear(backbone.config.hidden_size,3)
    def forward(self,pixel_values,adapt):
        if adapt:out=self.backbone(pixel_values=pixel_values)
        else:
            with torch.no_grad():out=self.backbone(pixel_values=pixel_values)
        return self.head(out.last_hidden_state[:,0,:].float())

def metrics(y,p):
    cm=confusion_matrix(y,p,labels=[0,1,2]);precision,recall,f1,support=precision_recall_fscore_support(y,p,labels=[0,1,2],zero_division=0)
    return {'accuracy':float(accuracy_score(y,p)),'macro_f1':float(f1.mean()),'balanced_accuracy':float(recall.mean()),
            'class_precision':precision.tolist(),'class_recall':recall.tolist(),'class_f1':f1.tolist(),'support':support.tolist(),'confusion_matrix':cm.tolist()}

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.set_num_threads(2)

def token_from_secret():
    value=os.environ.get('HF_TOKEN')
    if value:return value
    from kaggle_secrets import UserSecretsClient
    return UserSecretsClient().get_secret('HF_TOKEN')
'''),
('checkpoint and validation loops', '''# WORKER_SECTION: training_and_checkpointing
def run_trial(trial,config):
    seed_all(config['seed']);adapt=trial=='last2_blocks_norm_CE'
    root=Path(config['data_root']);out=Path(config['output_root'])/trial;out.mkdir(parents=True,exist_ok=True)
    temp=Path(config['temp_root'])/trial;temp.mkdir(parents=True,exist_ok=True)
    frame=pd.read_csv(config['manifest_path']);train=frame[frame.split.eq('train')].copy();valid=frame[frame.split.eq('val')].copy()
    if config['phase']=='smoke':
        train=train.groupby('class_id',group_keys=False).head(16);valid=valid.groupby('class_id',group_keys=False).head(16)
    assert set(train.class_id)==set(valid.class_id)=={0,1,2}
    token=token_from_secret()
    backbone=AutoModel.from_pretrained(MODEL_ID,revision=MODEL_REVISION,token=token,use_safetensors=True,trust_remote_code=False).to('cuda:0')
    del token
    assert backbone.config._commit_hash==MODEL_REVISION and backbone.config.num_hidden_layers==12
    model=Classifier(backbone).to('cuda:0')
    for parameter in model.parameters():parameter.requires_grad=False
    if adapt:
        for block in model.backbone.layer[-2:]:
            for parameter in block.parameters():parameter.requires_grad=True
        for parameter in model.backbone.norm.parameters():parameter.requires_grad=True
    for parameter in model.head.parameters():parameter.requires_grad=True
    trainable={n:p for n,p in model.named_parameters() if p.requires_grad}
    if adapt:assert any('backbone.layer.10.' in n for n in trainable) and any('backbone.layer.11.' in n for n in trainable)
    else:assert set(trainable)=={'head.weight','head.bias'}
    trainable_count=sum(p.numel() for p in trainable.values());total_count=sum(p.numel() for p in model.parameters())
    groups=[]
    enc=[p for n,p in trainable.items() if n.startswith('backbone.')]
    head=[p for n,p in trainable.items() if n.startswith('head.')]
    if enc:groups.append({'params':enc,'lr':config['encoder_lr']})
    groups.append({'params':head,'lr':config['head_lr']})
    optimizer=torch.optim.AdamW(groups,weight_decay=config['weight_decay'])
    scaler=torch.amp.GradScaler('cuda')
    train_generator=torch.Generator().manual_seed(config['seed'])
    train_loader=DataLoader(CropDataset(train,root,True,False),batch_size=config['batch_size'],shuffle=True,num_workers=2,pin_memory=True,
       generator=train_generator,worker_init_fn=lambda wid:np.random.seed(torch.initial_seed()%2**32))
    val_loader=DataLoader(CropDataset(valid,root,False,False),batch_size=64,shuffle=False,num_workers=2,pin_memory=True)
    signature=hashlib.sha256(json.dumps({k:v for k,v in config.items() if k not in ['output_root','temp_root','manifest_path','data_root'] }|{'trial':trial},sort_keys=True).encode()).hexdigest()
    resume=temp/'resume.pt';best=temp/'best.pt';history=[];best_score=-1.;stale=0;start_epoch=1
    if resume.exists() and config['phase']=='full':
        state=torch.load(resume,map_location='cpu',weights_only=False);assert state['signature']==signature
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler'])
        start_epoch=state['epoch']+1;history=state['history'];best_score=state['best_score'];stale=state['stale']
        random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng']);torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state(state['cuda_rng']);train_generator.set_state(state['loader_rng'])
    def evaluate():
        model.eval();ys=[];ps=[];probs=[];losses=[]
        with torch.inference_mode():
            for step,(x,y,_) in enumerate(val_loader):
                x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True)
                with torch.autocast('cuda',dtype=torch.float16):logits=model(x,adapt)
                loss=nn.functional.cross_entropy(logits.float(),y);prob=logits.float().softmax(1)
                ys.extend(y.cpu().tolist());ps.extend(prob.argmax(1).cpu().tolist());probs.extend(prob.cpu().tolist());losses.append(float(loss)*len(y))
                if config['phase']=='smoke' and step>=1:break
        return metrics(ys,ps),sum(losses)/len(ys),ys,ps,probs
    max_epochs=1 if config['phase']=='smoke' else config['max_epochs']
    started=time.monotonic();stop_reason='max_epochs'
    for epoch in range(start_epoch,max_epochs+1):
        if time.monotonic()-started>config['worker_seconds']:stop_reason='time_budget';break
        if config['phase']=='full' and stale>=config['patience']:stop_reason='early_stopping';break
        model.train();loss_sum=0.;seen=0
        for step,(x,y,_) in enumerate(train_loader):
            if config['phase']=='smoke' and step>=2:break
            x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True);optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):logits=model(x,adapt)
            loss=nn.functional.cross_entropy(logits.float(),y);scaler.scale(loss).backward()
            if step==0:
                assert model.head.weight.grad is not None and torch.isfinite(model.head.weight.grad).all()
                backbone_grads=[(name,p.grad) for name,p in model.backbone.named_parameters()]
                if adapt:
                    assert any(name.startswith('layer.10.') and grad is not None for name,grad in backbone_grads)
                    assert any(name.startswith('layer.11.') and grad is not None for name,grad in backbone_grads)
                    assert all(grad is None for name,grad in backbone_grads
                               if not (name.startswith('layer.10.') or name.startswith('layer.11.') or name.startswith('norm.')))
                else:assert all(grad is None for _,grad in backbone_grads)
                print('Gradient routing check passed:',trial,flush=True)
            scaler.step(optimizer);scaler.update()
            loss_sum+=float(loss.detach())*len(y);seen+=len(y)
            if step%50==0:(temp/'heartbeat.txt').write_text(str(time.time()))
        vm,vl,ys,ps,probs=evaluate();improved=vm['macro_f1']>best_score
        if improved:
            best_score=vm['macro_f1'];stale=0
            torch.save(model.state_dict(),best)
        else:stale+=1
        history.append({'epoch':epoch,'train_loss':loss_sum/seen,'val_loss':vl,'macro_f1':vm['macro_f1'],'accuracy':vm['accuracy'],
                        'trainable_parameters':trainable_count,'seconds':time.monotonic()-started})
        torch.save({'signature':signature,'epoch':epoch,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                    'scaler':scaler.state_dict(),'best_score':best_score,'stale':stale,'history':history,
                    'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),
                    'loader_rng':train_generator.get_state(),
                    'cuda_rng':torch.cuda.get_rng_state()},resume)
        pd.DataFrame(history).to_csv(out/'history.csv',index=False)
        (temp/'heartbeat.txt').write_text(str(time.time()))
        print(json.dumps({'trial':trial,**history[-1]}),flush=True)
        if config['phase']=='smoke':
            restore=torch.load(resume,map_location='cpu',weights_only=False);assert restore['signature']==signature
            model.load_state_dict(restore['model']);optimizer.load_state_dict(restore['optimizer']);scaler.load_state_dict(restore['scaler'])
            print('SMOKE_CHECKPOINT_RESTORE_OK',flush=True)
    if not best.exists():raise RuntimeError(f'No checkpoint saved for {trial}')
    model.load_state_dict(torch.load(best,map_location='cuda:0',weights_only=True));vm,vl,ys,ps,probs=evaluate()
    pred_frame=pd.DataFrame({'crop_id':valid.crop_id.tolist()[:len(ys)],'true_class':ys,'predicted_class':ps,
                             'probabilities':[json.dumps(x) for x in probs]})
    pred_frame.to_csv(out/'validation_predictions.csv',index=False)
    result={'trial':trial,'phase':config['phase'],'seed':config['seed'],'metrics':vm,'val_loss':vl,
            'epochs_completed':len(history),'selected_epoch':int(max(history,key=lambda r:r['macro_f1'])['epoch']),
            'stop_reason':stop_reason,'elapsed_seconds':time.monotonic()-started,'gpu':torch.cuda.get_device_name(0),
            'torch':torch.__version__,'transformers':__import__('transformers').__version__,
            'model_id':MODEL_ID,'model_revision':MODEL_REVISION,'manifest_sha256':MANIFEST_SHA,
            'trainable_parameters':trainable_count,'total_parameters':total_count,'trainable_names':list(trainable),
            'test_images_opened':0,'signature':signature}
    (out/'metrics.json').write_text(json.dumps(result,indent=2));return result
'''),
('isolated worker command entry point', '''# WORKER_SECTION: worker_entrypoint
def worker_main():
    parser=argparse.ArgumentParser();parser.add_argument('--trial',required=True);parser.add_argument('--config',required=True)
    args=parser.parse_args();config=json.loads(Path(args.config).read_text())
    run_trial(args.trial,config)
''')]
    for heading,source in worker_sections:
        cell_pair(cells, f'''## 7. Implementation appendix: {heading}

This readable worker section is executed as a definition cell, then embedded into the isolated one-GPU subprocess. It contains the real data, model, training, validation, and checkpoint logic used in this experiment.''', source)
    cell_pair(cells, '''
## 8. Assemble the worker source and define the bounded run

Notebook history is used only to collect the marked worker cells above. The worker source is reconstructed and hash-checked before launch. Checkpoints and temporary Hugging Face cache stay under `/kaggle/temp`; the published outputs contain metrics, histories, predictions and logs, not the gated base model weights.
''','''
def assemble_worker():
    history=get_ipython().history_manager.input_hist_raw
    sections=[source for source in history if source.lstrip().startswith('# WORKER_SECTION:')]
    assert len(sections)==3,(len(sections),'expected three marked worker sections')
    return '\\n\\n'.join(sections)+'\\n\\nif __name__=="__main__":\\n    worker_main()\\n'

WORKER_SOURCE=assemble_worker()
WORKER_PATH=Path('/kaggle/working/dinov3_partial_worker.py')
if RUN_EXPERIMENT:
    WORKER_PATH.write_text(WORKER_SOURCE)
    print('Worker source SHA256:',hashlib.sha256(WORKER_SOURCE.encode()).hexdigest(),
          '| source sections:',WORKER_SOURCE.count('# WORKER_SECTION:'))

if RUN_EXPERIMENT:
    candidates=[p for p in Path('/kaggle/input').rglob('manifest.csv') if sha256_file(p)==EXPECTED_MANIFEST_SHA256]
    assert len(candidates)==1
    manifest_path=candidates[0];DATA_ROOT=manifest_path.parent
    CONFIG.update({'data_root':str(DATA_ROOT),'manifest_path':str(manifest_path),
                   'output_root':'/kaggle/working/dentalselfie-11-2','temp_root':'/kaggle/temp/dentalselfie-11-2'})
    OUTPUT.mkdir(parents=True,exist_ok=True);TEMP.mkdir(parents=True,exist_ok=True)
    config_path=TEMP/'run_config.json';config_path.write_text(json.dumps(CONFIG,indent=2))
''')
    cell_pair(cells, '''
## 9. Run one isolated process per GPU with a no-progress watchdog

Both arms receive separate CUDA visibility, process RNG and log files. Smoke trains one epoch over two balanced batches and checks checkpoint restoration. The full phase has a 60-minute per-worker budget, 8-epoch patience, and a 2-hour Kaggle session limit.
''','''
def launch_workers():
    specs=[('frozen_CE',0),('last2_blocks_norm_CE',1)]
    processes={};logs={};started=time.monotonic()
    try:
        for trial,gpu in specs:
            env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu);env['HF_HOME']='/kaggle/temp/huggingface'
            env['HF_TOKEN']=HF_RUNTIME_TOKEN
            log=open(OUTPUT/f'{trial}.log','w',encoding='utf-8');logs[trial]=log
            process=subprocess.Popen([sys.executable,str(WORKER_PATH),'--trial',trial,'--config',str(config_path)],
                                     env=env,stdout=log,stderr=subprocess.STDOUT,text=True)
            processes[trial]=(process,gpu)
        while processes:
            time.sleep(20)
            for trial,(process,gpu) in list(processes.items()):
                heartbeat=TEMP/trial/'heartbeat.txt'
                age=time.time()-heartbeat.stat().st_mtime if heartbeat.exists() else time.monotonic()-started
                print(trial,'GPU',gpu,'return',process.poll(),'heartbeat age',round(age),'sec',flush=True)
                if age>600 and process.poll() is None:
                    raise TimeoutError(f'{trial} worker made no progress for 10 minutes')
                if process.poll() is not None:
                    logs[trial].close();del processes[trial]
                    if process.returncode:raise RuntimeError(f'{trial} failed; inspect {OUTPUT/trial}.log')
    finally:
        for process,_ in processes.values():
            if process.poll() is None:
                process.terminate()
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:process.kill();process.wait()
        for log in logs.values():log.close()
    print('Worker wall minutes:',round((time.monotonic()-started)/60,2))

if RUN_EXPERIMENT:
    try:launch_workers()
    finally:HF_RUNTIME_TOKEN=None
else:print('Execution disabled until staged as smoke/full.')
''')
    cell_pair(cells, '''
## 10. Compare learning curves, class recall and full validation predictions

Read each worker's saved predictions to recompute confusion matrices and scores. Full runs must contain every 1,920 validation crop exactly once. Smoke results contain the small stratified sample and are excluded from the research comparison. Test data is never accessed.
''','''
if RUN_EXPERIMENT:
    rows=[]
    for trial in CONFIG['trials']:
        folder=OUTPUT/trial;result=json.loads((folder/'metrics.json').read_text());pred=pd.read_csv(folder/'validation_predictions.csv')
        assert pred.crop_id.is_unique and len(pred)>0
        assert (pred.true_class.isin([0,1,2])&pred.predicted_class.isin([0,1,2])).all()
        cm=np.zeros((3,3),dtype=int);np.add.at(cm,(pred.true_class,pred.predicted_class),1)
        assert cm.tolist()==result['metrics']['confusion_matrix']
        expected=frame[frame.split.eq('val')]
        if PHASE=='full':
            assert len(pred)==len(expected)==1920 and set(pred.crop_id)==set(expected.crop_id)
            assert np.array_equal(expected.set_index('crop_id').loc[pred.crop_id].class_id,pred.true_class)
        for key,value in [('accuracy',np.trace(cm)/cm.sum()),('macro_f1',np.mean(precision_recall_fscore_support(pred.true_class,pred.predicted_class,labels=[0,1,2],zero_division=0)[2]))]:
            assert np.isclose(result['metrics'][key],value)
        rows.append({'trial':trial,'phase':PHASE,**{k:v for k,v in result['metrics'].items() if not isinstance(v,list)},
                     'selected_epoch':result['selected_epoch'],'epochs':result['epochs_completed'],
                     'trainable_parameters':result['trainable_parameters'],'seconds':result['elapsed_seconds']})
        display(pd.read_csv(folder/'history.csv'))
    comparison=pd.DataFrame(rows);display(comparison)
    comparison.to_csv(OUTPUT/'comparison.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for trial in CONFIG['trials']:
        history=pd.read_csv(OUTPUT/trial/'history.csv')
        axes[0].plot(history.epoch,history.macro_f1,marker='o',label=trial)
        axes[1].plot(history.epoch,history.val_loss,marker='o',label=trial)
    axes[0].set(xlabel='Epoch',ylabel='Validation macro-F1',title='Checkpoint selection metric')
    axes[1].set(xlabel='Epoch',ylabel='Validation CE loss',title='Validation loss')
    for ax in axes:ax.legend()
    fig.tight_layout();plt.show()
    for trial in CONFIG['trials']:
        result=json.loads((OUTPUT/trial/'metrics.json').read_text());cm=np.asarray(result['metrics']['confusion_matrix'])
        fig,ax=plt.subplots(figsize=(4,3));im=ax.imshow(cm,cmap='Blues')
        ax.set(xticks=[0,1,2],yticks=[0,1,2],xticklabels=['no-visible','mild','marked'],
               yticklabels=['no-visible','mild','marked'],xlabel='Predicted',ylabel='Reference',title=trial)
        for (i,j),value in np.ndenumerate(cm):ax.text(j,i,int(value),ha='center',va='center')
        fig.colorbar(im,ax=ax);fig.tight_layout();plt.show()
        display(pd.DataFrame({'class':['no-visible','mild','marked'],'precision':result['metrics']['class_precision'],
                              'recall':result['metrics']['class_recall'],'F1':result['metrics']['class_f1']}))
    save_json(OUTPUT/'run_config.json',CONFIG)
    print('Saved outputs:',sorted(p.name for p in OUTPUT.iterdir()))
else:display(Markdown('Training is disabled in the source notebook. Kaggle staging enables smoke, followed by a separate full version after both smoke workers pass.'))
''')
    save('11_2_dinov3_partial_adaptation.ipynb',
         "# 11.2 DINOv3 partial fine-tuning\n\nMatched CE classifiers: fully frozen ViT-S/16 versus final two transformer blocks plus final normalization adapted.\n\nMethod references: [DINOv3 paper](https://arxiv.org/abs/2508.10104), [official implementation](https://github.com/facebookresearch/dinov3), and [Transformers DINOv3 interface](https://huggingface.co/docs/transformers/en/model_doc/dinov3). Partial unfreezing is an adaptation experiment from this project's declared method plan, not a reproduced dental result.",cells)


if __name__=='__main__':
    notebook_111()
    notebook_112()

"""Build self-contained teaching notebooks with visible, process-isolated workers."""
from pathlib import Path
import ast
import hashlib
import json
import re
import nbformat as nb

ROOT = Path(__file__).resolve().parents[2]
OLD_SHA = '8ed4c05240a2d23e6c254ee74adf13af48ff45a4418f1e2024bd38498b02cab1'
NEW_SHA = json.loads((ROOT/'data/processed/gum_raw_roi_v1/receipt.json').read_text())['manifest_sha256']
worker = (ROOT/'src/dentalselfie/recovery_worker.py').read_text(encoding='utf-8')
worker = worker.split("\n\nif __name__ == '__main__':")[0]
sections = re.split(r'(?m)^# SECTION: (.+)\n',worker)
blocks = [(sections[i],sections[i+1].strip()) for i in range(1,len(sections),2)]
assembled = '\n\n'.join(source for _,source in blocks)+"\n\nif __name__ == '__main__':\n    main()\n"
ast.parse(assembled)
worker_sha = hashlib.sha256(assembled.encode()).hexdigest()

explanations = [
    'Each trial runs in its own Python process with one visible GPU. Record versions, seed worker processes and avoid the global-RNG interference found in notebook 07. Scientific imports are shared by the visible implementation and its generated worker script.',
    'Locate input manifests by their exact SHA-256 rather than guessing Kaggle mount paths. Verify each crop hash and train/validation grouping. Resize to 224, apply mild training-only flip/rotation, convert to tensor and normalize. All fine-tuning trials use the same transform and microbatch 32 without accumulation.',
    'Load official ImageNet ResNet18 weights and replace only its head. CE is the reference. Class-balanced CE uses beta=0.999 and training counts; Balanced Softmax adds log training counts to logits during training only. Their inference uses ordinary softmax. Do not stack oversampling or double prior correction.',
    'Report accuracy, macro-F1, balanced accuracy, class-wise precision/recall/F1, MAE, kappa and a full confusion matrix. Atomic checkpoint replacement prevents partially written resume files. Probabilities are exported in float32 and retain crop identities.',
    'Smoke mode trains two batches, validates and restores a real checkpoint. Full runs select macro-F1 checkpoints, save optimizer/scheduler/scaler/RNG states and embedded best weights every epoch, and resume only with matching protocol/trial hashes. The loader has a per-epoch seeded generator and seeded child workers.',
    'Frozen probes cache deterministic representations and fit StandardScaler on train only. Both probes use the same logistic-regression C grid [0.01,0.1,1,10] and validation selection. DINOv3 uses the CLS feature from the official model; it requires authorized weights. This probe is separate from end-to-end fine-tuning.',
    'The process entry point validates that exactly one GPU is visible, sets isolated seeds, verifies inputs and dispatches fine-tuning or a frozen probe. Smoke artifacts and full-run artifacts have separate folders, so smoke cannot initialize the reported screening run.'
]

launcher = r'''
import subprocess, sys, time, os
WORKER_PATH = Path('/kaggle/working/recovery_worker.py')
worker_source = '\n\n'.join(WORKER_PARTS) + "\n\nif __name__ == '__main__':\n    main()\n"
# Raw strings in this visible launcher are replaced below by the builder.
worker_source = worker_source.replace('\\n','\n') if False else worker_source
assert hashlib.sha256(worker_source.encode()).hexdigest() == EXPECTED_WORKER_SHA, 'Worker reconstruction differs from reviewed code'
WORKER_PATH.write_text(worker_source,encoding='utf-8')
CONFIG_PATH = Path('/kaggle/working/recovery_config.json')
CONFIG_PATH.write_text(json.dumps(CONFIG,indent=2),encoding='utf-8')

def run_pair(smoke):
    processes=[]; started=time.monotonic(); offsets={}
    try:
        for index,trial in enumerate(CONFIG['trials']):
            run_dir=Path(CONFIG['output_root'])/('smoke' if smoke else 'full')/trial['id']
            run_dir.mkdir(parents=True,exist_ok=True)
            log_path=run_dir/'worker.log'; stream=log_path.open('w',encoding='utf-8')
            env=os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES=str(index),PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
            command=[sys.executable,'-u',str(WORKER_PATH),'--config',str(CONFIG_PATH),'--trial',str(index)]
            if smoke: command.append('--smoke')
            if RESUME_ROOT and not smoke: command += ['--resume-root',str(RESUME_ROOT)]
            process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,env=env)
            processes.append((process,stream,log_path,run_dir,trial['id'])); offsets[trial['id']]=0
            print('Started',trial['id'],'on physical GPU',index,'smoke' if smoke else 'full',flush=True)
        while True:
            active=False
            for process,stream,path,run_dir,trial_id in processes:
                text=path.read_text(encoding='utf-8',errors='replace')
                if len(text)>offsets[trial_id]:
                    print(text[offsets[trial_id]:][-4500:],flush=True); offsets[trial_id]=len(text)
                code=process.poll()
                if code is not None and code!=0: raise RuntimeError(f'{trial_id} failed; inspect {path}')
                if code is None:
                    active=True
                    beat=run_dir/'heartbeat.txt'
                    age=time.time()-beat.stat().st_mtime if beat.exists() else time.monotonic()-started
                    if age>900: raise TimeoutError(f'{trial_id}: no progress heartbeat for 15 minutes')
            if not active: break
            if time.monotonic()-started > (1200 if smoke else 6600):
                raise TimeoutError('Session safety cap reached; retain checkpoints and resume')
            time.sleep(2)
    finally:
        for process,stream,*_ in processes:
            if process.poll() is None: process.terminate()
        for process,stream,*_ in processes:
            try: process.wait(timeout=20)
            except subprocess.TimeoutExpired: process.kill(); process.wait()
            stream.close()
    return time.monotonic()-started

if RUN_TRAINING:
    assert torch.cuda.device_count()==2, 'Select Kaggle GPU T4 x2'
    smoke_seconds=run_pair(True)
    print(f'All smoke workers passed in {smoke_seconds:.1f}s. Starting fresh, independent full runs.',flush=True)
    full_seconds=run_pair(False)
    print(f'Full experiments finished in {full_seconds/60:.1f} minutes.',flush=True)
else:
    print('Training disabled: review/fork without GPU consumption.')
'''
# In actual code, the join separator is a newline, not the literal backslash-n pair.
launcher = launcher.replace("'\\n\\n'.join", "'\n\n'.join") if False else launcher

trials = [
    ('09_1_resolution_control.ipynb','dentalselfie-09-1-resolution-control','DentalSelfie 09.1 Resolution Control',[
        dict(id='R0_old_CE',kind='finetune',data='old',loss='ce'),
        dict(id='R1_raw_CE',kind='finetune',data='raw',loss='ce')],
        'Does extracting the same local ROI from original source pixels improve grading? Only the crop preparation changes. R1 is also the shared CE control for 09.2.',True),
    ('09_2_imbalance_ablation.ipynb','dentalselfie-09-2-imbalance-ablation','DentalSelfie 09.2 Imbalance Ablation',[
        dict(id='R2_raw_CBCE',kind='finetune',data='raw',loss='class_balanced'),
        dict(id='R3_raw_BalancedSoftmax',kind='finetune',data='raw',loss='balanced_softmax')],
        'Compare two imbalance objectives on the fixed raw-source crops. The matched plain-CE control R1 runs independently in 09.1; it is not retrained here. Training does not require its checkpoint or metrics.',True),
    ('09_3_foundation_feature_probe.ipynb','dentalselfie-09-3-foundation-feature-probe','DentalSelfie 09.3 Foundation Feature Probe',[
        dict(id='P0_raw_frozen_ResNet18',kind='probe',data='raw',encoder='resnet18'),
        dict(id='P1_raw_frozen_DINOv3',kind='probe',data='raw',encoder='dinov3')],
        'Compare fixed feature representations with the same linear classifier search budget. No encoder is fine-tuned. This is a separate probe comparison, not an architectural-only attribution because the encoders have different pretraining.',False)
]

for filename,slug,title,experiments,question,enabled in trials:
    cells=[]
    def md(text): cells.append(nb.v4.new_markdown_cell(text.strip()))
    def code(text):
        ast.parse(text.strip()); cells.append(nb.v4.new_code_cell(text.strip()))
    md(f'''# {title}

{question}

| GPU | Trial | Input | Objective / encoder |
|---|---|---|---|
| 0 | {experiments[0]['id']} | {experiments[0]['data']} | {experiments[0].get('loss',experiments[0].get('encoder'))} |
| 1 | {experiments[1]['id']} | {experiments[1]['data']} | {experiments[1].get('loss',experiments[1].get('encoder'))} |

All inputs are the same 10,889 annotated train/validation ROIs, with locked group membership and 1,920 validation rows. No test images are attached. Historical notebook 07 is exploratory evidence; this is a new matched protocol. Primary metric: macro-F1, accompanied by all class recalls and precision. No improvement or clinical usability is assumed.

```mermaid
flowchart LR
 A[Hash-verified local crops] --> B[Matched preprocessing]
 B --> C[GPU 0 isolated worker]
 B --> D[GPU 1 isolated worker]
 C --> E[Smoke then independent full run]
 D --> E
 E --> F[Checkpoints, predictions, plots and comparison]
```

References: [Class-balanced loss](https://openaccess.thecvf.com/content_CVPR_2019/html/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.html), [Balanced Softmax](https://papers.nips.cc/paper_files/paper/2020/hash/2ba61cc3a8f44143e1f2f13b2b729ab3-Abstract.html), [DINOv3](https://arxiv.org/abs/2508.10104). Methods are adapted and cited, not claimed as globally novel.''')
    md('## 1. Fixed protocol and execution controls\n\n30 max epochs, patience 8, microbatch 32, no gradient accumulation, mild flip/rotation, seed 42 and the same optimizer are shared across 09.1/09.2. Separate process RNGs fix the concurrency problem in 07. An external resume root can point to attached prior full outputs; input and protocol fingerprints must match. 09.3 remains disabled until official DINOv3 weights access is provided.')
    config=dict(protocol='roi-recovery-screen-v1',seed=42,batch_size=32,max_epochs=30,
        patience=8,lr=.0003,weight_decay=.0005,worker_seconds=5400,
        manifest_hashes={'old':OLD_SHA,'raw':NEW_SHA},trials=experiments,
        output_root='/kaggle/working/'+slug)
    code(f"from pathlib import Path\nimport json, hashlib\nRUN_TRAINING = {enabled}\nRESUME_ROOT = None\nCONFIG = {config!r}\nEXPECTED_WORKER_SHA = {worker_sha!r}\nWORKER_PARTS = []\nprint(json.dumps(CONFIG,indent=2))")
    if not enabled:
        md('## DINOv3 access and environment\n\nRequest access to the official Hugging Face repository, enable Internet and add HF_TOKEN to Kaggle Secrets with notebook access enabled. Never paste a token into source or publish it. A local authorized Transformers-format model folder may instead be supplied through DINOV3_MODEL_DIR. Enable RUN_TRAINING only once access is ready. Both encoders see the full 224 crop with ImageNet normalization, without an extra center crop.')
        code("import subprocess, sys\nif RUN_TRAINING:\n    subprocess.check_call([sys.executable,'-m','pip','install','-q','transformers==4.57.1','safetensors'])\nprint('DINOv3 access is required; no alternate model is silently substituted.')")
    for i,((title_section,source),explanation) in enumerate(zip(blocks,explanations),start=2):
        md(f'## {i}. {title_section}\n\n{explanation}')
        capture="\n# CAPTURE_THIS_CELL\nWORKER_PARTS.append(get_ipython().history_manager.input_hist_raw[-1].rsplit('\\n# CAPTURE_THIS_CELL',1)[0])\nprint('Registered worker section: '+"+repr(title_section)+")"
        # String in generated code must represent a newline delimiter.
        capture=capture.replace("'\\n# CAPTURE_THIS_CELL'", "'\\n# CAPTURE_THIS_CELL'")
        code(source+capture)
    md('## Inspect class counts, model tensors and actual crop examples\n\nResolve both manifests by hash, compare membership where applicable and show a class-count plot plus one example from every class. Tensor normalization is visible in the dataset cell above. These images are not a favorable subset used for metric computation.')
    code('''import matplotlib.pyplot as plt
from IPython.display import display
preview_root, preview_frame = locate_data(CONFIG, CONFIG['trials'][0])
display(preview_frame.groupby(['split','class_name']).size().unstack(fill_value=0))
fig,axes=plt.subplots(1,3,figsize=(10,3))
for class_id,ax in enumerate(axes):
 row=preview_frame[(preview_frame.split=='train') & (preview_frame.class_id==class_id)].iloc[0]
 ax.imshow(Image.open(preview_root/row.local_path)); ax.set_title(row.class_name.replace('_',' '),fontsize=9); ax.axis('off')
plt.tight_layout(); plt.show()
sample=LocalROIDataset(preview_frame.head(2),preview_root)[0][0]
print('Input tensor:',tuple(sample.shape),'float range:',float(sample.min()),float(sample.max()))
counts=torch.tensor([210.,3829.,4930.]); beta=.999
weights=(1-beta)/(1-torch.pow(beta,counts)); weights=weights/weights.mean()
display(pd.DataFrame({'class_id':[0,1,2],'train_count':counts.numpy(),'CB_CE_weight':weights.numpy(),'balanced_softmax_log_count':counts.log().numpy()}))''')
    md('## Isolated GPU smoke, checkpoint validation and full screening\n\nReconstruct the worker only from the displayed implementation cells and verify its exact code hash. First launch two short smoke workers. Only if both pass, start fresh full workers. Output folders are separate; latest checkpoints allow explicit resumption. A 15-minute no-progress watchdog and 110-minute full-stage cap prevent a silent 12-hour session.')
    code(launcher)
    md('## Read the measured outputs and inspect convergence\n\nMetrics below are loaded from actual worker files. Show confusion matrices, predictions, and saved history. Frozen probe runs instead report their bounded linear-head grid. Neither notebook reads the results of another during training; combine the shared-control comparison after all tracks finish.')
    code('''results=[]
for trial in CONFIG['trials']:
 folder=Path(CONFIG['output_root'])/'full'/trial['id']
 if not (folder/'metrics.json').exists():
  print(trial['id'],': no full result yet'); continue
 result=json.loads((folder/'metrics.json').read_text()); results.append(result)
 print(trial['id'], '| macro-F1',round(result['macro_f1'],4),'| recall',np.round(result['class_recall'],3))
 display(pd.DataFrame(result['confusion_matrix'],index=['No visible','Mild','Marked'],columns=['No visible','Mild','Marked']))
 display(pd.read_csv(folder/'validation_predictions.csv').head())
 if (folder/'history.csv').exists():
  h=pd.read_csv(folder/'history.csv'); fig,ax=plt.subplots(1,2,figsize=(10,3)); ax[0].plot(h.epoch,h.train_loss,label='train'); ax[0].plot(h.epoch,h.val_loss,label='val'); ax[0].legend(); ax[1].plot(h.epoch,h.macro_f1); ax[1].set_ylabel('Val macro-F1'); plt.tight_layout(); plt.show()
 else: display(pd.read_json(folder/'probe_grid.json')[['C','accuracy','macro_f1','balanced_accuracy']])
if results:
 out=Path(CONFIG['output_root'])/'summary.json'; out.write_text(json.dumps(results,indent=2),encoding='utf-8')
 display(pd.DataFrame([{'trial':r['trial']['id'],'accuracy':r['accuracy'],'macro_f1':r['macro_f1'],'balanced_accuracy':r['balanced_accuracy'],'seconds':r['elapsed_seconds']} for r in results]))''')
    md('## Research interpretation\n\nKeep source-domain ROI grading distinct from end-to-end detection and phone-image evaluation. Compare 09.1 R0/R1 for preparation, and R1/R2/R3 for loss. Compare P0/P1 separately as frozen probes. Report failed or time-budget-limited runs, minority precision as well as recall, and correlated parent-image sampling. Additional seeds and the locked test stage follow only after selecting the candidate. Do not combine modules merely because they sound advanced.')
    notebook=nb.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec={'display_name':'Python 3','language':'python','name':'python3'}
    path=ROOT/'notebooks'/filename
    if path.exists(): raise FileExistsError(path)
    nb.write(notebook,path)
    stage=ROOT/'.venv/work/kaggle/kernels'/slug; stage.mkdir(parents=True,exist_ok=True)
    nb.write(notebook,stage/filename)
    inputs=['trihuynhviprovcl/dentalselfie-gum-raw-roi-v1/1']
    if filename.startswith('09_1'): inputs.insert(0,'trihuynhviprovcl/dentalselfie-gum-roi-v1/1')
    metadata=dict(id='trihuynhviprovcl/'+slug,title=title,code_file=filename,language='python',
        kernel_type='notebook',is_private=False,enable_gpu=True,enable_internet=True,
        machine_shape='NvidiaTeslaT4',dataset_sources=inputs,competition_sources=[],kernel_sources=[])
    (stage/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(filename,len(cells),'cells; worker SHA',worker_sha)


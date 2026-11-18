"""Standalone, process-isolated worker embedded visibly in the 09.x notebooks."""
# SECTION: Imports and deterministic setup
from pathlib import Path
import argparse
import copy
import hashlib
import json
import os
import random
import time
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import (accuracy_score, f1_score, balanced_accuracy_score,
                            confusion_matrix, cohen_kappa_score, precision_recall_fscore_support)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


# SECTION: Dataset and matched image preprocessing
class LocalROIDataset(Dataset):
    def __init__(self, frame, root, training=False):
        self.frame = frame.reset_index(drop=True)
        self.root = Path(root)
        steps = [transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR)]
        if training:
            steps += [transforms.RandomHorizontalFlip(), transforms.RandomRotation(
                7, interpolation=transforms.InterpolationMode.BILINEAR, fill=(128,128,128))]
        steps += [transforms.ToTensor(), transforms.Normalize(
            (0.485,0.456,0.406), (0.229,0.224,0.225))]
        self.transform = transforms.Compose(steps)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(self.root / row.local_path) as image:
            tensor = self.transform(image.convert('RGB'))
        return tensor, int(row.class_id), str(row.crop_id)


def locate_data(config, trial):
    paths = list(Path('/kaggle/input').glob('**/manifest.csv'))
    target_hash = config['manifest_hashes'][trial['data']]
    matches = [p for p in paths if hashlib.sha256(p.read_bytes()).hexdigest() == target_hash]
    if len(matches) != 1:
        raise RuntimeError(f'Expected exactly one manifest for {trial["data"]}; found {len(matches)}')
    root = matches[0].parent
    frame = pd.read_csv(matches[0])
    assert set(frame.split) == {'train','val'} and frame.crop_id.is_unique
    assert frame.groupby('group_id').split.nunique().max() == 1
    expected = ['no_visible_inflammation','mild_inflammation','marked_inflammation']
    assert np.array_equal(frame.class_name.map({n:i for i,n in enumerate(expected)}), frame.class_id)
    for row in frame.itertuples():
        if hashlib.sha256((root / row.local_path).read_bytes()).hexdigest() != row.local_sha256:
            raise RuntimeError(f'Image checksum differs: {row.crop_id}')
    return root, frame


# SECTION: Models and independent class-imbalance objectives
def build_encoder(kind):
    if kind == 'dinov3':
        from transformers import AutoModel
        token = os.environ.get('HF_TOKEN')
        if not token:
            try:
                from kaggle_secrets import UserSecretsClient
                token = UserSecretsClient().get_secret('HF_TOKEN')
            except Exception:
                token = None
        source = os.environ.get('DINOV3_MODEL_DIR', 'facebook/dinov3-vits16-pretrain-lvd1689m')
        return AutoModel.from_pretrained(source, token=token)
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Identity()
    return model


def make_classifier():
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 3)
    return model


def classification_loss(logits, labels, objective, counts):
    if objective == 'balanced_softmax':
        return nn.functional.cross_entropy(logits.float() + counts.log(), labels)
    if objective == 'class_balanced':
        beta = 0.999
        weights = (1-beta)/(1-torch.pow(beta, counts))
        weights = weights / weights.mean()
        return nn.functional.cross_entropy(logits.float(), labels, weight=weights)
    if objective != 'ce':
        raise ValueError(objective)
    return nn.functional.cross_entropy(logits.float(), labels)


# SECTION: Metrics and atomic checkpoints
def metric_bundle(truth, prediction):
    cm = confusion_matrix(truth, prediction, labels=[0,1,2])
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, prediction, labels=[0,1,2], zero_division=0)
    return dict(accuracy=float(accuracy_score(truth,prediction)),
        macro_f1=float(f1_score(truth,prediction,labels=[0,1,2],average='macro',zero_division=0)),
        balanced_accuracy=float(balanced_accuracy_score(truth,prediction)),
        ordinal_mae=float(np.abs(np.array(truth)-prediction).mean()),
        quadratic_kappa=float(cohen_kappa_score(truth,prediction,weights='quadratic')),
        confusion_matrix=cm.tolist(), class_precision=precision.tolist(),
        class_recall=recall.tolist(), class_f1=f1.tolist(), support=support.tolist())


def atomic_checkpoint(path, value):
    temp = path.with_suffix('.tmp')
    torch.save(value, temp)
    os.replace(temp, path)


def heartbeat(folder, message):
    (folder/'heartbeat.txt').write_text(f'{time.time()} {message}',encoding='utf-8')


def evaluate(model, loader, device, objective, counts, limit=None):
    model.eval()
    truth, predictions, ids, probabilities = [], [], [], []
    loss_sum = 0.
    with torch.no_grad():
        for step,(images,labels,crop_ids) in enumerate(loader):
            images,labels = images.to(device),labels.to(device)
            with torch.autocast('cuda',dtype=torch.float16):
                logits = model(images)
            loss = classification_loss(logits,labels,objective,counts)
            probs = logits.float().softmax(1)
            loss_sum += float(loss)*len(labels)
            truth.extend(labels.cpu().tolist()); predictions.extend(probs.argmax(1).cpu().tolist())
            ids.extend(crop_ids); probabilities.extend(probs.cpu().tolist())
            if limit and step+1 >= limit:
                break
    return metric_bundle(truth,predictions), loss_sum/len(truth), pd.DataFrame(
        {'crop_id':ids,'true_class':truth,'predicted_class':predictions,
         'probabilities':[json.dumps(p) for p in probabilities]})


# SECTION: Fine-tuning, GPU smoke and exact-protocol resumption
def finetune(config, trial, root, frame, folder, smoke=False, resume_root=None):
    device = torch.device('cuda:0')
    seed = config['seed']
    model = make_classifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=.5,patience=2)
    scaler = torch.amp.GradScaler('cuda')
    train_frame, val_frame = frame[frame.split=='train'],frame[frame.split=='val']
    counts = torch.tensor(train_frame.class_id.value_counts().reindex(range(3)).to_numpy(),device=device,dtype=torch.float32)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(LocalROIDataset(train_frame,root,True),batch_size=config['batch_size'],
        shuffle=True,num_workers=0 if smoke else 2,pin_memory=True,worker_init_fn=seed_worker,generator=generator)
    val_loader = DataLoader(LocalROIDataset(val_frame,root),batch_size=64,shuffle=False,
        num_workers=0 if smoke else 2,pin_memory=True)
    run_signature = fingerprint({'protocol':config,'trial':trial})
    history, start_epoch, best_score, stale = [], 1, -1., 0
    best_state = None
    checkpoint = folder/'resume.pt'
    source_checkpoint = checkpoint if checkpoint.exists() else (
        Path(resume_root)/trial['id']/'resume.pt' if resume_root else None)
    if source_checkpoint and source_checkpoint.exists() and not smoke:
        state = torch.load(source_checkpoint,map_location='cpu',weights_only=False)
        assert state['signature']==run_signature, 'Resume protocol/data/trial mismatch'
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler']); scaler.load_state_dict(state['scaler'])
        start_epoch=state['epoch']+1; history=state['history']; best_score=state['best_score']
        stale=state['stale']; best_state=state['best_state']
        random.setstate(state['python_rng']); np.random.set_state(state['numpy_rng'])
        torch.set_rng_state(state['torch_rng']); torch.cuda.set_rng_state(state['cuda_rng'])
        print(f"{trial['id']}: resume epoch {start_epoch}",flush=True)
    started=time.monotonic(); stop_reason='max_epochs'
    for epoch in range(start_epoch,(1 if smoke else config['max_epochs'])+1):
        if time.monotonic()-started > config['worker_seconds']:
            stop_reason='time_budget'; break
        if stale>=config['patience']:
            stop_reason='early_stopping'; break
        generator.manual_seed(seed+epoch)
        model.train(); running_loss=0.; total=0
        heartbeat(folder,f'train epoch {epoch}')
        for step,(images,labels,_) in enumerate(train_loader):
            images,labels=images.to(device,non_blocking=True),labels.to(device,non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):
                logits=model(images)
            loss=classification_loss(logits,labels,trial['loss'],counts)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            running_loss+=float(loss.detach())*len(labels); total+=len(labels)
            if step%50==0: heartbeat(folder,f'epoch {epoch} batch {step}')
            if smoke and step>=1: break
        metrics,val_loss,predictions=evaluate(model,val_loader,device,trial['loss'],counts,limit=2 if smoke else None)
        scheduler.step(metrics['macro_f1'])
        improved=metrics['macro_f1']>best_score
        if improved:
            best_score=metrics['macro_f1']; stale=0
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            atomic_checkpoint(folder/'best.pt',best_state)
        else: stale+=1
        history.append(dict(epoch=epoch,train_loss=running_loss/total,val_loss=val_loss,
            macro_f1=metrics['macro_f1'],accuracy=metrics['accuracy'],lr=optimizer.param_groups[0]['lr'],
            seconds=time.monotonic()-started))
        state=dict(signature=run_signature,config=config,trial=trial,epoch=epoch,
            model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
            scaler=scaler.state_dict(),best_state=best_state,best_score=best_score,stale=stale,history=history,
            python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state())
        atomic_checkpoint(checkpoint,state)
        pd.DataFrame(history).to_csv(folder/'history.csv',index=False)
        heartbeat(folder,f'epoch {epoch} saved')
        print(json.dumps({'trial':trial['id'],**history[-1]}),flush=True)
        if smoke:
            reloaded=torch.load(checkpoint,map_location=device,weights_only=False)
            assert reloaded['signature']==run_signature
            model.load_state_dict(reloaded['model'])
            optimizer.load_state_dict(reloaded['optimizer']); scheduler.load_state_dict(reloaded['scheduler'])
            scaler.load_state_dict(reloaded['scaler'])
            print('SMOKE_OK: finite training, validation and checkpoint restore',flush=True)
    if best_state is None: raise RuntimeError('No trained checkpoint exists')
    model.load_state_dict(best_state)
    metrics,val_loss,predictions=evaluate(model,val_loader,device,trial['loss'],counts,limit=2 if smoke else None)
    predictions.to_csv(folder/'validation_predictions.csv',index=False)
    metrics.update(trial=trial,signature=run_signature,seed=seed,epochs=len(history),
        selected_epoch=int(max(history,key=lambda row:row['macro_f1'])['epoch']),
        stop_reason='smoke' if smoke else ('early_stopping' if stale>=config['patience'] else stop_reason),
        elapsed_seconds=time.monotonic()-started,gpu=torch.cuda.get_device_name(0),
        torch_version=torch.__version__,manifest_sha256=config['manifest_hashes'][trial['data']],
        parameters=sum(p.numel() for p in model.parameters()),test_images_opened=0)
    (folder/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')


# SECTION: Frozen-feature probes and train-only feature standardization
def feature_probe(config,trial,root,frame,folder,smoke=False):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    import joblib
    encoder=build_encoder(trial['encoder']).eval().to('cuda:0')
    for p in encoder.parameters(): p.requires_grad=False
    features={}; labels={}; ids={}
    started=time.monotonic()
    for split in ['train','val']:
        loader=DataLoader(LocalROIDataset(frame[frame.split==split],root),batch_size=64,
            shuffle=False,num_workers=0 if smoke else 2,pin_memory=True)
        chunks=[]; targets=[]; crop_ids=[]
        for step,(images,y,crop_id) in enumerate(loader):
            with torch.no_grad(), torch.autocast('cuda',dtype=torch.float16):
                if trial['encoder']=='dinov3':
                    result=encoder(pixel_values=images.to('cuda:0'))
                    z=result.last_hidden_state[:,0]
                else: z=encoder(images.to('cuda:0'))
            chunks.append(z.float().cpu().numpy()); targets.extend(y.tolist()); crop_ids.extend(crop_id)
            heartbeat(folder,f'extract {split} batch {step}')
            if smoke and step>=1: break
        features[split]=np.concatenate(chunks); labels[split]=np.array(targets); ids[split]=crop_ids
        np.save(folder/f'{split}_features.npy',features[split])
    assert set(labels['train'])=={0,1,2}, 'Probe smoke must include all classes; increase smoke batches if needed'
    scaler=StandardScaler().fit(features['train'])
    train_x=scaler.transform(features['train']); val_x=scaler.transform(features['val'])
    best_score=-1.; best=None; history=[]
    for c in ([.1] if smoke else [.01,.1,1.,10.]):
        classifier=LogisticRegression(C=c,max_iter=500,solver='lbfgs',random_state=config['seed'])
        classifier.fit(train_x,labels['train']); predictions=classifier.predict(val_x)
        metrics=metric_bundle(labels['val'],predictions)
        history.append({'C':c,**metrics}); heartbeat(folder,f'linear C={c}')
        print(json.dumps({'trial':trial['id'],'C':c,'macro_f1':metrics['macro_f1']}),flush=True)
        if metrics['macro_f1']>best_score:
            best_score=metrics['macro_f1']; best=(classifier,metrics,c)
    classifier,metrics,c=best
    joblib.dump({'scaler':scaler,'classifier':classifier},folder/'linear_probe.joblib')
    probabilities=classifier.predict_proba(val_x)
    pd.DataFrame({'crop_id':ids['val'],'true_class':labels['val'],
        'predicted_class':classifier.predict(val_x),
        'probabilities':[json.dumps(p.tolist()) for p in probabilities]}).to_csv(folder/'validation_predictions.csv',index=False)
    metrics.update(trial=trial,seed=config['seed'],selected_C=c,elapsed_seconds=time.monotonic()-started,
        gpu=torch.cuda.get_device_name(0),feature_dim=features['train'].shape[1],
        manifest_sha256=config['manifest_hashes'][trial['data']],test_images_opened=0,
        stop_reason='smoke' if smoke else 'grid_complete',signature=fingerprint({'protocol':config,'trial':trial}))
    (folder/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    (folder/'probe_grid.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
    if smoke: print('SMOKE_OK: feature extraction and train-fitted linear probe',flush=True)


# SECTION: Isolated worker entry point
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--trial',type=int,required=True)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--resume-root')
    args=parser.parse_args()
    config=json.loads(Path(args.config).read_text())
    trial=config['trials'][args.trial]
    seed=config['seed']; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
    torch.set_num_threads(2)
    if torch.cuda.device_count()!=1: raise RuntimeError('Worker must see exactly one assigned GPU')
    folder=Path(config['output_root'])/('smoke' if args.smoke else 'full')/trial['id']
    folder.mkdir(parents=True,exist_ok=True); heartbeat(folder,'worker started')
    root,frame=locate_data(config,trial)
    if args.smoke and trial['kind']=='probe':
        frame=pd.concat([g.sample(n=min(32,len(g)),random_state=seed)
                         for _,g in frame.groupby(['split','class_id'])])
    print(json.dumps({'trial':trial,'rows':len(frame),'gpu':torch.cuda.get_device_name(0),
                     'manifest_sha256':config['manifest_hashes'][trial['data']]}),flush=True)
    if trial['kind']=='probe': feature_probe(config,trial,root,frame,folder,args.smoke)
    else: finetune(config,trial,root,frame,folder,args.smoke,args.resume_root)
    heartbeat(folder,'complete')


if __name__ == '__main__':
    main()

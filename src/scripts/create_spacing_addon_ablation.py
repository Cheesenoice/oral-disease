"""Create a matched base-vs-external-data spacing ablation for Kaggle."""
from __future__ import annotations

import json
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[2]
SOURCE_NB = ROOT / "notebooks/alignment/16_64_spacing_ciou_nwd_ablation.ipynb"
OUT_NB = ROOT / "notebooks/alignment/16_67_spacing_train_addon_ablation.ipynb"
STAGE = ROOT / ".venv/work/kaggle/spacing_train_addon_ablation"
ADDON_MANIFEST_SHA = "9c413bc2f2e811c22fb3e31f46f7ce584840fca75c5d6b99372ce17f1cb94036"


def main() -> None:
    nb = nbformat.read(SOURCE_NB, as_version=4)
    c = nb.cells
    c[0].source = """# 16.67 - Train-only external-data adaptation for tooth spacing

## Question and fixed comparison

Does adding a small, screened external source improve the spacing detector when the architecture, tooth-anatomy initialization, NWD loss, seed, training schedule, and locked validation are held constant?

| Arm | Training data | Loss and model |
|---|---|---|
| Control | Locked project train split | YOLO26s + common tooth-anatomy initialization + CIoU-NWD |
| Data adaptation | Same project train split plus the 119 external train images repeated four times | Identical |

Repeating the small source four times makes its 216 boxes visible enough to test source adaptation, while the source validation partition remains diagnostic-only and the source test partition is excluded. The main comparison always uses the same 477-image project validation and its 473-image overlap-excluded sensitivity view. We do not choose validation images to raise scores.

The extra source is heterogeneous and its annotation quality is not independently adjudicated. This is an experiment, not a claim of clinical generalization. The project validation has already guided development and is not an independent test."""
    c[1].source = "## 1. Pin runtime and verify the two Kaggle T4 devices"
    c[3].source = c[3].source.replace(
        "INPUT=Path('/kaggle/input'); OUT=Path('/kaggle/working/spacing_nwd'); OUT.mkdir(parents=True,exist_ok=True)",
        "INPUT=Path('/kaggle/input'); OUT=Path('/kaggle/working/spacing_addon'); OUT.mkdir(parents=True,exist_ok=True); TMP=Path('/tmp/spacing_addon'); TMP.mkdir(parents=True,exist_ok=True)")
    c[4].source = "## 2. Verify both immutable manifests and keep the source test unopened"
    c[5].source += "\n\nADDON_MANIFEST_SHA='" + ADDON_MANIFEST_SHA + "'\naddon_manifests=[p for p in INPUT.rglob('manifest.csv') if sha256_file(p)==ADDON_MANIFEST_SHA]\nassert len(addon_manifests)==1, addon_manifests\nADDON=addon_manifests[0].parent\naddon_manifest=pd.read_csv(addon_manifests[0])\nassert len(addon_manifest)==141 and addon_manifest.split.value_counts().to_dict()=={'train':119,'valid':22}\nassert addon_manifest.loc[addon_manifest.split.eq('train'),'boxes'].sum()==216\nassert not (ADDON/'test').exists()\naddon_train_rows=addon_manifest[addon_manifest.split.eq('train')].copy()\nprint({'addon_manifest_sha256':ADDON_MANIFEST_SHA,'addon_train_images':len(addon_train_rows),'addon_train_boxes':int(addon_train_rows.boxes.sum()),'addon_valid_reserved_only':22,'addon_test_opened':False})"
    c[6].source = "## 3. Build matched base and 4x add-on training manifests\n\nOnly train images are modified between arms. The locked project validation and sensitivity IDs are copied unchanged into both YAML configurations. External images are repeated four times in the add-on arm; the 22 positive-only external validation images are not used for training or the main mAP comparison. Prepared files go under `/tmp`, so Kaggle output contains model evidence rather than a second copy of the dataset."
    c[7].source = '''BASE=TMP/'base'; ADDON_RUN=TMP/'addon4x'; counts={}
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
        (label_dir/(image.stem+'.txt')).write_text(('\\n'.join(kept)+'\\n') if kept else '',encoding='utf-8')
        boxes+=len(kept); positives+=bool(kept)
    return {'images':len(rows),'positive_images':int(positives),'boxes':int(boxes)}
for root in [BASE,ADDON_RUN]:
    counts[str(root.name)]={}
    for split,rows in [('train',train_rows),('valid',valid_rows),('valid_sensitivity',sensitivity_rows)]:
        counts[str(root.name)][split]=prepare_project_split(root,split,rows)
base_train_images=BASE/'train'/'images'; base_train_labels=BASE/'train'/'labels'
addon_img_dir=ADDON_RUN/'train'/'images'; addon_lbl_dir=ADDON_RUN/'train'/'labels'
for repeat in range(4):
    for row in addon_train_rows.itertuples(index=False):
        image=ADDON/'train'/'images'/row.output_image
        label=ADDON/'train'/'labels'/(Path(row.output_image).stem+'.txt')
        assert image.is_file() and label.is_file()
        assert hashlib.sha256(image.read_bytes()).hexdigest()==row.image_sha256
        assert hashlib.sha256(label.read_bytes()).hexdigest()==row.label_sha256
        name=f'ext{repeat}_{row.output_image}'
        (addon_img_dir/name).symlink_to(image)
        shutil.copyfile(label,addon_lbl_dir/(Path(name).stem+'.txt'))
addon_train_count=len(list(addon_img_dir.iterdir()))
addon_box_count=sum(len(p.read_text().splitlines()) for p in addon_lbl_dir.glob('*.txt'))
assert addon_train_count==15030+4*119 and addon_box_count==8223+4*216
for root in [BASE,ADDON_RUN]:
    (root/'data.yaml').write_text(yaml.safe_dump({'path':str(root),'train':'train/images','val':'valid/images','names':{0:'gap_or_spacing'}},sort_keys=False))
    (root/'sensitivity.yaml').write_text(yaml.safe_dump({'path':str(root),'train':'train/images','val':'valid_sensitivity/images','names':{0:'gap_or_spacing'}},sort_keys=False))
assert counts['base']['valid']==counts['addon4x']['valid'] and counts['base']['valid_sensitivity']==counts['addon4x']['valid_sensitivity']
display(pd.DataFrame({'control':counts['base'],'external_4x':counts['addon4x']}))
print({'addon_arm_train_images':addon_train_count,'addon_arm_boxes':addon_box_count,'external_valid_used_for_main_metrics':False})'''
    c[8].source = "## 4. Reconstruct the same common tooth-anatomy initialization for both arms"
    c[10].source = "## 5. Compute the NWD scale from the locked project train split only\n\nThe NWD normalizer remains unchanged between arms: only the original project train boxes contribute. This keeps the loss scale controlled and avoids allowing external validation or project validation labels to influence training configuration."
    c[11].source = c[11].source.replace("assert len(sizes)==8223", "assert len(sizes)==8223")
    c[12].source = "## 6. Define one shared NWD worker\n\nBoth arms use CIoU-NWD. They differ only in training-data membership and the declared 4x repetition of the external train partition. Both use the same seed, initialization hash, epochs, optimizer, augmentations, image size, batch size and validation IDs."
    c[13].source = c[13].source.replace("p.add_argument('--data',required=True);", "p.add_argument('--data',required=True);")
    c[13].source = c[13].source.replace("if a.arm=='ciou_nwd':patch_nwd(a.C,a.mix)", "patch_nwd(a.C,a.mix)")
    c[13].source = c[13].source.replace("epochs=60,patience=10", "epochs=60,patience=10")
    c[14].source = "## 7. Launch the matched data comparison on separate T4 devices"
    c[15].source = '''processes={}; logs={}; handles={}
arms_to_run=[('project_control',BASE/'data.yaml',0),('external_train_4x',ADDON_RUN/'data.yaml',1)]
for arm,data_yaml,gpu in arms_to_run:
    log_path=OUT/f'{arm}.log'; handle=log_path.open('w',encoding='utf-8'); handles[arm]=handle
    cmd=[sys.executable,str(worker_path),'--arm',arm,'--device',str(gpu),'--data',str(data_yaml),
         '--init',str(COMMON_INIT),'--out',str(OUT/arm),'--C',str(C_NWD),'--mix',str(NWD_LAMBDA)]
    processes[arm]=subprocess.Popen(cmd,stdout=handle,stderr=subprocess.STDOUT)
    print({'arm':arm,'gpu':gpu,'pid':processes[arm].pid,'log':str(log_path)},flush=True)
deadline=time.monotonic()+6.25*3600
while any(p.poll() is None for p in processes.values()):
    states={k:p.poll() for k,p in processes.items()}
    print({'worker_exit_codes':states,'elapsed_hours':round((6.25*3600-(deadline-time.monotonic()))/3600,2)},flush=True)
    if time.monotonic()>deadline:
        for p in processes.values():
            if p.poll() is None:p.terminate()
        raise TimeoutError('Safety guard reached. Preserve checkpoints and logs for continuation.')
    time.sleep(300)
for handle in handles.values():handle.close()
assert all(p.returncode==0 for p in processes.values()),{k:p.returncode for k,p in processes.items()}'''
    c[16].source = "## 8. Evaluate both checkpoints on the same fixed project cohorts\n\nThe external positive-only validation is intentionally not used to compute AP or the primary comparison. Primary validation is 477 images; sensitivity is the predeclared 473-image view with known overlap counterparts removed."
    c[17].source = '''arms={}
for arm,data_root,gpu in [('project_control',BASE,0),('external_train_4x',ADDON_RUN,1)]:
    run=OUT/arm/arm; best=run/'weights'/'best.pt'; assert best.is_file()
    history=pd.read_csv(run/'results.csv'); history.columns=history.columns.str.strip()
    primary=YOLO(str(best)).val(data=str(data_root/'data.yaml'),imgsz=640,batch=16,device=gpu,workers=2,plots=True)
    sensitivity=YOLO(str(best)).val(data=str(data_root/'sensitivity.yaml'),imgsz=640,batch=16,device=gpu,workers=2,plots=False)
    arms[arm]={'epochs':len(history),'checkpoint_sha256':sha256_file(best),'primary_477':primary.results_dict,'sensitivity_473':sensitivity.results_dict}
    shutil.copy2(best,OUT/f'{arm}_best.pt'); shutil.copy2(run/'results.csv',OUT/f'{arm}_results.csv')
summary={'experiment':'fixed_nwd_external_train_addon_4x','spacing_manifest_sha256':EXPECTED_MANIFEST,
 'addon_manifest_sha256':ADDON_MANIFEST_SHA,'mask_pretrain_checkpoint_sha256':EXPECTED_MASK_CKPT,
 'common_initialization_sha256':sha256_file(COMMON_INIT),'project_train_images':15030,'project_train_boxes':8223,
 'addon_unique_train_images':119,'addon_unique_train_boxes':216,'addon_repeats':4,
 'addon_arm_train_images':addon_train_count,'addon_arm_train_boxes':addon_box_count,'NWD_C_input_pixels':C_NWD,
 'NWD_mix_weight':NWD_LAMBDA,'primary_validation_images':477,'sensitivity_validation_images':473,
 'external_positive_only_validation_used_for_ap':False,'source_test_opened':False,'validation_reused':True,
 'patient_independence_verified':False,'arms':arms}
(OUT/'comparison.json').write_text(json.dumps(summary,indent=2,default=str),encoding='utf-8')
display(pd.DataFrame({k:{'AP50':v['primary_477'].get('metrics/mAP50(B)'),'AP50-95':v['primary_477'].get('metrics/mAP50-95(B)'),'precision':v['primary_477'].get('metrics/precision(B)'),'recall':v['primary_477'].get('metrics/recall(B)'),'epochs':v['epochs']} for k,v in arms.items()}).T)
print(json.dumps(summary,indent=2,default=str))'''
    c[18].source = """## Interpretation and promotion rule

Report the paired primary and sensitivity results, checkpoint hashes, data counts and source attribution. Promote the data-adapted arm only if it improves primary AP50 by at least 1 percentage point, does not reduce AP50-95 on either fixed project cohort, and does not reduce recall by more than 1.5 points. Otherwise retain the control and document the negative result. Do not select a validation subset or tune the reported cohort after seeing the results.

The 22-image external validation is positive-only and is not a full detection benchmark. The project validation is reused development data; this experiment does not establish patient-independent or clinical performance."""
    # Use temporary preprocessing, but keep models and compact evidence in Kaggle output.
    nbformat.validate(nb)
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "code" and not cell.source.startswith("%"):
            compile(cell.source, f"cell_{i}", "exec")
        if cell.cell_type == "code" and "worker_source =" in cell.source:
            import ast
            tree = ast.parse(cell.source)
            worker = next(ast.literal_eval(n.value) for n in ast.walk(tree)
                          if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "worker_source" for t in n.targets))
            compile(worker, "embedded_worker", "exec")
    OUT_NB.parent.mkdir(parents=True, exist_ok=True)
    STAGE.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, OUT_NB)
    nbformat.write(nb, STAGE / OUT_NB.name)
    metadata = {
        "id": "hngphc73/ds-spacing-train-add-on-ablation",
        "title": "DS Spacing Train Add-on Ablation",
        "code_file": OUT_NB.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": False,
        "enable_gpu": True,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": ["trihuynhviprovcl/dentalselfie-visible-alignment-v2/1", "hngphc73/teeth-gap-train-addon-v1/2"],
        "competition_sources": [],
        "kernel_sources": ["hngphc73/ds-spacing-iop-anatomy-transfer/2"],
    }
    (STAGE / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_NB} and Kaggle stage {STAGE}; {len(nb.cells)} cells validated.")


if __name__ == "__main__":
    main()

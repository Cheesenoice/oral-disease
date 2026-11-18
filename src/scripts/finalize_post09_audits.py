"""Add executed evidence summaries while retaining completed notebook cells."""
from pathlib import Path
import hashlib
import json
import os

import nbformat as nb
import pandas as pd
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / 'docs/reports/post09'


def finish_notebook(name, title, markdown, code):
    path = ROOT / 'notebooks' / name
    book = nb.read(path, 4)
    assert not any(o.output_type == 'error' for c in book.cells if c.cell_type == 'code' for o in c.outputs)
    flow = ('flowchart LR\n    A[Locked validation inputs] --> B[Verify identities and processing]\n'
            '    B --> C[Recompute metrics and inspect errors]\n    C --> D[Save evidence and decide next experiment]')
    book.cells[0].source = title + '\n\n```mermaid\n' + flow + '\n```'
    marker = '## 9. Collected evidence and presentation summary'
    if any(c.cell_type == 'markdown' and c.source.startswith(marker) for c in book.cells):
        print('Preserved previously executed summary:', name)
        return
    cells = [nb.v4.new_markdown_cell(marker + '\n\n' + markdown), nb.v4.new_code_cell(code)]
    mini = nb.v4.new_notebook(cells=cells, metadata=book.metadata)
    os.environ['JUPYTER_PATH'] = str(ROOT / '.venv/share/jupyter') + os.pathsep + os.environ.get('JUPYTER_PATH', '')
    NotebookClient(mini, timeout=300, kernel_name='dentalselfie-local',
                   resources={'metadata': {'path': str(ROOT)}}).execute()
    book.cells.extend(mini.cells)
    nb.validate(book)
    nb.write(book, path)
    print('Finalized', name)


def main():
    gum = json.loads((REPORT / 'gingival/summary.json').read_text())
    caries = json.loads((REPORT / 'caries/summary.json').read_text())
    assert caries['runs'] == 8 and gum['verified_prediction_runs'] == 10
    finish_notebook('10_1_gingival_failure_audit.ipynb',
                    '# 10.1 Gingival failure audit\n\nExecuted validation audit; no training or test evaluation.',
                    'The common-error population is a review priority, not proof that labels are wrong. '
                    'Report the small minority support and preserve the full validation population.', '''
from pathlib import Path
import json, pandas as pd
from IPython.display import display, Markdown
root = Path.cwd()
out = root / 'docs/reports/post09/gingival'
summary = json.loads((out / 'summary.json').read_text())
data = pd.read_csv(out / 'all_validation_diagnostics.csv')
display(data.groupby('class_name').agg(crops=('crop_id','size'), all_wrong=('all_wrong','sum'),
                                      unanimous_wrong=('unanimous_wrong','sum'), all_correct=('all_correct','sum')))
worker = (root / 'src/dentalselfie/recovery_worker.py').read_text()
assert 'transforms.Resize((224, 224)' in worker
assert '(0.485,0.456,0.406), (0.229,0.224,0.225)' in worker
print('Recovery worker contract inspected: 224 RGB crops, ImageNet normalization.')
display(Markdown(f"**Observed:** {summary['all_models_wrong']} crops were wrong for every verified model; "
                 f"{summary['unanimous_wrong']} had unanimous wrong predictions. "
                 "Technical checks passed within their stated scope. Qualified clinical review remains pending. "
                 "Patch-readout and partial-adaptation experiments remain exploratory next stages."))
''')
    finish_notebook('10_2_caries_full_image_audit.ipynb',
                    '# 10.2 Caries full-image audit\n\nExecuted existing-weight CPU inference, fixed stress tests and annotation-relative error analysis.',
                    'Show native clean-data precision-recall curves next to the operating-point diagnostics. '
                    'Native AP and fixed-threshold precision/recall answer different questions. '
                    'CPU re-evaluation is a new inference run; minor differences from historical GPU metrics are retained.', '''
from pathlib import Path
import pandas as pd
from PIL import Image
from IPython.display import display, Markdown
out = Path.cwd() / 'docs/reports/post09/caries'
metrics = pd.read_csv(out / 'native_detection_metrics.csv')
display(metrics[['model','variant','map50','map50_95','wall_seconds']])
for name in ['YOLOv8n','YOLO26n']:
    display(Markdown('**' + name + ': clean validation precision-recall curve**'))
    with Image.open(out / (name + '_clean') / 'BoxPR_curve.png') as im:
        preview=im.copy();preview.thumbnail((900,650));display(preview)
print('All 327 validation images evaluated for both models across four fixed variants. Test images opened: 0.')
''')

    metrics = pd.read_csv(REPORT / 'caries/native_detection_metrics.csv')
    images = pd.read_csv(REPORT / 'caries/per_image_errors.csv')
    totals = images.groupby(['model','variant'])[['ground_truth','true_positive','missed','duplicate','localization','unmatched']].sum()
    lines = ['# Post-09 audit findings - 22 October 2026', '',
             'Both 10.1 and 10.2 executed locally with real saved outputs. No training or Kaggle GPU job was started. '
             'Test images opened: zero. No clinical labels changed.', '', '## Gingival audit', '',
             f"Ten full prediction tables agree with the locked 1,920-row validation population. "
             f"All ten models are wrong on {gum['all_models_wrong']} ROIs; {gum['unanimous_wrong']} have the same wrong label across models. "
             'This is model error agreement, not evidence that the source labels are wrong.', '',
             'No-visible support: 210 train ROIs from 87 observable groups, 49 validation ROIs from 23 groups. '
             'No no-visible ROI is correct for every one of the ten models. Counts do not certify patient independence.', '',
             'Mapping, source annotation identity, manifest membership, validation crop checksums, probability decoding, '
             'saved metrics and selected history epoch passed the executed assertions. Recovery-worker resize/normalization '
             'contracts were inspected. This does not certify clinical labels or exhaustively rule out every implementation defect.', '',
             'The review sheet contains 114 cases: every no-visible validation crop, seeded samples from the other classes, '
             'and separately identified hard examples. Model predictions are omitted from the reviewer sheet. Clinical review fields '
             'remain empty pending a qualified reviewer. Eight review pages and parent/crop panels are saved.', '',
             'Quality strata are descriptive, not causal. Almost all crops fall in the lowest predeclared Laplacian-variance bin, '
             'so that proxy does not meaningfully separate quality in this dataset. No images were filtered.', '',
             '## Caries full-image validation', '',
             f"Both existing checkpoints were evaluated on all {caries['validation_images']} validation images and "
             f"{caries['validation_lesions']} source annotations, across clean, blur radius 1.5 px, brightness x0.65 and JPEG quality 35.", '',
             '| Model | Variant | mAP50 | mAP50-95 | Validator wall seconds |',
             '|---|---|---:|---:|---:|']
    for row in metrics.itertuples():
        lines.append(f'| {row.model} | {row.variant} | {row.map50:.2%} | {row.map50_95:.2%} | {row.wall_seconds:.1f} |')
    lines += ['', '### Fixed operating point', '',
              'Confidence >=0.25; one-to-one IoU >=0.5. Greedy matching checks unmatched GT first. '
              'A nonmatched prediction is an apparent annotation-relative false positive, not a clinically verified healthy negative.', '',
              '| Model / variant | TP | Missed GT | Duplicate | Localization | Unmatched |',
              '|---|---:|---:|---:|---:|---:|']
    for (model, variant), row in totals.iterrows():
        lines.append(f'| {model} / {variant} | {int(row.true_positive)} | {int(row.missed)} | '
                     f'{int(row.duplicate)} | {int(row.localization)} | {int(row.unmatched)} |')
    lines += ['', 'At the same confidence 0.25 on clean images, YOLOv8n has annotation-relative precision/recall '
              '78.71%/95.07%, while YOLO26n has 87.62%/85.96%. The slight AP advantage does not select a '
              'deployment operating point. Blur reduces fixed-threshold recall more than its small AP change suggests; '
              'scores and thresholds must be considered before selecting a final detector.', '',
              'There are 611 medium and 443 large annotated boxes and zero small boxes under these definitions. '
              'Consequently this validation set provides no estimate of small-box recall. Annotation box area is '
              'not necessarily the anatomical extent of a lesion.', '',
              'Lesion size uses the area after isotropic scaling to maximum edge 640: small <32^2, '
              'medium <96^2, otherwise large. Per-size output is recall, not size-specific AP. '
              'CPU timings do not establish phone latency. Historical GPU results remain intact; '
              'native PR operating points and floating-point/runtime behavior can differ from this CPU re-evaluation.', '',
              '## Next executable decision', '',
              'The explicit technical checks did not fail. Implement the bounded 11.1 patch-readout and 11.2 partial-adaptation '
              'comparisons under the predeclared plan, without adding losses/context modules. They may run independently after '
              'input and weight pinning. Clinical ambiguity and minority scarcity remain unresolved, so no improvement is promised.', '',
              'Keep caries as the provisional demo anchor, subject to capture feasibility and independently labelled phone-domain '
              'evaluation. Synthetic blur/lighting/JPEG tests cannot establish phone generalization. '
              'Personal pilot images stay local. Mobile implementation/export remains deferred.', '',
              '## Execution notes', '',
              'The initial notebook runner stopped while forwarding Unicode progress text to a cp1252 Windows console. '
              'Logging now explicitly uses UTF-8. Completed inference receipts and prediction hashes were checked before reuse. '
              'A diagnostic assertion also exposed the serializer category convention: in pinned Ultralytics 8.4.183, '
              'non-COCO exported category IDs start at 1, so exported category 1 maps to the single model class 0. '
              'This was verified against DetectionValidator.init_metrics and pred_to_json, then corrected in the diagnostic reader; '
              'no model weights, GT labels or native AP results were changed. JSON diagnostics use serialized rounded coordinates/scores.', '',
              '## Evidence', '',
              '- Executed notebooks: `notebooks/10_1_gingival_failure_audit.ipynb` and `notebooks/10_2_caries_full_image_audit.ipynb`.',
              '- `gingival/`: verified metrics, support, full error joins, reviewer sheet and galleries.',
              '- `caries/`: native metrics/curves, prediction JSON, local hash-verification receipts, per-image errors and per-lesion recall.',
              '- `local_runtime.json`: exact local package versions. Temporary stress images remain under `.venv/work/audit10/`.',
              '- Matching implementation has three focused tests covering empty inputs, duplicate partitioning and overlapping-GT assignment.']
    (REPORT / 'FINDINGS_2026-10-22.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    inventory = {}
    for path in [ROOT/'notebooks/10_1_gingival_failure_audit.ipynb', ROOT/'notebooks/10_2_caries_full_image_audit.ipynb',
                 *REPORT.rglob('*.csv'), *REPORT.rglob('summary.json'), *REPORT.rglob('audit_receipt.json'),
                 REPORT/'FINDINGS_2026-10-22.md']:
        inventory[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    (REPORT / 'collection_inventory.json').write_text(json.dumps(inventory, indent=2), encoding='utf-8')
    print('Saved combined findings and local evidence inventory')


if __name__ == '__main__':
    main()

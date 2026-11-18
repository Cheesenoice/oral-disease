"""Rebuild the supervisor report and figures from collected project evidence.

Run from any working directory with the project's .venv Python. No image datasets,
private captures, model weights, network APIs or test predictions are opened.
"""
from pathlib import Path
import hashlib
import json
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "report"
FIG = OUT / "figure"
INPUTS = {}


def read(path):
    source = ROOT / path
    INPUTS[path] = hashlib.sha256(source.read_bytes()).hexdigest()
    if source.suffix == ".json":
        return json.loads(source.read_text(encoding="utf-8"))
    if source.suffix == ".csv":
        return pd.read_csv(source)
    return source.read_text(encoding="utf-8")


def table(frame):
    """Small Markdown renderer: no optional tabulate dependency."""
    def value(v):
        return str(v).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(map(value, frame.columns)) + " |",
             "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    return "\n".join(lines + ["| " + " | ".join(value(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None)])


def save_figure(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def write(name, text):
    (OUT / name).write_text(text.strip() + "\n", encoding="utf-8")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size":10, "axes.spines.top":False, "axes.spines.right":False,
                         "axes.titleweight":"bold", "figure.facecolor":"white"})
    audit = read("docs/reports/data_preparation/audit_summary.csv")
    splits = read("docs/reports/data_preparation/split_summary.csv")
    exclusion = read("docs/reports/data_preparation/final_exclusion_reasons.csv")
    support = read("docs/reports/post09/gingival/class_group_support.csv")
    matrix = read("docs/reports/synthesis/roi_experiment_matrix.csv")
    native = read("docs/reports/post09/caries/native_detection_metrics.csv")
    operating = read("docs/reports/synthesis/caries_detector_matrix.csv")
    recovery = read("docs/reports/training/roi_recovery_comparison.json")
    bootstrap = read("docs/reports/training/roi_recovery_group_bootstrap.json")
    dino = read("docs/reports/training/dinov3_publication/11_2_full_v8/verification.json")
    probe = read("docs/reports/training/dinov3_publication/11_1_full_v8/verification.json")
    original = read("docs/reports/training/roi_method_comparison.json")
    read("docs/reports/data_preparation/DATA_CARD.md")
    read("src/configs/sources.json")
    assert len(matrix) == 17 and len(native) == 8
    assert support.query("split=='val'").crops.sum() == 1920
    assert dino["state"] == probe["state"] == "COMPLETE"
    assert dino["test_images_opened"] == probe["test_images_opened"] == 0

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(audit))
    axes[0].bar(x, audit.retained_candidates, label="Retained", color="#087e8b")
    axes[0].bar(x, audit.excluded_images, bottom=audit.retained_candidates, label="Excluded", color="#c8cdd3")
    axes[0].set_xticks(x, ["Caries", "Gingival"])
    for i, r in enumerate(audit.itertuples()):
        axes[0].text(i, r.source_images+80, f"{r.source_images:,}", ha="center")
    axes[0].set(ylabel="Source photographs", title="Audited source retention", ylim=(0,7100)); axes[0].legend()
    bottom=np.zeros(2)
    for split,color in [("train","#087e8b"),("val","#ed9b40"),("test","#9da4b0")]:
        values=[int(splits.loc[splits.task.eq(task) & splits.split.eq(split), "images"].iloc[0]) for task in ["caries","gum"]]
        axes[1].bar(x,values,bottom=bottom,label=split,color=color)
        for i,n in enumerate(values): axes[1].text(i,bottom[i]+n/2,str(n),ha="center",va="center",fontsize=9)
        bottom+=values
    axes[1].set_xticks(x,["Caries","Gingival"]);axes[1].set(title="Retained-image split",ylabel="Photographs")
    axes[1].legend();save_figure(fig,"01_data_retention_and_splits")

    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,split in zip(axes,["train","val"]):
        group=support[support.split.eq(split)].sort_values("class_id")
        ax.bar(["No-visible","Mild","Marked"],group.crops,color=["#ed9b40","#087e8b","#555b8b"])
        for i,row in enumerate(group.itertuples()): ax.text(i,row.crops+max(group.crops)*.025,f"{row.crops:,}\n{row.groups} groups",ha="center",fontsize=9)
        ax.set(ylim=(0,max(group.crops)*1.22),ylabel="Annotated crops",title=f"{split.title()}: {group.crops.sum():,} ROIs")
    fig.suptitle("Rare no-visible class; source groups are not certified patient IDs")
    save_figure(fig,"02_gingival_class_support")

    labels=matrix.stage+" | "+matrix.experiment
    fig,axes=plt.subplots(1,2,figsize=(15,8))
    y=np.arange(len(matrix))
    axes[0].barh(y,matrix.macro_f1*100,color="#087e8b",label="Macro-F1")
    axes[0].scatter(matrix.accuracy*100,y,color="#ed9b40",s=22,label="Accuracy",zorder=3)
    axes[0].set_yticks(y,labels,fontsize=8);axes[0].invert_yaxis();axes[0].set(xlim=(0,100),xlabel="Validation score (%)",title="All collected ROI comparisons")
    axes[0].legend(loc="lower right",fontsize=8)
    axes[1].barh(y,matrix.no_visible_recall*100,color="#555b8b")
    axes[1].set_yticks(y,labels,fontsize=8);axes[1].invert_yaxis();axes[1].set(xlim=(0,100),xlabel="Recall (%)",title="No-visible validation recall (49 crops)")
    save_figure(fig,"03_roi_results_and_minority_recall")

    rr={r["trial"]["id"]:r for r in recovery}
    deltas=[100*(rr["R1_raw_CE"]["macro_f1"]-rr["R0_old_CE"]["macro_f1"]),
            100*(rr["R2_raw_CBCE"]["macro_f1"]-rr["R1_raw_CE"]["macro_f1"]),
            100*(rr["R3_raw_BalancedSoftmax"]["macro_f1"]-rr["R1_raw_CE"]["macro_f1"]),dino["group_bootstrap"]["macro_f1_delta_pp"]]
    intervals=[bootstrap["intervals_delta_macro_f1_percentage_points"][key] for key in ["R1-R0","R2-R1","R3-R1"]]+[dino["group_bootstrap"]["percentile_95_interval_pp"]]
    fig,ax=plt.subplots(figsize=(9,4))
    for i,(delta,(lo,hi)) in enumerate(zip(deltas,intervals)):
        ax.plot([lo,hi],[i,i],color="#087e8b",lw=3);ax.scatter(delta,i,color="#ed9b40",s=60,zorder=3)
        ax.text(hi+.2,i,f"{delta:+.2f} [{lo:+.2f}, {hi:+.2f}]",va="center",fontsize=9)
    ax.axvline(0,color="gray",ls="--");ax.set_yticks(range(4),["Raw vs old crops (R1-R0)","CB-CE vs CE (R2-R1)","Balanced Softmax vs CE (R3-R1)","DINO adaptation vs frozen CE"])
    ax.invert_yaxis();ax.set(xlim=(-9,11),xlabel="Macro-F1 difference (percentage points)",title="Paired source-group bootstrap: 2,000 draws")
    save_figure(fig,"04_matched_effects_and_uncertainty")

    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,record,title in [(axes[0],rr["R2_raw_CBCE"],"R2: ResNet18 + CB-CE"),
                            (axes[1],dino["verified_results"][1],"11.2: DINOv3 partial adaptation")]:
        cm=np.array(record["confusion_matrix"]);im=ax.imshow(cm,cmap="Blues",vmin=0,vmax=950)
        for i in range(3):
            for j in range(3):ax.text(j,i,str(cm[i,j]),ha="center",va="center",color="white" if cm[i,j]>475 else "black")
        ax.set(xticks=range(3),yticks=range(3),xticklabels=["No-visible","Mild","Marked"],yticklabels=["No-visible","Mild","Marked"],xlabel="Predicted",ylabel="Source label",title=title)
    save_figure(fig,"05_selected_confusion_matrices")

    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for name,group in native.groupby("model",sort=False):
        axes[0].plot(group.variant,group.map50_95*100,marker="o",label=name)
    axes[0].set(ylabel="mAP50-95 (%)",ylim=(0,100),title="Full-image validation and fixed stress");axes[0].legend()
    clean=operating[operating.variant.eq("clean")]
    x=np.arange(len(clean))
    axes[1].bar(x-.18,clean.precision_at_025*100,.36,label="Precision",color="#087e8b")
    axes[1].bar(x+.18,clean.recall_at_025*100,.36,label="Recall",color="#ed9b40")
    axes[1].set_xticks(x,clean.model);axes[1].set(ylim=(0,105),ylabel="Annotation-relative score (%)",title="Clean: confidence 0.25, matching IoU 0.5");axes[1].legend(loc="lower right")
    for i,row in enumerate(clean.itertuples()):
        axes[1].text(i-.18,row.precision_at_025*100+1,f"{row.precision_at_025*100:.1f}",ha="center",fontsize=8)
        axes[1].text(i+.18,row.recall_at_025*100+1,f"{row.recall_at_025*100:.1f}",ha="center",fontsize=8)
    save_figure(fig,"06_caries_robustness_and_operating_point")

    detector_rows=[]
    for eid,model in [("E0","YOLOv8n"),("E1","YOLO26n")]:
        for task in ["caries","gum"]:
            r=read(f"data/runs/{eid}_full_v4/experiments/{eid}_{task}_seed42.result.json")
            assert r["status"]=="completed" and not r["smoke_only"]
            m=r["metrics"]
            detector_rows.append([model,task,r["epochs_completed"],f"{m['metrics/mAP50(B)']*100:.2f}%",f"{m['metrics/mAP50-95(B)']*100:.2f}%"])
    detector_table=table(pd.DataFrame(detector_rows,columns=["Model","Task","Completed epochs","mAP50","mAP50-95"]))
    score_table=matrix[["stage","experiment","accuracy","macro_f1","no_visible_recall"]].copy()
    for c in ["accuracy","macro_f1","no_visible_recall"]:score_table[c]=score_table[c].map(lambda v:f"{100*v:.2f}%")
    score_table.columns=["Stage","Experiment","Accuracy","Macro-F1","No-visible recall"]
    split_table=table(splits[["task","split","images","groups","boxes"]])
    retention_table=table(audit)
    support_table=table(support[["split","class_name","crops","groups"]])
    native_display=native[["model","variant","map50","map50_95"]].copy()
    for c in ["map50","map50_95"]:native_display[c]=native_display[c].map(lambda v:f"{v*100:.2f}%")
    catalog=read("README.md")
    notebook_rows=[]
    for line in catalog.splitlines():
        match=re.match(r"\| \[([^]]+)\]\((notebooks/[^)]+)\) \| (.*?) \| (.*?) \|",line)
        if match:
            label,path,purpose,state=match.groups()
            assert (ROOT/path).is_file()
            notebook_rows.append([f"[{label}](../{path})",purpose,state])
    notebook_table=table(pd.DataFrame(notebook_rows,columns=["Notebook","Task and research role","Actual status"]))

    write("README.md",f'''# DentalSelfie: Supervisor Progress Report

**Reporting date: 23 October 2026. Evidence cutoff: completed work through 22 October 2026.**

## 1. Purpose and current position

DentalSelfie investigates AI-assisted analysis of ordinary oral photographs, with an eventual offline mobile application. The present report covers data preparation, object detection, gingival ROI classification, method development, error analysis and a local capture-feasibility pipeline. Development is temporarily paused at the student's request while this report is prepared.

The strongest practical evidence is source-domain caries detection. Gingival three-class grading remains unresolved: the best observed validation macro-F1 is **57.40%**, and more complex foundation-model variants did not surpass the ResNet18 references. This is a progress report, not a claim that a clinically usable mobile system has been delivered.

## 2. Reading guide

| Chapter | Contents |
|---|---|
| [01 - Data and problem formulation](01_DATA_AND_PROTOCOL.md) | Fixed sources, exclusions, labels, splits, crop preparation and evaluation units |
| [02 - The research journey](02_RESEARCH_JOURNEY.md) | Questions, methods, reasons for choices, measured outcomes and subsequent decisions |
| [03 - Results and interpretation](03_RESULTS_AND_LIMITATIONS.md) | Complete result tables, uncertainty, confusion matrices and validity limits |
| [04 - Notebook and artifact index](04_NOTEBOOK_AND_ARTIFACT_INDEX.md) | Every current notebook, its task, execution evidence and provenance |
| [05 - Supervisor discussion and next decisions](05_DISCUSSION_AND_NEXT_STEPS.md) | Contributions, unresolved issues, reporting narrative and paused next milestones |
| [06 - Technical walkthrough](06_TECHNICAL_WALKTHROUGH.md) | Detailed theory, image geometry, augmentation, model methods, DINO adaptation and actual demonstrations |

## 3. Project progression

```mermaid
flowchart TD
    A[Ordinary oral photo as intended input] --> B[Audit two fixed annotated datasets]
    B --> C[YOLO baselines for each task]
    C --> D[Promising source-domain caries detection]
    C --> E[Weak gingival detection and grading]
    E --> F[Scratch CNN and ResNet controls]
    F --> G[Context fusion and ordinal hypotheses]
    G --> H[No overall gain: inspect data and errors]
    H --> I[Raw-source crops and imbalance objectives]
    I --> J[Frozen DINOv3 and limited adaptation]
    J --> K[Matched gains but unresolved minority class]
    D --> L[Local phone capture pipeline]
    K --> M[Scope and annotation review]
    L --> N[Await ordinary phone captures]
    M --> O[Pause development and discuss evidence]
    N --> O
```

## 4. Key observations

| Question | Evidence at this checkpoint |
|---|---|
| Is data preparation complete? | Two sources audited; 2,175 caries images and 1,067 gingival images retained; splits locked |
| Was learning from scratch implemented? | Custom CNN and manually implemented ResNet18 trained with random initialization |
| Were methods developed beyond simple fine-tuning? | Context concatenation, learned gating, ordinal supervision, source-detail recovery, imbalance objectives and foundation readout/adaptation were studied |
| Did every improvement attempt work? | No. Several increased complexity or minority recall while reducing macro-F1 |
| Best observed gingival reference? | R2 ResNet18 + class-balanced CE: 69.11% accuracy, 57.40% macro-F1; R1 is near-tied at 57.26% macro-F1 |
| Caries result? | Full-image CPU re-evaluation: YOLO26n mAP50 94.12%, mAP50-95 73.01%; operating-point trade-offs remain |
| Is a phone demo validated? | No personal phone captures processed; both existing detectors passed a three-image source-validation plumbing check |
| What remains unopened? | The project held-out test split is not used for model selection or final performance claims |

## 5. Report structure and reproduction

```text
report/
  README.md
  01_DATA_AND_PROTOCOL.md
  02_RESEARCH_JOURNEY.md
  03_RESULTS_AND_LIMITATIONS.md
  04_NOTEBOOK_AND_ARTIFACT_INDEX.md
  05_DISCUSSION_AND_NEXT_STEPS.md
  script/build_report.py
  figure/                         PNG and SVG versions of six figures
  evidence_manifest.json          Input hashes and generation checks
```

From the project root:

```powershell
.venv\\Scripts\\python.exe report/script/build_report.py
```

The script regenerates the Markdown and charts from existing evidence. It reads collected tables and manifests, performs no training or downloads, and opens no dataset images or private captures. Mermaid blocks require a Markdown viewer with Mermaid support; PNG figures are embedded directly. Source archives and detailed experiment artifacts remain in their original locations. Relative links are intended for use within this project folder.
''')

    write("01_DATA_AND_PROTOCOL.md",f'''# 1. Data and Problem Formulation

## 1.1 Intended application and implemented research tasks

The intended input is an ordinary phone photograph of visible teeth and accessible oral surfaces. The research currently separates **caries localization** from **gingival appearance grading**. The datasets do not provide complete cross-disease annotations, so an image from one source must not be treated as a negative for the other disease.

| Task | Input used in experiments | Output | Evaluation |
|---|---|---|---|
| Caries detection | Complete source photograph | Boxes for visible_caries | mAP50, mAP50-95 and annotation-relative precision/recall |
| Gingival detection | Complete source photograph | Boxes with three appearance groups | Detection AP; localization and classification jointly matter |
| Gingival ROI classification | Annotation-defined local crop; sometimes paired context | No-visible, mild or marked appearance | Accuracy, macro-F1, per-class recall and ordinal diagnostics |
| Phone feasibility | Future ordinary local captures | Candidate boxes and qualitative review | No accuracy without independent reference annotations |

An annotation-provided crop removes the need to find the region. Consequently crop-classification accuracy cannot be compared directly with detector AP or interpreted as end-to-end application performance.

## 1.2 Fixed sources and provenance

| Source | Role | Provenance |
|---|---|---|
| D1: [Zenodo 14827784](https://doi.org/10.5281/zenodo.14827784), `Dataset.zip` | Caries images and rectangle annotations | Publisher MD5 verified; immutable original archive retained |
| D2: [Mendeley 3253gj88rr v1](https://doi.org/10.17632/3253gj88rr.1), *A dental intraoral image dataset of gingivitis for image captioning* | Gingival photographs and region annotations | User-downloaded original RAR; local SHA-256 recorded; no publisher checksum available in project metadata |

The recorded source metadata declares CC BY 4.0. Attribution is retained in derived datasets. Alternative oral datasets discussed initially were not silently substituted into these experiments. Sources contain supported clinical photography and have not been established as a phone-selfie population.

## 1.3 Audit, exclusions and normalization

{retention_table}

![Figure 1 - Dataset retention and split](figure/01_data_retention_and_splits.png)

**Figure 1.** Counts come from the completed audit and split summary. Test counts describe the locked partition; they do not imply test evaluation.

| Audit operation | Reason and implementation |
|---|---|
| Archive verification | Preserve immutable source bytes and checksum provenance |
| Annotation schema and overlays | Check class meanings, coordinate conventions, actual geometry and representative overlays before remapping |
| Missing annotations | Quarantine; do not invent healthy labels |
| Unknown shapes/classes and invalid geometry | Record reasons; do not silently clip or guess semantics |
| Exact and near duplicates | Inspect conflicts and observable source constraints before split assignment |
| RGB preparation | JPEG quality 95, 4:4:4, maximum edge 1280; no upsampling or augmentation during normalization |
| Orientation | Retain supported stored orientation after checks; quarantine unresolved nonstandard EXIF rotations |

Recorded exclusion reasons:

{table(exclusion)}

Reasons can overlap on one image. In particular, caries reason counts must not be summed as if they were mutually exclusive. The authoritative excluded-image total is 4,090. The 28 gingival exact-duplicate annotation conflicts were excluded in their entirety.

## 1.4 Label mapping

| Source annotation | Prepared label | Interpretation |
|---|---|---|
| Caries D/d rectangle | visible_caries | Positive annotated region; missing boxes do not establish absence |
| Gingival source 0/1 | Excluded from grading | Jaw ROIs rather than target appearance classes |
| Gingival source 2 | 0: no_visible_inflammation | Source-defined visible appearance, not a verified healthy diagnosis |
| Gingival source 3/4 | 1: mild_inflammation | Merged source appearance group |
| Gingival source 5/6 | 2: marked_inflammation | Merged source appearance group |

The ordering motivates an ordinal-learning hypothesis. It does not imply equally spaced clinical severity measurements or complete periodontal diagnosis.

## 1.5 Locked splits and observation units

{split_table}

Approximately 70/15/15 splits were constructed with seed 42 and observable grouping constraints. They differ from the source providers' original splits. Source filenames, original split names, raw/prepared hashes and annotation provenance remain traceable. Original-source directory names such as `Test` do not alone identify membership in the project's held-out test partition; the locked manifest does.

Grouping uses available filenames, hashes and conservative visual constraints. Verified patient IDs are unavailable: **patient independence is not claimed**. Multiple gingival crops from one photograph are correlated observations.

## 1.6 From whole images to paired ROI views

```mermaid
flowchart LR
    A[Locked train or validation image] --> B[Verified annotation box]
    B --> C[Local ROI]
    B --> D[Context box expanded by factor 1.5]
    C --> E[RGB crop resized to 224 by 224]
    D --> F[Gray padding when required then 224 by 224]
    E --> G[Local input]
    F --> H[Context input]
    G --> I[Same source label and inherited group split]
    H --> I
```

Notebook 05 generated **10,889 train/validation ROI pairs and 21,778 views**: 8,969 training ROIs and 1,920 validation ROIs. The held-out test was not used for this ROI derivation. Context is auxiliary; the label belongs to the central ROI.

{support_table}

![Figure 2 - Class support](figure/02_gingival_class_support.png)

**Figure 2.** Only 210 training and 49 validation ROIs have the no-visible label. Their source-group support is 87 and 23, respectively. A group can contain several classes, so class-specific group counts must not be summed to obtain unique groups.

The initial pipeline cropped after whole-image preparation at maximum edge 1280. The later audit found median training ROI dimensions of about 100 by 102 pixels at that stage versus 569 by 582 pixels in original-source coordinates. Notebook 08 therefore extracted raw-source crops before resizing to 224. Crop identities, labels and split membership were held fixed; the raw-source dataset contains 10,889 local crops.

## 1.7 Training and evaluation contract

Training augmentation is applied only to training inputs; validation uses deterministic preprocessing. ROI models use RGB tensors and ImageNet normalization. Exact recipes vary by experiment and are recorded in notebooks: early comparisons should not be described as perfectly controlled where augmentation, seed or stopping settings differ.

Macro-F1 gives each of the three classes equal weight; accuracy weights each crop equally and is dominated by the two common classes. Balanced accuracy is mean class recall. AP evaluates detection across confidence thresholds and overlap criteria. A fixed-confidence precision/recall table answers a different question from AP.

Detailed sources: [data card](../docs/reports/data_preparation/DATA_CARD.md), [source config](../src/configs/sources.json), [label mapping](../src/configs/label_mappings.json), [ROI config](../src/configs/roi.json).
''')

    write("02_RESEARCH_JOURNEY.md",'''# 2. The Research Journey: Questions, Choices and Evidence

## 2.1 Initial choice: an image that a user can plausibly capture

The original ambition combined mobile AI, image analysis and health support. The scope was narrowed to visible oral photographs because they can plausibly be acquired with a phone. Deep cancer diagnosis or an additional RNA modality would require inputs beyond this intended capture workflow. Offline Android remains the eventual deployment target; the current work establishes the data and modeling evidence first.

Two annotated sources were selected rather than inventing a joint disease label space. Caries and gingival models were trained separately. This made missing cross-disease labels an explicit constraint of the design.

## 2.2 Establish a full-image reference before adding complexity

**Question:** Can compact existing detectors localize the annotated regions?

Notebooks 03/04 trained YOLOv8n and YOLO26n separately for caries and gingival appearance. Successful full-v4 artifacts are the valid baseline evidence. The caries results were promising within the source domain; gingival detection remained much weaker, with mAP50-95 near 29%. This motivated a separate grading study using known ROIs, without assuming that all detector errors were classification errors.

**Decision:** retain detection as the full-image reference and study gingival classification as a clearly labelled subproblem. The research chronology starts with YOLO; the later CNN study must not be presented as having preceded those runs.

## 2.3 Build and understand the baseline models

| Model | Construction | What the comparison asks |
|---|---|---|
| C0 | Custom CNN trained from random weights | Can a compact learned visual hierarchy establish a baseline? |
| C1 | Manually implemented ResNet18 trained from random weights | Does residual learning provide a useful architecture? |
| C2 | The same ResNet18 implementation with checked ImageNet initialization | What changes when initialization transfers prior visual features? |

The baseline notebooks expose data inspection, resize and augmentation, tensor conversion, normalization, model layers, training loops and evaluation. This supports explaining implementation rather than showing only a library fine-tuning call.

C0 achieved 71.61% accuracy but 48.32% macro-F1 and zero no-visible recall. C1/C2 reached 55.42%/55.76% macro-F1. The first lesson was that overall accuracy concealed poor class balance. C1 was close to C2; the experiment did not show an overwhelming pretrained advantage.

## 2.4 First method hypothesis: combine local detail, context and ordered labels

**Motivation:** a tight crop may omit neighboring tissue; the three appearance labels have an order. These observations motivated explicit components that could be tested separately.

```mermaid
flowchart TD
    L[Local crop] --> E[Shared ResNet18 encoder]
    C[Expanded context crop] --> E
    E --> FL[Local feature]
    E --> FC[Context feature]
    FL --> M1[Concatenation and CE head: M1]
    FC --> M1
    FL --> G[Learned feature fusion gate]
    FC --> G
    G --> M2[CE head: M2]
    FL --> M3[Ordinal head: M3]
    G --> M4[Ordinal head: M4]
```

| Component | Practical role | Evidence |
|---|---|---|
| Concatenation | Expose both feature vectors to a classifier | M1 macro-F1 53.60% |
| Learned gating | Learn how the two views contribute to a fused representation | M2 macro-F1 53.99% |
| CORAL-style ordinal head | Learn ordered cumulative decisions rather than three unrelated class scores | M3 macro-F1 51.46%; no-visible recall improved but mild recall fell |
| Gating plus ordinal head | Test the proposed combination | M4 macro-F1 49.48%; the combination did not win |

These are literature-inspired, task-specific implementations, not claims of globally novel modules or exact paper reproduction. In particular, the ordinal implementation is a monotone shared-score adaptation and contains a redundant unused output in the historical v1 code. The research plan records the reference basis.

**Decision:** do not select M4 just because it is more elaborate. Keep the failures and inspect the pipeline. The 07.x screening has seed/RNG, augmentation, stopping and training-layout confounders, so differences from C2 cannot isolate each component causally. The report preserves this limitation rather than rewriting history as a clean factorial study.

## 2.5 Second hypothesis: perhaps preprocessing removed useful detail

Notebook 08 compared crop geometry before and after whole-image downscaling. It showed a substantial reduction in available source pixels. The next experiment changed crop extraction while retaining the same ROI IDs, labels, splits and downstream input size.

```mermaid
flowchart LR
    A[Original source image] --> B[Whole-image resize to max edge 1280]
    B --> C[Old ROI crop then resize to 224]
    A --> D[Verified source-coordinate ROI crop]
    D --> E[Raw-source ROI resized to 224]
    C --> F[R0: ResNet18 with CE]
    E --> G[R1: same matched recipe with CE]
    F --> H[Compare identical validation crop IDs]
    G --> H
```

R1 improved macro-F1 from 56.61% to 57.26%, a 0.66-point change. The group-bootstrap interval crossed zero and no-visible recall decreased. Detail preservation was a justified engineering correction to investigate, but it did not solve the grading problem.

## 2.6 Third hypothesis: the training objective may underweight the rare class

Notebook 09.2 compared raw-crop CE with two train-count-derived objectives. Class-balanced CE uses effective-number weights, proportional to `(1 - beta) / (1 - beta^n_c)`, with beta 0.999 in the implementation. Balanced Softmax adjusts training logits using class counts. Neither invents new examples or changes validation labels.

R2 class-balanced CE reached 57.40% macro-F1, only 0.13 points above R1. R3 Balanced Softmax reached 53.36%. **Decision:** retain R1/R2 as close references; do not claim a reliable winner from the small R2 advantage.

## 2.7 Fourth hypothesis: stronger pretrained representations may separate the classes

Notebook 09.3 compared frozen ResNet18 and frozen DINOv3 features with a scaled logistic classifier. Freezing the encoder was a bounded way to test representation quality before expensive adaptation. Valid version 3 gave macro-F1 50.08% and 47.83%, respectively. The newer foundation encoder was not automatically better on this task.

Notebook 10.1 then verified prediction identities, source mappings, crop hashes and errors. Ten model tables shared 99 wrong ROIs, including 79 unanimous wrong predictions. This motivated a review set of 114 cases, but agreement among models does not establish that source labels are wrong. Clinical review remains pending.

## 2.8 Refine the foundation-model question rather than change many factors at once

```mermaid
flowchart TD
    A[Pinned DINOv3 ViT-S/16 and raw crops] --> B[11.1: freeze encoder]
    B --> C[CLS feature]
    B --> D[CLS plus mean spatial patch feature]
    C --> E[Same scaler and logistic head grid]
    D --> E
    A --> F[11.2: CLS with same CE head]
    F --> G[Frozen encoder control]
    F --> H[Unfreeze last two blocks and final norm]
    G --> I[Matched training and selection]
    H --> I
```

11.1 increased feature dimensions from 384 to 768 and improved macro-F1 from 47.83% to 52.21%. Mean patch pooling excludes CLS and register tokens; it is not a learned tissue segmentation mask. This supports a readout effect on this validation population, but performance remained below R1/R2.

11.2 held the CLS readout and CE head fixed while adapting the final two transformer blocks and final normalization. Trainable parameters increased from 1,155 to 3,551,619. Macro-F1 rose from 44.16% to 47.64%, but both arms recognized 0 of 49 no-visible cases. The adapted branch early-stopped at epoch 21 and selected epoch 13. More training of the same recipe has no demonstrated solution to that failure.

**Decision:** retain the controlled findings, do not promote DINOv3 as a successful teacher and do not start distillation without a stronger teacher. The completed 11.2 result archive has predictions and logs but no exported adapted weights; checkpoints were written to temporary Kaggle storage.

## 2.9 Return to the intended input and practical scope

Notebook 10.2 evaluated the existing caries detectors on full validation images and fixed corruptions. Notebook 12.1 consolidated the evidence. Notebook 12.2 implemented local photo inspection, existing-weight inference and qualitative review. It passed a three-image source-validation plumbing check with both detectors; actual phone captures are still pending at the report cutoff.

**Current conclusion:** caries remains a provisional demo direction. Gingival grading provides a substantial research study with measured limitations. Neither a source-data AP score nor an ROI accuracy score establishes successful ordinary-phone deployment. The next decision should be informed by capture feasibility and annotation review.

Evidence: [original plan and reference basis](../KE_HOACH_DO_AN.md), [07.x protocol audit](../docs/reports/training/ROI_METHOD_FINDINGS_2026-10-21.md), [09.x findings](../docs/reports/training/ROI_09_COMBINED_FINDINGS_2026-10-22.md), [11.x full findings](../docs/reports/training/dinov3_publication/FULL_FINDINGS_2026-10-22.md).
''')

    write("03_RESULTS_AND_LIMITATIONS.md",f'''# 3. Results, Interpretation and Validity

## 3.1 Full-image detector baselines: original Kaggle v4

{detector_table}

These are original successful GPU-run metrics. The later full-image audit re-evaluated the existing weights on CPU; its small numerical differences are retained rather than merged with the original results. Native AP precision/recall values also differ from the explicitly fixed-confidence matching analysis below.

## 3.2 Gingival ROI comparison

{table(score_table)}

![Figure 3 - ROI results](figure/03_roi_results_and_minority_recall.png)

**Figure 3.** Every classifier row refers to the same 1,920 validation crop identities, but crop preparation, heads and training protocols vary. The 09.3 DINO CLS and 11.1 CLS rows repeat the same representation/control question; they are not independent evidence of generalization. Most runs use one seed; 07.x declared seeds and shared-thread RNG handling introduce additional confounding.

The highest accuracy belongs to the scratch CNN (71.61%), yet it misses every no-visible crop. R2 has the highest observed macro-F1 (57.40%), while R1 is close (57.26%). These differences demonstrate why selecting solely by accuracy would misrepresent the three-class objective.

## 3.3 What the matched comparisons support

![Figure 4 - Matched effects](figure/04_matched_effects_and_uncertainty.png)

**Figure 4.** Points are observed macro-F1 differences; intervals are paired source-group bootstrap percentile intervals with 2,000 draws. Groups are observable source constraints, not verified patients. Intervals condition on validation-selected models and exclude training-seed and model-selection uncertainty.

| Comparison | Change | Interpretation |
|---|---:|---|
| R1 raw crops vs R0 old crops | +0.66 pp | Small observed gain; interval includes zero |
| R2 CB-CE vs R1 CE | +0.13 pp | Near tie; no established superiority |
| R3 Balanced Softmax vs R1 CE | -3.90 pp | Worse in this controlled screening run |
| 11.1 patch readout vs CLS | +4.38 pp | Useful readout signal, still below ResNet18 reference; no interval shown here |
| 11.2 adaptation vs frozen CE | +3.48 pp | Matched gain, but no-visible recall remains zero |

The 75% macro-F1 target discussed during planning is an engineering aspiration, not an externally validated clinical threshold.

## 3.4 Class-specific errors

![Figure 5 - Confusion matrices](figure/05_selected_confusion_matrices.png)

**Figure 5.** Rows are source labels and columns are predictions. Supports are 49 no-visible, 932 mild and 939 marked ROIs. R2 recovers 20 no-visible crops; adapted DINOv3 recovers none. R2 still has substantial confusion between mild and marked appearance, so it should not be presented as solving the task.

The historical 07.x ordinal model M3 detects 32/49 no-visible crops, but no-visible precision is only 18.08% and mild recall falls to 39.38%. Improving one class's recall by predicting it more often can reduce the overall balance of errors. Ordinal output monotonicity did not guarantee lower observed ordinal error.

## 3.5 Full-image caries robustness audit

{table(native_display)}

![Figure 6 - Caries robustness and operating point](figure/06_caries_robustness_and_operating_point.png)

**Figure 6.** All eight audit evaluations use 327 validation images with 1,054 annotated boxes. Stress variants are blur radius 1.5, brightness multiplier 0.65 and JPEG quality 35. The left plot uses a full 0-100 scale to avoid exaggerating small AP changes. These are synthetic corruptions, not independent phone photographs.

At confidence 0.25 and one-to-one IoU 0.5, YOLOv8n precision/recall is **78.71%/95.07%**, while YOLO26n is **87.62%/85.96%**. The choice therefore depends on the intended operating trade-off; slightly higher AP does not automatically select a deployment threshold. Unmatched predictions are apparent annotation-relative false positives, not verified healthy tissue.

The audit contains 611 medium and 443 large boxes, and no small boxes under the specified 640-scale area bins. It provides no estimate of small-box recall. Annotation box size is not a direct measure of anatomical lesion extent.

## 3.6 What has and has not been demonstrated

| Evidence category | Established | Not established |
|---|---|---|
| Data integrity | Recorded hashes, mapping checks, overlays, exclusions and locked memberships | Exhaustive clinical correctness or certified patient independence |
| Model implementation | Scratch/pretrained controls, fusion/ordinal branches, recovery and DINO experiments | A novel globally superior algorithm or exact reproduction of every cited paper |
| Classification | Recomputed metrics from collected evidence; later full predictions checked against locked IDs | Successful full-image gingival localization or clinical readiness |
| Caries detection | Source-validation AP, error decomposition and fixed stress results | Ordinary-phone generalization or small-lesion performance |
| Robustness and uncertainty | Source-group bootstrap for selected comparisons; deterministic synthetic stress | Independent external validation, repeated-seed stability or final test confirmation |
| Deployment | Existing-weight local CPU pipeline and private capture workflow | Android implementation, quantization, on-phone latency or memory measurements |

## 3.7 Reproducibility and execution limitations

- The 6.2 version-2 initialization failure is excluded; corrected v3 is used. 09.3 v2 had training disabled; valid v3 supplies the probe results, and cancelled v4 supplies no result.
- Early detector full-v3 runs were cancelled after long unproductive sessions. The original cause was not established; valid v4 completed with bounded workers. Do not equate cancelled runs with converged experiments.
- The 07.x screening differs from C2 in augmentation, stopping, batch execution and RNG isolation; it is not a clean factorial ablation. These issues are documented in the original findings.
- 11.x required corrected access to gated weights and compatible HF runtime packages. Both full v8 runs completed; 11.2 took 34 min 15 s, including 682.5 s of image-hash verification before training.
- 11.2 best/resume weights remained in Kaggle temporary storage. Collected predictions and metrics are valid, but no reusable adapted inference checkpoint exists in the downloaded archive.
- The validation set has guided many decisions. All findings are exploratory until the protocol is frozen and independent confirmation is performed. The project test split remains unused for outcome claims.

Complete receipts: [11.1](../docs/reports/training/dinov3_publication/11_1_full_v8/verification.json), [11.2](../docs/reports/training/dinov3_publication/11_2_full_v8/verification.json), [09.x uncertainty](../docs/reports/training/roi_recovery_group_bootstrap.json), [post-09 audit](../docs/reports/post09/FINDINGS_2026-10-22.md).
''')

    write("04_NOTEBOOK_AND_ARTIFACT_INDEX.md",f'''# 4. Notebook and Artifact Index

## 4.1 Complete current notebook register

The following {len(notebook_rows)} canonical notebooks are listed in the project README. A canonical training source can have run gates disabled or lack local outputs; collected remote executed copies are the evidence of successful training. The status column records the valid run rather than assuming that the latest remote version is always the valid one.

{notebook_table}

## 4.2 How to navigate the study

```mermaid
flowchart LR
    A[01 audit] --> B[02 preparation]
    B --> C[02b publication]
    C --> D[03 and 04 detectors]
    D --> E[05 paired ROI data]
    E --> F[6.1 6.2 6.3 baselines]
    F --> G[07.1 and 07.2 methods]
    G --> H[08 source-detail audit]
    H --> I[09.1 crop comparison]
    H --> J[09.2 objectives]
    H --> K[09.3 frozen probes]
    I --> L[10.1 grading audit]
    J --> L
    K --> L
    D --> M[10.2 caries audit]
    L --> N[11.1 readout and 11.2 adaptation]
    N --> O[12.1 synthesis]
    M --> O
    O --> P[12.2 local capture feasibility]
```

Arrows represent evidence dependencies and the narrative sequence, not a claim that every notebook must execute serially. Independent tracks were run on separate GPU workers where appropriate.

## 4.3 Execution and artifact locations

| Stage | Primary evidence |
|---|---|
| Source audit and preparation | [data card](../docs/reports/data_preparation/DATA_CARD.md), [audit records](../docs/reports/data_preparation/), `data/manifests/` and immutable `data/raw/` |
| Detector full v4 | [E0 collection](../docs/reports/training/E0_full_v4/collection_receipt.json), [E1 collection](../docs/reports/training/E1_full_v4/collection_receipt.json); `data/runs/E0_full_v4/` and `E1_full_v4/` |
| 6.x baselines | [baseline comparison](../docs/reports/training/roi_baseline_comparison.json), [findings](../docs/reports/training/ROI_BASELINE_FINDINGS_2026-10-21.md) |
| 07.x methods | [comparison](../docs/reports/training/roi_method_comparison.json), [protocol limitations](../docs/reports/training/ROI_METHOD_FINDINGS_2026-10-21.md) |
| 09.x recovery/probes | [combined findings](../docs/reports/training/ROI_09_COMBINED_FINDINGS_2026-10-22.md), [recovery metrics](../docs/reports/training/roi_recovery_comparison.json), [valid probe results](../docs/reports/training/roi_foundation_probe_comparison.json) |
| 10.x audit | [findings](../docs/reports/post09/FINDINGS_2026-10-22.md), [114-case reviewer sheet](../docs/reports/post09/gingival/reviewer_sheet.csv) |
| 11.x full v8 | [combined findings](../docs/reports/training/dinov3_publication/FULL_FINDINGS_2026-10-22.md); [executed 11.1](../data/runs/ROI_11_1_full_v8/executed.ipynb), [executed 11.2](../data/runs/ROI_11_2_full_v8/executed.ipynb) |
| 12.x synthesis | [synthesis findings](../docs/reports/synthesis/FINDINGS.md), [phone feasibility guide](../docs/PHONE_PHOTO_FEASIBILITY.md) |

## 4.4 Engineering work supporting the experiments

| Component | Why it was needed |
|---|---|
| Immutable raw archives and provenance | Make data transformations traceable |
| Split locks and hashes | Detect accidental changes and retain group membership |
| Notebook-first presentation | Expose processing and model logic with explanatory Markdown and actual outputs |
| Shared Python modules | Reuse implementation without diverging notebook copies; teaching baselines intentionally remain self-contained |
| Kaggle T4 x2 execution | Run expensive training with bounded, isolated workers and explicit configuration |
| Smoke checks and gradient checks | Catch access, shape, checkpoint and trainable-layer failures before interpreting full results |
| Collection and local recomputation | Verify IDs, probabilities, confusion matrices and selected checkpoints rather than trusting a dashboard alone |
| Public/private artifact separation | Publish research evidence while excluding personal captures, credentials and restricted base weights |

Notebook 12.2 is **local-only** and may contain personal images once captures are supplied. Its current smoke uses source-validation images, but it must still not be uploaded or casually bundled with the supervisor report. This report contains aggregate charts only and does not copy private photo material.

## 4.5 Rebuild and audit this report

`script/build_report.py` generates all report Markdown plus six PNG/SVG figure pairs. `evidence_manifest.json` records input file hashes and structural checks. The script uses installed NumPy, pandas and Matplotlib in the project environment and requires no new training or cloud access. Re-running replaces the report's generated files; edit narrative text in the script if regeneration is desired.
''')

    write("05_DISCUSSION_AND_NEXT_STEPS.md",'''# 5. Supervisor Discussion and Next Decisions

## 5.1 What the work contributes so far

The current contribution is a reproducible, task-specific experimental study: audited oral datasets, traceable preparation, compact detector references, from-scratch model implementation, explicit method hypotheses, controlled recovery comparisons and failure analysis. The effort includes diagnosing unsuccessful ideas and documenting why they were not promoted.

The strongest result is not a newly combined gingival model. It is the evidence that several plausible combinations fail to improve the current grading task, while source-domain caries localization is considerably more promising. Reporting this distinction protects the scientific value of the work and informs a feasible application scope.

## 5.2 Suggested explanation during the supervisor meeting

| Step | Explanation | Evidence to show |
|---|---|---|
| 1 | We chose visible oral photographs to match an ordinary-phone input constraint. | Task definitions and source table |
| 2 | We audited the annotations instead of assuming every available image was a usable labelled example. | Retention/split chart and exclusion ledger |
| 3 | We established detectors first; weak gingival results led to a separate grading investigation. | Four detector baselines |
| 4 | We built scratch CNN/ResNet controls, then studied initialization. | 6.x notebook code and results |
| 5 | We tested context and ordered labels through explicit variants. The combination did not win. | M1-M4 table and protocol caveats |
| 6 | We investigated preprocessing and imbalance instead of only increasing model size. | Source-crop diagram and R0-R3 comparisons |
| 7 | We tested a foundation representation, spatial readout and limited adaptation. Improvements remained insufficient. | 09.3 and 11.x results; minority confusion matrix |
| 8 | We audited common errors and prepared a real-capture study. | Reviewer sheet, source-domain caries audit and current readiness |
| 9 | We need a scope decision based on feasibility and label review before claiming a completed mobile solution. | Evidence gaps and decision flow below |

Do not say that performance improved at every stage, that a later architecture is inherently better, or that the experiments reproduce all cited papers exactly. The valid narrative is a sequence of hypotheses tested against evidence.

## 5.3 Decisions to discuss with the supervisor

1. **Primary demonstration scope:** retain caries as the provisional main demonstration, subject to ordinary-photo feasibility. Keep gingival grading as a limited research branch unless stronger evidence emerges.
2. **Clinical annotation review:** arrange qualified review of the existing 114-case sheet. Model consensus alone is insufficient to change labels. Preserve original labels and document any future revision separately.
3. **Dataset extension:** if the capture domain differs substantially, define an expert-annotated, consented extension with a separate evaluation protocol rather than mixing unlabeled personal photos into existing splits.
4. **Final evaluation:** freeze the selected method, endpoints and operating thresholds before any held-out test evaluation. Repeated-seed work should target an evidence-supported candidate and control rather than every model.
5. **Offline deployment:** resume Android work only after model/data feasibility is established. On-phone optimization, export and latency remain future work.

## 5.4 Paused next-stage decision flow

```mermaid
flowchart TD
    A[Supervisor reviews current evidence] --> B[Agree primary task and claims]
    B --> C[Collect ordinary local phone captures]
    B --> D[Qualified review of gingival ambiguity]
    C --> E[Run existing local inference notebook]
    E --> F{Capture and domain feasible?}
    F -->|Yes| G[Define reference labels and external evaluation]
    F -->|No| H[Revise capture protocol or justified data extension]
    D --> I{Evidence-backed correction or new labels available?}
    I -->|Yes| J[Predeclare one matched experiment]
    I -->|No| K[Keep gingival limits explicit]
    G --> L[Freeze finalist protocol]
    J --> L
    H --> B
    L --> M[Repeated-seed confirmation and final test]
    M --> N[Conditional offline mobile implementation]
```

This diagram is a proposed decision process, not a list of completed work or authorization to resume training. Development is paused for this reporting checkpoint. No real-phone accuracy, final-test performance, repeated-seed superiority or on-device benchmark has been established.

## 5.5 Concise progress statement

> We completed the data and modeling research pipeline through detector baselines, scratch/pretrained classifiers, context and ordinal variants, preprocessing and imbalance studies, foundation-model probes, limited adaptation and error audits. The caries detectors show strong source-validation localization, while gingival three-class grading remains below the project target and has persistent minority-class errors. We have prepared a local phone-capture feasibility pipeline and paused implementation to review scope and evidence with the supervisor.
''')

    # Link and structure checks catch broken embedded figures and missing local evidence.
    broken=[]
    for path in OUT.glob("*.md"):
        text=path.read_text(encoding="utf-8")
        assert text.count("```") % 2 == 0, path.name
        for target in re.findall(r"\]\(([^)]+)\)",text):
            if not target.startswith(("http://","https://","#")) and not (path.parent/target).exists():
                broken.append((path.name,target))
    assert not broken, broken
    receipt={"report_date":"2026-10-23","evidence_cutoff":"2026-10-22","inputs":INPUTS,
             "notebooks_indexed":len(notebook_rows),"roi_rows":len(matrix),"detector_audit_rows":len(native),
             "figure_pairs":6,"local_links_verified":True,"training_started":False,
             "dataset_images_opened":False,"private_captures_opened":False,
             "python":sys.version.split()[0],"matplotlib":matplotlib.__version__,"pandas":pd.__version__}
    (OUT/"evidence_manifest.json").write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf-8")
    print(f"Built 6 Markdown reports, 6 PNG/SVG figure pairs; indexed {len(notebook_rows)} notebooks. All local links verified.")


if __name__ == "__main__":
    main()

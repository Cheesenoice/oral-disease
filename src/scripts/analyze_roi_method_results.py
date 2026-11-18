"""Verify collected ROI method predictions and produce the validation findings."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "docs/reports/training"
EXPECTED_SHA = "8ed4c05240a2d23e6c254ee74adf13af48ff45a4418f1e2024bd38498b02cab1"


def metrics_from_matrix(matrix):
    cm = np.asarray(matrix, dtype=float)
    support, predicted = cm.sum(1), cm.sum(0)
    recall = np.divide(cm.diagonal(), support, out=np.zeros(3), where=support > 0)
    precision = np.divide(cm.diagonal(), predicted, out=np.zeros(3), where=predicted > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(3), where=precision + recall > 0)
    distance = np.abs(np.arange(3)[:, None] - np.arange(3)[None, :])
    expected = np.outer(support, predicted) / cm.sum()
    return {"accuracy": float(cm.trace() / cm.sum()), "macro_f1": float(f1.mean()),
            "balanced_accuracy": float(recall.mean()), "class_precision": precision.tolist(),
            "class_recall": recall.tolist(), "class_f1": f1.tolist(),
            "ordinal_mae": float((cm * distance).sum() / cm.sum()),
            "quadratic_kappa": float(1 - (cm * distance ** 2).sum() /
                                      (expected * distance ** 2).sum()),
            "extreme_errors": int(cm[0, 2] + cm[2, 0])}


def main():
    path = ROOT / "data/processed/gum_roi_v1/manifest.csv"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == EXPECTED_SHA
    manifest = pd.read_csv(path)
    val = manifest[manifest.split == "val"].set_index("crop_id")
    baseline = json.loads((REPORT / "roi_baseline_comparison.json").read_text())
    rows, details = [], {}
    for track, item in baseline.items():
        label = {"6.1": "C0 scratch CNN", "6.2": "C1 scratch ResNet18",
                 "6.3": "C2 ImageNet ResNet18"}[track]
        measured = metrics_from_matrix(item["validation"]["confusion_matrix"])
        rows.append({"experiment": label, **{k: v for k, v in measured.items()
                     if not isinstance(v, list)}, "epochs": item["epochs_completed"],
                     "selected_epoch": item["selected_epoch"],
                     "runtime_minutes": item["elapsed_seconds"] / 60})
        details[label] = measured
    method_results = []
    for track in ("07_1", "07_2"):
        run = ROOT / "data/runs" / f"ROI_{track}_v1" / "dentalselfie_07"
        for metric_path in sorted(run.glob("*/metrics.json")):
            item = json.loads(metric_path.read_text())
            assert item["manifest_sha256"] == EXPECTED_SHA
            folder = metric_path.parent
            pred = pd.read_csv(folder / "validation_predictions.csv")
            assert pred.crop_id.is_unique and set(pred.crop_id) == set(val.index)
            assert np.array_equal(val.loc[pred.crop_id, "class_id"], pred.true_class)
            matrix = np.zeros((3, 3), dtype=int)
            for true, predicted in zip(pred.true_class, pred.predicted_class):
                matrix[int(true), int(predicted)] += 1
            assert matrix.tolist() == item["confusion_matrix"]
            measured = metrics_from_matrix(matrix)
            for key in ("accuracy", "macro_f1", "balanced_accuracy", "ordinal_mae", "quadratic_kappa"):
                assert np.isclose(measured[key], item[key]), (item["experiment"], key)
            probabilities = np.array(pred.probabilities.map(json.loads).tolist())
            assert np.isfinite(probabilities).all()
            if item["ordinal"]:
                assert (probabilities[:, 0] >= probabilities[:, 1]).all()
                assert np.array_equal((probabilities > 0.5).sum(1), pred.predicted_class)
            else:
                # Probabilities were computed in float16 autocast before float32 export.
                assert np.allclose(probabilities.sum(1), 1, atol=1e-3)
                assert np.array_equal(probabilities.argmax(1), pred.predicted_class)
            history = pd.read_csv(folder / "history.csv")
            best = history.loc[history.macro_f1.idxmax()]
            assert int(best.epoch) == item["selected_epoch"]
            assert np.isclose(best.macro_f1, item["macro_f1"])
            assert len(history) == item["epochs"] == item["selected_epoch"] + 5
            detail = {**item, **measured, "validation_rows": len(pred),
                      "validation_parents": val.parent_id.nunique(),
                      "steady_epoch_seconds_median": float(history.seconds.diff().median()),
                      "stop_reason": "Early stopping: five epochs without a new best macro-F1",
                      "selected_train_loss": float(best.train_loss),
                      "selected_val_loss": float(best.val_loss),
                      "last_train_loss": float(history.iloc[-1].train_loss),
                      "last_val_loss": float(history.iloc[-1].val_loss)}
            method_results.append(detail)
            details[item["experiment"]] = detail
            rows.append({"experiment": item["experiment"],
                         **{k: v for k, v in measured.items() if not isinstance(v, list)},
                         "epochs": item["epochs"], "selected_epoch": item["selected_epoch"],
                         "runtime_minutes": item["runtime_seconds"] / 60})
    frame = pd.DataFrame(rows)
    frame["delta_macro_f1_pp_vs_C2"] = (frame.macro_f1 - details["C2 ImageNet ResNet18"]["macro_f1"]) * 100
    frame.to_csv(REPORT / "roi_method_comparison.csv", index=False)
    (REPORT / "roi_method_comparison.json").write_text(
        json.dumps({"manifest_sha256": EXPECTED_SHA, "validation_rows": len(val),
                    "validation_parents": int(val.parent_id.nunique()),
                    "summary": frame.to_dict("records"), "details": details}, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    labels = frame.experiment.str.replace("_", " ").tolist()
    colors = ["#a0a0a0", "#909090", "#236393", "#5995bd", "#5995bd", "#c18b50", "#c18b50"]
    axes[0].bar(labels, frame.macro_f1 * 100, color=colors)
    axes[0].set_ylabel("Validation macro-F1 (%)")
    axes[0].axhline(details["C2 ImageNet ResNet18"]["macro_f1"] * 100,
                    color="#236393", linestyle="--", linewidth=1)
    axes[0].set_ylim(0, 65)
    axes[0].tick_params(axis="x", rotation=55)
    compare = ["C2 ImageNet ResNet18", *[x["experiment"] for x in method_results]]
    x = np.arange(3)
    for i, name in enumerate(compare):
        axes[1].bar(x + (i - 2) * .15, np.array(details[name]["class_recall"]) * 100,
                    width=.15, label=name)
    axes[1].set_xticks(x, ["No visible", "Mild", "Marked"])
    axes[1].set_ylabel("Validation recall (%)"); axes[1].set_ylim(0, 100)
    axes[1].legend(fontsize=7)
    fig.suptitle("ROI screening results: gains in minority recall do not improve macro-F1")
    fig.tight_layout()
    fig.savefig(REPORT / "roi_method_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    lines = ["# ROI Method Findings - 21 October 2026", "",
             "Both public version-1 Kaggle kernels completed. All four sets of 1,920 validation predictions were rechecked against the locked manifest and labels; confusion matrices, accuracy, macro-F1, balanced accuracy, ordinal MAE and quadratic weighted kappa match the saved metrics. Ordinal probabilities are monotone and decode to the saved predictions. No test data was read.", "",
             "## Measured comparison", "",
             "| Experiment | Accuracy | Macro-F1 | Balanced accuracy | MAE | QWK | Best / total epochs | Minutes |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['experiment']} | {r['accuracy']*100:.2f}% | {r['macro_f1']*100:.2f}% | {r['balanced_accuracy']*100:.2f}% | {r['ordinal_mae']:.3f} | {r['quadratic_kappa']:.3f} | {r['selected_epoch']}/{r['epochs']} | {r['runtime_minutes']:.1f} |")
    lines += ["", "## Findings", "",
        "- C2 remains the best measured macro-F1 reference (55.76%). M1, M2, M3 and M4 trail it by 2.16, 1.77, 4.30 and 6.28 percentage points respectively. These single-run differences are descriptive, not causal evidence.",
        "- M2 gating slightly exceeds M1 concatenation in macro-F1 (53.99% vs 53.60%) and increases no-visible recall, but its accuracy decreases. The average gate value near 0.45 does not establish which anatomical context is useful.",
        "- M3 has the highest balanced accuracy (63.86%) and no-visible recall (32/49 = 65.31%), but mild recall falls to 39.38%. No-visible precision is only 18.08% (32/177). Predicting the rare class more often is not the same as improving overall grading.",
        "- M4 has the highest marked recall (88.60%), but the lowest method macro-F1 (49.48%). It predicts marked for 462/932 mild ROIs. The combined method is not the current winner.",
        "- CORAL did not improve measured ordinal error: MAE is 0.373 for M3/M4 vs 0.304 for C2. Extreme 0-to-2 or 2-to-0 errors are 0 for C2, 3 for M1, 8 for M2, 12 for M3 and 18 for M4. Monotone cumulative outputs do not guarantee better ordinal classification.",
        "- All four workers stopped normally after five stale epochs: best/total epochs 4/9, 7/12, 7/12 and 10/15. Worker runtimes were 22.1-26.4 minutes. Kernel logs end near 26.3 and 27.4 minutes. Neither the 20-epoch ceiling nor 70-minute worker budget was reached.",
        "- Training loss kept decreasing while validation loss increased. For example M2 ends at train/validation CE 0.117/1.080. This is evidence of a growing generalization gap; extending the same runs blindly is not supported. CE and CORAL loss magnitudes should not be compared directly.", "",
        "## Protocol audit and limitations", "",
        "The version-1 experiments are exploratory screening, not a fully controlled factorial ablation. The implementation introduced the following confounders; correct them before a final method claim:", "",
        "1. C2 uses 30 max epochs, patience 8, color jitter and DataParallel; 07 uses 20 max epochs, patience 5, no color jitter, one GPU per trial and gradient accumulation. An effective batch of 64 does not guarantee the same BatchNorm or optimizer dynamics (including the partial final batch).",
        "2. M1/M3 declare seed 42, while M2/M4 declare seed 43. Concurrent threads also reset global Python/NumPy/PyTorch RNG state, so the declared per-worker seeds are not reliably isolated. This prevents an exact same-seed comparison and bitwise reproducibility.",
        "3. The fusion heads introduce dropout and different parameter counts. Shared-encoder paired forwards update BatchNorm twice per optimizer microbatch. These are additional architectural/training changes beyond the loss label.",
        "4. The ordinal head is a monotone shared-score adaptation, not an exact published CORAL reproduction. The published 07.2 v1 creates two score channels but uses only channel zero; the unused row is redundant and should be removed. It does not invalidate the recomputed predictions, but the implementation must be documented and cleaned up.",
        "5. Only 49 validation ROIs belong to no-visible inflammation, and multiple ROIs come from each of 160 source photographs. They are not 1,920 independent patients. No clinical diagnostic sensitivity or specificity is inferred.",
        "6. ROI grading assumes annotation-provided localization. These numbers are not detector mAP or end-to-end phone-camera results. Test remains locked.", "",
        "## Next executable action", "",
        "Keep these v1 artifacts as real screening evidence. First prepare a matched local-CE control and matched local-ordinal control with the same encoder/head conventions, seed list, augmentation, microbatch/accumulation, early stopping and validation rule. Isolate worker RNGs (prefer separate processes) and clean the unused ordinal output. Then compare context concatenation/gating under that identical recipe. Use additional seeds for the selected comparison; do not start a new run or tune on test as part of this analysis request.", "",
        "If class imbalance is investigated, declare it as a separate train-derived loss/sampling ablation. Do not add it silently and attribute any improvement to context or ordinal supervision. Keep C2 as the provisional model rather than selecting M4 for its complexity.", "",
        "The full artifact bundles are in `data/runs/ROI_07_1_v1/` and `data/runs/ROI_07_2_v1/`. Raw public source, metadata and execution logs are in matching folders under `docs/reports/training/`. The first CLI downloads returned a Windows log-encoding error after saving data and checkpoints; logs were recovered with UTF-8 enabled. No training failure is implied."]
    (REPORT / "ROI_METHOD_FINDINGS_2026-10-21.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(frame.to_string(index=False))
    print("Verified all four prediction files and wrote findings, CSV, JSON and chart.")


if __name__ == "__main__":
    main()

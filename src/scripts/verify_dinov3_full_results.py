"""Verify collected full v8 predictions without loading images or model weights."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil

import nbformat
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "data/runs/ROI_11_2_full_v8"
SOURCE = RUN / "dentalselfie-11-2"
REPORT = ROOT / "docs/reports/training/dinov3_publication"
OUTPUT = REPORT / "11_2_full_v8"


def metrics(cm):
    tp = np.diag(cm)
    support = cm.sum(1)
    predicted = cm.sum(0)
    precision = np.divide(tp, predicted, out=np.zeros(3), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros(3), where=support > 0)
    denominator = support + predicted
    f1 = np.divide(2 * tp, denominator, out=np.zeros(3), where=denominator > 0)
    return dict(accuracy=float(tp.sum() / cm.sum()), macro_f1=float(f1.mean()),
                balanced_accuracy=float(recall.mean()), class_precision=precision.tolist(),
                class_recall=recall.tolist(), class_f1=f1.tolist(),
                support=support.tolist(), confusion_matrix=cm.tolist())


def main():
    OUTPUT.mkdir(exist_ok=True)
    manifest = ROOT / "data/processed/gum_raw_roi_v1/manifest.csv"
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    val = pd.read_csv(manifest).query("split == 'val'").set_index("crop_id").sort_index()
    assert len(val) == 1920
    nb = nbformat.read(RUN / "executed.ipynb", as_version=4)
    nbformat.validate(nb)
    assert not [o for c in nb.cells for o in c.get("outputs", []) if o.output_type == "error"]
    executed = sum(c.get("execution_count") is not None for c in nb.cells)
    assert executed > 10
    results, tables, histories = [], {}, {}
    for trial in ["frozen_CE", "last2_blocks_norm_CE"]:
        folder = SOURCE / trial
        saved = json.loads((folder / "metrics.json").read_text())
        pred = pd.read_csv(folder / "validation_predictions.csv").set_index("crop_id").sort_index()
        history = pd.read_csv(folder / "history.csv")
        assert pred.index.is_unique and pred.index.equals(val.index)
        np.testing.assert_array_equal(pred.true_class, val.class_id)
        probabilities = np.stack(pred.probabilities.map(json.loads))
        assert probabilities.shape == (1920, 3) and np.isfinite(probabilities).all()
        assert ((probabilities >= 0) & (probabilities <= 1)).all()
        np.testing.assert_allclose(probabilities.sum(1), 1, atol=1e-6)
        np.testing.assert_array_equal(probabilities.argmax(1), pred.predicted_class)
        cm = np.zeros((3, 3), dtype=int)
        np.add.at(cm, (pred.true_class.to_numpy(), pred.predicted_class.to_numpy()), 1)
        checked = metrics(cm)
        for key, value in checked.items():
            np.testing.assert_allclose(value, saved["metrics"][key], atol=1e-12)
        assert saved["phase"] == "full" and saved["test_images_opened"] == 0
        assert saved["manifest_sha256"] == digest
        assert saved["model_revision"] == "114c1379950215c8b35dfcd4e90a5c251dde0d32"
        assert len(history) == saved["epochs_completed"]
        assert int(history.loc[history.macro_f1.idxmax(), "epoch"]) == saved["selected_epoch"]
        np.testing.assert_allclose(history.macro_f1.max(), checked["macro_f1"], atol=1e-12)
        assert "Gradient routing check passed" in (SOURCE / f"{trial}.log").read_text()
        names = saved["trainable_names"]
        assert all(n.startswith(("head.", "backbone.layer.10.", "backbone.layer.11.", "backbone.norm.")) for n in names)
        if trial == "frozen_CE":
            assert set(names) == {"head.weight", "head.bias"}
        target = OUTPUT / trial
        target.mkdir(exist_ok=True)
        for name in ["history.csv", "metrics.json", "validation_predictions.csv"]:
            shutil.copy2(folder / name, target / name)
        results.append(dict(trial=trial, **checked, **{k: saved[k] for k in
                       ["epochs_completed", "selected_epoch", "stop_reason", "elapsed_seconds", "trainable_parameters"]}))
        tables[trial], histories[trial] = pred, history
    for name in ["comparison.csv", "run_config.json"]:
        shutil.copy2(SOURCE / name, OUTPUT / name)

    # Observable source groups, not assumed patient IDs; use identical draws for both arms.
    groups, codes = np.unique(val.group_id.to_numpy(), return_inverse=True)
    draws = np.random.default_rng(42).multinomial(len(groups), np.full(len(groups), 1 / len(groups)), size=2000)
    scores = {}
    for name, pred in tables.items():
        gc = np.zeros((len(groups), 3, 3), dtype=int)
        np.add.at(gc, (codes, val.class_id.to_numpy(), pred.predicted_class.to_numpy()), 1)
        cms = np.einsum("bg,gij->bij", draws, gc)
        tp = np.diagonal(cms, axis1=1, axis2=2)
        denominator = cms.sum(1) + cms.sum(2)
        scores[name] = np.divide(2 * tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator > 0).mean(1)
    delta = scores["last2_blocks_norm_CE"] - scores["frozen_CE"]
    bootstrap = dict(comparison="last2_blocks_norm_CE minus frozen_CE", repetitions=2000, seed=42,
                     observable_groups=len(groups), macro_f1_delta_pp=100 * (results[1]["macro_f1"] - results[0]["macro_f1"]),
                     percentile_95_interval_pp=(100 * np.quantile(delta, [.025, .975])).tolist(),
                     note="Exploratory reused validation; observable groups are not verified patient IDs. Not seed-repeat uncertainty.")
    receipt = dict(kernel="trihuynhviprovcl/dentalselfie-11-2-dinov3-partial-adaptation", version=8,
                   script_version_id=361136686, state="COMPLETE", phase="full", runtime_seconds=2054.6,
                   collected_utc=datetime.now(timezone.utc).isoformat(), collection_method="Public browser exact-version download",
                   archive_sha256=hashlib.sha256((RUN / "kaggle_outputs.zip").read_bytes()).hexdigest(),
                   manifest_sha256=digest, verified_results=results, group_bootstrap=bootstrap,
                   executed_notebook_code_cells=executed, test_images_opened=0,
                   exported_weights_present=any(RUN.rglob("*.pt")))
    (OUTPUT / "verification.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    plot(histories, results)


def plot(histories, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    names = ["CLS + logistic", "CLS+patch + logistic", "Frozen + CE", "Last 2 blocks + CE"]
    probe = json.loads((REPORT / "11_1_full_v8/verification.json").read_text())["verified_results"]
    all_results = probe + results
    x = np.arange(4)
    axes[0].bar(x - .18, [r["accuracy"] * 100 for r in all_results], .36, label="Accuracy")
    axes[0].bar(x + .18, [r["macro_f1"] * 100 for r in all_results], .36, label="Macro-F1")
    axes[0].axhline(57.4, color="black", linestyle="--", label="Historical R2 macro-F1")
    axes[0].set_xticks(x, names, rotation=22, ha="right")
    axes[0].set_ylim(0, 85)
    axes[0].set_ylabel("Validation score (%)")
    axes[0].legend(fontsize=7)
    for name, history in histories.items():
        axes[1].plot(history.epoch, history.macro_f1 * 100, label=name)
        axes[2].plot(history.epoch, history.train_loss, label=name + " train")
        axes[2].plot(history.epoch, history.val_loss, linestyle="--", label=name + " val")
    axes[1].set_title("11.2 validation macro-F1")
    axes[1].set_xlabel("Epoch")
    axes[1].legend(fontsize=7)
    axes[2].set_title("11.2 CE loss")
    axes[2].set_xlabel("Epoch")
    axes[2].legend(fontsize=6)
    fig.suptitle("DINOv3 full v8: locked validation crops, seed 42")
    fig.tight_layout()
    fig.savefig(REPORT / "FULL_V8_COMPARISON.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()

"""Create a readable notebook that runs the local review audit."""
import nbformat as nb

b = nb.v4.new_notebook()
b.cells = [
    nb.v4.new_markdown_cell('# 16.8 - Frontal review readiness\n\nValidate reviewer decisions and box geometry before preparing a training dataset. Pending cases are excluded; the audit is not a clinical validation.'),
    nb.v4.new_code_cell('from pathlib import Path\nimport sys\nimport pandas as pd\nroot=Path.cwd().resolve()\nwhile not (root / "AGENTS.md").exists():\n    if root == root.parent: raise RuntimeError("Project root not found")\n    root=root.parent\nsys.path.insert(0, str(root / "src/scripts"))\nfrom audit_frontal_review import audit\nbase=root / "docs/reports/alignment/frontal_pilot_16_7"\nqueue=pd.read_csv(base / "review_queue.csv", dtype={"source_image_id":str})\ndisplay(queue.groupby(["split", "view_review"]).size().to_frame("images"))'),
    nb.v4.new_markdown_cell('## Geometry and review consistency\n\nCheck named reviewers, target/box agreement, original-image coordinates and frontal status. A candidate count is not approval to train; review provenance and dataset size remain separate decisions.'),
    nb.v4.new_code_cell('result=audit(base / "review_queue.csv",base / "boxes_template.csv")\nprint("Images:",result["images"],"| Boxes:",result["boxes"],"| Reviewed candidates:",result["reviewed_candidates"])\nprint("Consistency issues:",len(result["issues"]))\nfor issue in result["issues"]: print("-",issue)\nif result["reviewed_candidates"] == 0: print("Training is blocked: no reviewed candidates.")'),
    nb.v4.new_markdown_cell('## Next gate\n\nConfirm annotation rules with a dental reviewer, expand and review both classes and negative images, investigate same-scene relationships and source rights, then prepare a frozen split and independent phone evaluation. Do not infer unannotated findings as negatives.')
]
b.metadata.kernelspec = {'display_name':'DentalSelfie','language':'python','name':'dentalselfie-local'}
nb.write(b, 'notebooks/alignment/16_8_frontal_review_readiness.ipynb')

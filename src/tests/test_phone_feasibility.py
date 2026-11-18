import json
import zipfile
import pandas as pd
from PIL import Image
from dentalselfie.phone_feasibility import inspect_photos, sha
from dentalselfie.package import package_code


def test_capture_duplicate_overlap_consent_and_corrupt_file(tmp_path):
    Image.new("RGB", (24, 32), "white").save(tmp_path / "a.png")
    (tmp_path / "b.png").write_bytes((tmp_path / "a.png").read_bytes())
    (tmp_path / "broken.jpg").write_bytes(b"invalid image")
    pd.DataFrame([dict(file="a.png", view="front_teeth", consent="yes"),
                  dict(file="b.png", view="front_teeth", consent="")]).to_csv(tmp_path / "capture_manifest.csv", index=False)
    result = inspect_photos(tmp_path, tmp_path / "capture_manifest.csv", {sha(tmp_path / "a.png")}).set_index("file")
    assert "exact_bytes_overlap" in result.loc["a.png", "reason"]
    assert "duplicate_capture" in result.loc["b.png", "reason"]
    assert "consent_not_recorded" in result.loc["b.png", "reason"]
    assert result.loc["broken.jpg", "status"] == "unreadable_or_invalid"
    assert not result.status.eq("ready").any()


def test_code_archive_excludes_personal_notebook_outputs(tmp_path):
    config = tmp_path / "src/configs/kaggle.json"
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps(dict(owner="synthetic", code_slug="code")))
    books = tmp_path / "notebooks"
    books.mkdir()
    for name, metadata in [("public.ipynb", {}), ("12_2_phone_photo_feasibility.ipynb", {}),
                           ("other_private.ipynb", {"dentalselfie": {"local_only": True}})]:
        (books / name).write_text(json.dumps(dict(metadata=metadata, cells=[])))
    stage = package_code(tmp_path)
    with zipfile.ZipFile(stage / "project.zip") as archive:
        names = archive.namelist()
    assert "notebooks/public.ipynb" in names
    assert "notebooks/12_2_phone_photo_feasibility.ipynb" not in names
    assert "notebooks/other_private.ipynb" not in names

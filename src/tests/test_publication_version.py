"""Prevent a newly uploaded artifact from being bound to a stale ready version."""
import pytest
from dentalselfie.publishing import publication_ready


def test_old_ready_version_does_not_complete_new_upload():
    assert not publication_ready({"status": "ready", "current_version_number": 10}, 11)
    assert not publication_ready({"status": "processing", "current_version_number": 11}, 11)
    assert publication_ready({"status": "ready", "current_version_number": 11}, 11)


def test_concurrent_newer_publication_is_not_silently_adopted():
    with pytest.raises(RuntimeError, match="advanced beyond"):
        publication_ready({"status": "ready", "current_version_number": 12}, 11)

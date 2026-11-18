"""Execute a canonical notebook locally and preserve real outputs, including failures."""
import argparse
import os
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[2]


class ProgressClient(NotebookClient):
    """Keep upload progress visible without dumping notebook images into the terminal."""
    def process_message(self, msg, cell, cell_index):
        if msg["header"]["msg_type"] == "stream":
            content = msg["content"]["text"].strip()
            if content:
                print(content[-1800:], flush=True)
        return super().process_message(msg, cell, cell_index)


def main():
    # Windows redirected consoles may default to cp1252 while libraries emit
    # Unicode progress bars. Keep logging from interrupting notebook execution.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    args = parser.parse_args()
    path = args.notebook.resolve()
    if not path.is_relative_to(ROOT / "notebooks"):
        raise ValueError("Only project notebooks may be executed through this entry point")
    os.environ["JUPYTER_PATH"] = str(ROOT / ".venv/share/jupyter") + os.pathsep + os.environ.get("JUPYTER_PATH", "")
    notebook = nbformat.read(path, as_version=4)

    def progress(cell, cell_index, **kwargs):
        if cell.cell_type == "code":
            print(f"{path.name}: executing cell {cell_index + 1}/{len(notebook.cells)}", flush=True)

    def save_progress(cell, cell_index, **kwargs):
        # Publication can package this notebook mid-run. Keep completed evidence
        # on disk instead of bundling an older failed execution snapshot.
        nbformat.write(notebook, path)

    client = ProgressClient(notebook, timeout=3600, kernel_name="dentalselfie-local",
                            resources={"metadata": {"path": str(ROOT)}}, on_cell_start=progress,
                            on_cell_executed=save_progress)
    try:
        client.execute()
    finally:
        nbformat.write(notebook, path)
    print("Executed and saved:", path.name, flush=True)


if __name__ == "__main__":
    main()

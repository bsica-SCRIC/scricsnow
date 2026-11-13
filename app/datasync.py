"""Bring the `data` branch into ./data, so every reader stays a local read.

The collectors push to the `data` branch, which Streamlit Cloud does not
deploy (see .github/scripts/commit-data.sh for why). This is the other half:
the app downloads that branch as one archive and lays it over its own data/
folder. Nothing that reads data/ needs to know.

The copy in data/ on main is the floor. If GitHub is unreachable the files
already on disk stay where they are, so an outage reads as "a little stale",
never as a blank page.

Each file is swapped in with os.replace, which is atomic, so a reader never
sees half a CSV.
"""

import io
import os
import tarfile
import tempfile

import requests

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(ROOT, "data")
REPO = os.environ.get("SCRICSNOW_REPO", "bsica-SCRIC/scricsnow")
URL = f"https://codeload.github.com/{REPO}/tar.gz/refs/heads/data"


def pull(url=URL, dest=DATA_DIR, timeout=30):
    """Download the branch and swap its files into dest. Returns files copied.

    Raises on any failure, having changed nothing that was not already
    complete. The caller decides what a failure means.
    """
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    dest = os.path.abspath(dest)
    os.makedirs(dest, exist_ok=True)
    copied = 0
    # Extract beside dest, not in the system temp dir, so os.replace never
    # crosses a filesystem boundary.
    with tempfile.TemporaryDirectory(dir=os.path.dirname(dest)) as tmp:
        with tarfile.open(fileobj=io.BytesIO(r.content), mode="r:gz") as tar:
            tar.extractall(tmp, filter="data")
        (top,) = os.listdir(tmp)                  # scricsnow-data/
        top = os.path.join(tmp, top)
        for here, dirs, files in os.walk(top):
            # The branch root carries a stray .github/ and .gitignore.
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            rel = os.path.relpath(here, top)
            for name in files:
                if name.startswith("."):
                    continue
                target_dir = os.path.normpath(os.path.join(dest, rel))
                os.makedirs(target_dir, exist_ok=True)
                os.replace(os.path.join(here, name),
                           os.path.join(target_dir, name))
                copied += 1
    return copied

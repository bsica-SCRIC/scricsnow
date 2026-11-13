"""The data branch sync: what lands, what is skipped, and what a failure leaves."""

import io
import tarfile

import pytest

import datasync


def tarball(files):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, body in files.items():
            data = body.encode()
            info = tarfile.TarInfo(f"scricsnow-data/{name}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


class Reply:
    def __init__(self, content=b"", status=200):
        self.content, self.status = content, status

    def raise_for_status(self):
        if self.status != 200:
            raise RuntimeError(f"HTTP {self.status}")


def test_branch_files_land_and_dotfiles_do_not(tmp_path, monkeypatch):
    dest = tmp_path / "data"
    (dest / "outcomes").mkdir(parents=True)
    (dest / "outcomes" / "checks.csv").write_text("old")
    (dest / "calls").mkdir()
    (dest / "calls" / ".gitkeep").write_text("")
    body = tarball({"outcomes/checks.csv": "new",
                    "outcomes/calendar_2026.csv": "cal",
                    ".github/workflows/x.yml": "no",
                    ".gitignore": "no"})
    monkeypatch.setattr(datasync.requests, "get", lambda *a, **k: Reply(body))

    assert datasync.pull(dest=str(dest)) == 2
    assert (dest / "outcomes" / "checks.csv").read_text() == "new"
    assert (dest / "outcomes" / "calendar_2026.csv").read_text() == "cal"
    assert not (dest / ".github").exists()
    assert (dest / "calls" / ".gitkeep").exists()     # floor files untouched


def test_a_failed_download_changes_nothing(tmp_path, monkeypatch):
    dest = tmp_path / "data"
    dest.mkdir()
    (dest / "keep.csv").write_text("floor")
    monkeypatch.setattr(datasync.requests, "get",
                        lambda *a, **k: Reply(status=503))
    with pytest.raises(RuntimeError):
        datasync.pull(dest=str(dest))
    assert (dest / "keep.csv").read_text() == "floor"

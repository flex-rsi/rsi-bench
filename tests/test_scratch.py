"""Inventory, archive and purge of a run's scratch directory."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from rsibench.scratch import MANIFEST, archive, inventory, list_archives, purge


def _fake_scratch(root):
    hub = root / "cache" / "huggingface" / "hub" / "models--facebook--sam-vit-large"
    (hub / "snapshots" / "abc123").mkdir(parents=True)
    (hub / "blobs").mkdir()
    (hub / "blobs" / "w").write_bytes(b"x" * 1000)
    (root / "cache" / "pkgs" / "timm-1.0.9.dist-info").mkdir(parents=True)
    (root / "cache" / "pkgs" / "timm").mkdir()
    repo = root / "cache" / "repos" / "segment-anything"
    (repo / ".git" / "refs" / "heads").mkdir(parents=True)
    (repo / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    (repo / ".git" / "refs" / "heads" / "main").write_text("deadbeef\n")
    (repo / ".git" / "config").write_text(
        '[remote "origin"]\n\turl = https://github.com/facebookresearch/segment-anything\n')
    (root / "home").mkdir()


def test_inventory_lists_models_packages_and_repositories(tmp_path):
    scratch = tmp_path / "scratch"
    _fake_scratch(scratch)
    inv = inventory(scratch)
    assert inv["hf_models"] == [{"repo": "facebook/sam-vit-large", "type": "model",
                                 "revisions": ["abc123"], "bytes": 1000}]
    assert inv["python_packages"] == [{"name": "timm", "version": "1.0.9"}]
    assert inv["git_repos"] == [{"path": "cache/repos/segment-anything", "commit": "deadbeef",
                                 "remote": "https://github.com/facebookresearch/segment-anything"}]
    assert inv["bytes"] >= 1000 and {t["name"] for t in inv["top_level"]} == {"cache", "home"}
    assert inventory(tmp_path / "missing")["bytes"] == 0


def test_archive_moves_scratch_and_large_artifacts_then_purge_respects_retention(tmp_path):
    scratch, art, root = tmp_path / "scratch", tmp_path / "art", tmp_path / "archive"
    _fake_scratch(scratch)
    (art / "t" / "r001" / "e1").mkdir(parents=True)
    (art / "t" / "r001" / "e1" / "big.npz").write_bytes(b"0" * 3_000_000)
    (art / "t" / "r001" / "e1" / "metrics.json").write_text("{}")
    (art / "t" / "transcripts").mkdir()
    (art / "t" / "transcripts" / "r001-executor.jsonl").write_bytes(b"0" * 3_000_000)
    rec = archive(scratch, art, "run/x", "run--x", max_file_mb=1, retention_days=7, root=root)
    dest = root / rec["archive"]
    assert not scratch.exists() and (dest / "scratch" / "home").is_dir()
    assert (dest / "artifacts" / "t" / "r001" / "e1" / "big.npz").exists()
    assert (art / "t" / "r001" / "e1" / "metrics.json").exists()  # small files stay
    assert (art / "t" / "transcripts" / "r001-executor.jsonl").exists()  # transcripts stay
    assert rec["large_artifacts"] == 1 and json.loads((dest / MANIFEST).read_text())["run_id"] == "run/x"
    assert [a["archive"] for a in list_archives(root)] == [rec["archive"]]
    assert purge(root) == [] and dest.exists()  # not expired yet
    later = datetime.now(timezone.utc) + timedelta(days=8)
    assert [a["archive"] for a in purge(root, now=later)] == [rec["archive"]]
    assert not dest.exists()


def test_purge_force_deletes_one_run_now(tmp_path):
    root = tmp_path / "archive"
    for name in ("a", "b"):
        s = tmp_path / name
        s.mkdir()
        archive(s, tmp_path / "none", f"run/{name}", f"run--{name}", 5, root=root)
    assert [a["run_id"] for a in purge(root, run="run/a", force=True)] == ["run/a"]
    assert [a["run_id"] for a in list_archives(root)] == ["run/b"]

"""Tests for scripts/download_dclm_bundle.py.

Network isn't available in CI for the actual DCLM bundle, so the download
path is mocked. We verify the sha + unzip + manifest + cleanup logic
end-to-end against a synthetic local zip.
"""
from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ralph_bootstrap  # noqa: F401
from scripts.download_dclm_bundle import (
    _build_parser,
    download_and_verify,
    sha256_of_file,
)


def _make_fake_zip(path: Path, contents: dict[str, bytes]) -> str:
    """Write a fake zip; return its sha256 hex."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in contents.items():
            zf.writestr(name, data)
    return sha256_of_file(path)


def _patch_download(monkeypatch, source_zip: Path):
    """Monkey-patch _download to copy source_zip to dest instead of network."""
    import shutil

    import scripts.download_dclm_bundle as mod

    def fake_download(url, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(source_zip, dest)

    monkeypatch.setattr(mod, "_download", fake_download)


# ============================================================================
# sha256_of_file
# ============================================================================


def test_sha256_of_file_known_content(tmp_path):
    path = tmp_path / "x.bin"
    path.write_bytes(b"hello world")
    expected = (
        "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
    )
    assert sha256_of_file(path) == expected


def test_sha256_chunked_large_file(tmp_path):
    """A file larger than _CHUNK reads through multiple iterations."""
    path = tmp_path / "big.bin"
    path.write_bytes(b"x" * (1 << 17))  # 128 KB
    h = sha256_of_file(path)
    assert len(h) == 64


# ============================================================================
# download_and_verify happy path
# ============================================================================


def test_download_and_verify_happy_path(tmp_path, monkeypatch):
    src = tmp_path / "src.zip"
    sha = _make_fake_zip(src, {"task_a.jsonl": b'{"x":1}\n', "task_b.jsonl": b''})
    _patch_download(monkeypatch, src)
    out_dir = tmp_path / "out"
    result = download_and_verify(
        url="https://example.com/eval_bundle.zip",
        output_dir=out_dir,
        expected_sha=sha,
    )
    assert result["sha256"] == sha
    assert result["extracted_count"] == 2
    assert (out_dir / "task_a.jsonl").read_text() == '{"x":1}\n'
    assert (out_dir / "SHA256SUM").exists()
    # Zip auto-deleted by default
    assert not (out_dir / "eval_bundle.zip").exists()


def test_download_and_verify_keeps_zip_when_requested(tmp_path, monkeypatch):
    src = tmp_path / "src.zip"
    _make_fake_zip(src, {"a.jsonl": b''})
    _patch_download(monkeypatch, src)
    out_dir = tmp_path / "out"
    download_and_verify(
        url="x",
        output_dir=out_dir,
        keep_zip=True,
    )
    assert (out_dir / "eval_bundle.zip").exists()


def test_download_and_verify_sha_mismatch_raises(tmp_path, monkeypatch):
    src = tmp_path / "src.zip"
    _make_fake_zip(src, {"a.jsonl": b''})
    _patch_download(monkeypatch, src)
    with pytest.raises(ValueError, match=r"SHA mismatch"):
        download_and_verify(
            url="x",
            output_dir=tmp_path / "out",
            expected_sha="0" * 64,
        )


def test_download_and_verify_no_expected_sha_accepts_any(tmp_path, monkeypatch):
    src = tmp_path / "src.zip"
    _make_fake_zip(src, {"a.jsonl": b''})
    _patch_download(monkeypatch, src)
    result = download_and_verify(
        url="x",
        output_dir=tmp_path / "out",
        expected_sha=None,
    )
    assert result["sha256"]  # set


def test_manifest_schema(tmp_path, monkeypatch):
    src = tmp_path / "src.zip"
    _make_fake_zip(src, {"a.jsonl": b''})
    _patch_download(monkeypatch, src)
    out_dir = tmp_path / "out"
    download_and_verify(url="https://example.com/foo.zip", output_dir=out_dir)
    manifest = json.loads((out_dir / "SHA256SUM").read_text())
    assert manifest["_meta"] == "ralph-dclm-bundle-manifest"
    assert manifest["url"] == "https://example.com/foo.zip"
    assert len(manifest["sha256"]) == 64
    assert manifest["extracted_member_count"] == 1
    assert manifest["downloaded_at_iso"]


# ============================================================================
# CLI
# ============================================================================


def test_cli_default_url_is_pinned_constant():
    from eval.downstream.core22 import DCLM_EVAL_BUNDLE_URL
    parser = _build_parser()
    args = parser.parse_args([])
    assert args.url == DCLM_EVAL_BUNDLE_URL


def test_cli_overrides(tmp_path):
    parser = _build_parser()
    args = parser.parse_args([
        "--url", "https://x.example/y.zip",
        "--output-dir", str(tmp_path / "out"),
        "--expected-sha", "a" * 64,
        "--keep-zip",
    ])
    assert args.url == "https://x.example/y.zip"
    assert args.output_dir == tmp_path / "out"
    assert args.expected_sha == "a" * 64
    assert args.keep_zip is True


def test_cli_main_succeeds(tmp_path, monkeypatch):
    """CLI main returns 0 on a successful (mocked) download."""
    from scripts.download_dclm_bundle import main
    src = tmp_path / "src.zip"
    _make_fake_zip(src, {"a.jsonl": b''})

    with mock.patch("scripts.download_dclm_bundle._download") as fake_dl:
        def _copy(url, dest):
            import shutil
            shutil.copy(src, dest)
        fake_dl.side_effect = _copy
        rc = main([
            "--url", "x",
            "--output-dir", str(tmp_path / "out"),
        ])
    assert rc == 0

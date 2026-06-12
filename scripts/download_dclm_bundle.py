"""Download + SHA-verify + unzip the DCLM CORE-22 eval bundle.

One-time operational step (per DEFERRED.md B1-D2). Pulls the bundle from
the pinned S3 URL (`eval/downstream/core22.py::DCLM_EVAL_BUNDLE_URL`),
verifies the SHA256 (if an expected digest is supplied), and unzips to
the local mirror at `eval/private/downstream_pool/bundle_v1/`.

After running this script, **manually** update the
`DCLM_EVAL_BUNDLE_SHA256` constant in `eval/downstream/core22.py` with
the digest printed below. A one-line PR closes B1-D2's last open item.

USAGE:
    python scripts/download_dclm_bundle.py \\
        [--url <override>] \\
        [--output-dir eval/private/downstream_pool/bundle_v1] \\
        [--expected-sha <hex>] \\
        [--keep-zip]

OUTPUTS:
  eval/private/downstream_pool/bundle_v1/
    eval_bundle.zip          (kept iff --keep-zip)
    SHA256SUM                manifest with the digest + url + timestamp
    <task>.jsonl, ...        unzipped task files (DCLM-native schema)

The downloader does NOT re-key the per-task JSONLs into the canonical
Karpa schema — that's `cache_hf_assets.py`'s and a follow-up adapter's
concern. The bundle here is the raw upstream artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.downstream.core22 import DCLM_EVAL_BUNDLE_URL  # noqa: E402

_CHUNK = 1 << 16  # 64KB streaming chunks for both download and hash


def _download(url: str, dest: Path) -> None:
    """Stream-download `url` to `dest`. Overwrites if present."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as resp, dest.open("wb") as f:
        shutil.copyfileobj(resp, f)


def sha256_of_file(path: Path) -> str:
    """Lowercase hex SHA256 of a file, chunked to keep memory bounded."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _unzip(zip_path: Path, target_dir: Path) -> list[str]:
    """Unzip `zip_path` into `target_dir`. Returns the list of extracted
    member names for the operator's discovery report."""
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        zf.extractall(target_dir)
    return names


def _write_manifest(
    manifest_path: Path,
    *,
    url: str,
    sha256: str,
    extracted_count: int,
) -> None:
    """Write a small JSON manifest that the SHA-pin commit references."""
    manifest_path.write_text(json.dumps({
        "_meta": "karpa-dclm-bundle-manifest",
        "url": url,
        "sha256": sha256,
        "extracted_member_count": extracted_count,
        "downloaded_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, indent=2, sort_keys=True))


def download_and_verify(
    *,
    url: str,
    output_dir: Path,
    expected_sha: str | None = None,
    keep_zip: bool = False,
) -> dict:
    """Run the whole pipeline. Returns the manifest dict."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / "eval_bundle.zip"

    _download(url, zip_path)
    actual_sha = sha256_of_file(zip_path)

    if expected_sha is not None and expected_sha.lower() != actual_sha:
        raise ValueError(
            f"SHA mismatch: expected {expected_sha}, got {actual_sha}. "
            "Either the upstream bundle rotated or the network corrupted "
            "the download. Re-pull and re-verify before proceeding."
        )

    member_names = _unzip(zip_path, output_dir)
    manifest_path = output_dir / "SHA256SUM"
    _write_manifest(
        manifest_path, url=url, sha256=actual_sha,
        extracted_count=len(member_names),
    )

    if not keep_zip:
        zip_path.unlink()

    return {
        "url": url,
        "sha256": actual_sha,
        "extracted_count": len(member_names),
        "manifest_path": str(manifest_path),
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="scripts.download_dclm_bundle")
    p.add_argument("--url", default=DCLM_EVAL_BUNDLE_URL)
    p.add_argument("--output-dir", type=Path,
                   default=Path("eval/private/downstream_pool/bundle_v1"))
    p.add_argument("--expected-sha", default=None,
                   help="If supplied, abort on SHA mismatch.")
    p.add_argument("--keep-zip", action="store_true",
                   help="Retain eval_bundle.zip after unzip (default: delete).")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        result = download_and_verify(
            url=args.url,
            output_dir=args.output_dir,
            expected_sha=args.expected_sha,
            keep_zip=args.keep_zip,
        )
    except Exception as e:
        print(f"download_dclm_bundle FAILED: {type(e).__name__}: {e}",
              file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    print(
        f"\nNext step: update DCLM_EVAL_BUNDLE_SHA256 in "
        f"eval/downstream/core22.py with:\n  \"{result['sha256']}\"\n",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

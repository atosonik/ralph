"""
Submission bundle assembly + signing.

Phase 0 flow:
  1. miner.submit.request_handshake_nonce() — simulated on-chain handshake
     (Phase 0 = local "chain" = JSON file). Returns a fresh nonce committed
     to the chain alongside the patch hash and miner hotkey.
  2. Miner runs proof.runner with the nonce + patch.
  3. miner.submit.assemble_submission() — bundles outputs + signs.

The signer is Ed25519 over the bundle hash + handshake nonce + miner hotkey,
using a stub keypair stored in `miner/keys/`. Phase 0.5+ replaces this with
real Bittensor hotkey signing via the `bittensor` SDK.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import time
from dataclasses import asdict, dataclass
from pathlib import Path


# ----------------------------- chain stub -----------------------------------

# Phase 0 "chain" = JSON file under chain/. Every handshake commits an entry.
# Phase 0.5+ replaces this with Bittensor on-chain commitment.


def _chain_dir(karpa_root: Path) -> Path:
    d = karpa_root / "chain"
    d.mkdir(parents=True, exist_ok=True)
    return d


def request_handshake_nonce(
    karpa_root: Path,
    miner_hotkey: str,
    patch_hash: str,
) -> str:
    """Simulate the on-chain proof-test handshake (§5.4)."""
    chain = _chain_dir(karpa_root)
    nonce = "0x" + secrets.token_hex(32)
    entry = {
        "type": "proof_test_handshake",
        "timestamp": time.time(),
        "miner_hotkey": miner_hotkey,
        "patch_hash": patch_hash,
        "nonce": nonce,
    }
    handshakes = chain / "handshakes.jsonl"
    with handshakes.open("a") as f:
        f.write(json.dumps(entry) + "\n")
    return nonce


def lookup_handshake(
    karpa_root: Path,
    nonce: str,
) -> dict | None:
    """Validator-side: confirm a nonce was committed to the chain."""
    handshakes = _chain_dir(karpa_root) / "handshakes.jsonl"
    if not handshakes.exists():
        return None
    for line in handshakes.read_text().splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if entry.get("nonce") == nonce:
            return entry
    return None


# ----------------------------- signing --------------------------------------

# Phase 0 signer: ed25519 with a per-miner local keypair stored in miner/keys/.
# Phase 0.5+: replaced by real Bittensor hotkey signing (substrate sr25519).


def _ensure_keypair(karpa_root: Path, miner_hotkey: str) -> tuple[bytes, bytes]:
    """Return (private_key_bytes, public_key_bytes). Generates and persists on
    first use."""
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey,
            Ed25519PublicKey,
        )
        from cryptography.hazmat.primitives.serialization import (
            Encoding,
            PrivateFormat,
            PublicFormat,
            NoEncryption,
        )
    except ImportError:  # pragma: no cover
        raise RuntimeError("install `cryptography` for Phase 0 signer support") from None

    import os as _os
    keys_dir = karpa_root / "miner" / "keys"
    keys_dir.mkdir(parents=True, exist_ok=True)
    try:
        keys_dir.chmod(0o700)
    except OSError:
        pass  # filesystem may not support chmod (e.g. some Docker bind mounts)
    sk_path = keys_dir / f"{miner_hotkey}.sk"
    pk_path = keys_dir / f"{miner_hotkey}.pk"
    if not sk_path.exists():
        sk = Ed25519PrivateKey.generate()
        sk_bytes = sk.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())
        pk_bytes = sk.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
        # Write under a restrictive umask so we never race with chmod.
        old_umask = _os.umask(0o077)
        try:
            sk_path.write_bytes(sk_bytes)
            pk_path.write_bytes(pk_bytes)
        finally:
            _os.umask(old_umask)
        try:
            sk_path.chmod(0o600)
            pk_path.chmod(0o644)
        except OSError:
            pass

    # Audit existing key file modes — reject loading if the private key is
    # readable by anyone but the owner. Defense-in-depth: prevents a previously
    # leaked key from being silently re-used.
    try:
        sk_mode = sk_path.stat().st_mode & 0o077
        if sk_mode != 0:
            raise RuntimeError(
                f"refusing to load private key {sk_path}: mode allows group/other "
                f"access (octal {oct(sk_path.stat().st_mode & 0o777)}). "
                f"chmod 600 {sk_path} or regenerate."
            )
    except FileNotFoundError:
        pass  # was just created; will be 0600

    return sk_path.read_bytes(), pk_path.read_bytes()


def _signed_payload(
    miner_hotkey: str,
    handshake_nonce: str,
    bundle_hash: str,
    hypothesis: str = "",
) -> bytes:
    """Canonical signed payload. Includes the hypothesis hash so the miner
    can't swap their pre-submission claim for a different one post-merge.

    Backward compatibility: if hypothesis is empty, the payload matches the
    pre-fix encoding so older signatures still verify.
    """
    base = f"{miner_hotkey}|{handshake_nonce}|{bundle_hash}"
    if hypothesis:
        import hashlib as _h
        hyp_hash = _h.sha256(hypothesis.encode("utf-8")).hexdigest()
        base += f"|{hyp_hash}"
    return base.encode("utf-8")


def sign_submission(
    karpa_root: Path,
    miner_hotkey: str,
    bundle_hash: str,
    handshake_nonce: str,
    hypothesis: str = "",
) -> dict:
    """Returns {signature_hex, public_key_hex}.

    Hypothesis is folded into the signed payload via its sha256 hash so the
    signature commits the miner to the exact text they submitted. The full
    hypothesis still rides on submission.json; the hash binding prevents
    post-merge edits.
    """
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    sk_bytes, pk_bytes = _ensure_keypair(karpa_root, miner_hotkey)
    sk = Ed25519PrivateKey.from_private_bytes(sk_bytes)
    payload = _signed_payload(miner_hotkey, handshake_nonce, bundle_hash, hypothesis)
    sig = sk.sign(payload)
    return {
        "signature_hex": sig.hex(),
        "public_key_hex": pk_bytes.hex(),
    }


def verify_signature(
    miner_hotkey: str,
    bundle_hash: str,
    handshake_nonce: str,
    signature_hex: str,
    public_key_hex: str,
    hypothesis: str = "",
) -> bool:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    try:
        pk = Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_key_hex))
        # Try the hypothesis-included variant first; fall back to the
        # pre-fix (empty-hypothesis) form for backward compatibility with
        # signatures generated before this change.
        for hyp_variant in (hypothesis, "") if hypothesis else ("",):
            payload = _signed_payload(miner_hotkey, handshake_nonce, bundle_hash, hyp_variant)
            try:
                pk.verify(bytes.fromhex(signature_hex), payload)
                return True
            except Exception:
                continue
        return False
    except Exception:
        return False


# ----------------------------- submission -----------------------------------


@dataclass
class SubmissionBundle:
    """The artifact a miner sends to the network."""
    miner_hotkey: str
    handshake_nonce: str
    patch_path: str
    proof_dir: str
    bundle_hash: str
    signature_hex: str
    public_key_hex: str
    submitted_at: float

    def to_dict(self) -> dict:
        return asdict(self)


def assemble_submission(
    karpa_root: Path,
    miner_hotkey: str,
    submission_dir: Path,
    proof_dir: Path,
) -> SubmissionBundle:
    """
    Build the final submission bundle. Expects that proof.runner has already
    been run against the submission_dir and proof_dir is its output.
    """
    manifest = json.loads((proof_dir / "bundle_manifest.json").read_text())
    bundle_hash = manifest["bundle_hash"]
    handshake_nonce = manifest["handshake_nonce"]
    sig = sign_submission(karpa_root, miner_hotkey, bundle_hash, handshake_nonce)

    bundle = SubmissionBundle(
        miner_hotkey=miner_hotkey,
        handshake_nonce=handshake_nonce,
        patch_path=str(submission_dir / "patch.diff"),
        proof_dir=str(proof_dir),
        bundle_hash=bundle_hash,
        signature_hex=sig["signature_hex"],
        public_key_hex=sig["public_key_hex"],
        submitted_at=time.time(),
    )
    # Write to the proof_dir for the validator to pick up.
    (proof_dir / "submission.json").write_text(
        json.dumps(bundle.to_dict(), indent=2, sort_keys=True)
    )
    return bundle


# ----------------------------- CLI ------------------------------------------


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["handshake", "assemble"])
    p.add_argument("--karpa-root", type=Path, default=Path(__file__).resolve().parent.parent)
    p.add_argument("--miner-hotkey", required=True)
    p.add_argument("--patch", type=Path, help="patch file (for handshake)")
    p.add_argument("--submission-dir", type=Path)
    p.add_argument("--proof-dir", type=Path)
    args = p.parse_args()

    if args.command == "handshake":
        patch_hash = (
            hashlib.sha256(args.patch.read_bytes()).hexdigest()
            if args.patch and args.patch.exists()
            else hashlib.sha256(b"").hexdigest()
        )
        nonce = request_handshake_nonce(args.karpa_root, args.miner_hotkey, patch_hash)
        print(nonce)
    elif args.command == "assemble":
        bundle = assemble_submission(
            args.karpa_root,
            args.miner_hotkey,
            args.submission_dir,
            args.proof_dir,
        )
        print(json.dumps(bundle.to_dict(), indent=2))


if __name__ == "__main__":
    main()

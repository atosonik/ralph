"""Encrypt a proof bundle to the validator key (libsodium sealed box).

Miners pack the bundle and encrypt it to a published validator public key
(X25519); only holders of the matching private key can decrypt it. The
submission transport (HF PR) is unchanged — the repo carries an opaque
`bundle.enc` plus a small plaintext `manifest.json` for indexing/lookup. The
decrypted blob reproduces the bundle directory byte-for-byte, so every
downstream check (signature, manifest hashes, bundle_hash recompute,
attestation verify) runs exactly as on a plaintext bundle.

Keys:
  - public key: `RALPH_VALIDATOR_PUBKEY` env, else `DEFAULT_VALIDATOR_PUBKEY`.
  - private key: `RALPH_VALIDATOR_PRIVKEY` env, else the `privkey` field of the
    JSON file at `RALPH_VALIDATOR_PRIVKEY_FILE`. Held by validators, distributed
    out of band, never committed.

Generate a keypair once with `python -m proof.bundle_crypto keygen`.
"""
from __future__ import annotations

import io
import json
import os
import tarfile
from pathlib import Path

# Published validator encryption public key (X25519, base64).
DEFAULT_VALIDATOR_PUBKEY = "+dCaAtEE/NCKUjOfktuKlKSs5WaER558CqPXJnz3eng="

ENC_SCHEME = "sealed_box_x25519_v1"
ENC_FILENAME = "bundle.enc"
PUBLIC_MANIFEST = "manifest.json"


def _nacl():
    try:
        from nacl.encoding import Base64Encoder
        from nacl.public import PrivateKey, PublicKey, SealedBox
        return PrivateKey, PublicKey, SealedBox, Base64Encoder
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "PyNaCl not installed — run `pip install 'ralph-subnet[attest]'`"
        ) from e


def load_validator_pubkey(explicit: str | None = None) -> str:
    pk = (explicit or os.environ.get("RALPH_VALIDATOR_PUBKEY") or DEFAULT_VALIDATOR_PUBKEY).strip()
    if not pk:
        raise RuntimeError("no validator public key (set RALPH_VALIDATOR_PUBKEY)")
    return pk


def load_validator_privkey(explicit: str | None = None) -> str:
    sk = (explicit or os.environ.get("RALPH_VALIDATOR_PRIVKEY") or "").strip()
    if not sk:
        path = os.environ.get("RALPH_VALIDATOR_PRIVKEY_FILE", "")
        if path and Path(path).exists():
            sk = (json.loads(Path(path).read_text()).get("privkey") or "").strip()
    if not sk:
        raise RuntimeError(
            "no validator private key (set RALPH_VALIDATOR_PRIVKEY or RALPH_VALIDATOR_PRIVKEY_FILE)"
        )
    return sk


def encrypt(plaintext: bytes, pubkey_b64: str | None = None) -> bytes:
    """Seal plaintext to the validator public key (anonymous sender)."""
    _, PublicKey, SealedBox, B64 = _nacl()
    box = SealedBox(PublicKey(load_validator_pubkey(pubkey_b64).encode(), encoder=B64))
    return box.encrypt(plaintext)


def decrypt(ciphertext: bytes, privkey_b64: str | None = None) -> bytes:
    """Open a sealed box with the validator private key. Raises on a wrong key
    or any tampering (the sealed box is authenticated)."""
    PrivateKey, _, SealedBox, B64 = _nacl()
    box = SealedBox(PrivateKey(load_validator_privkey(privkey_b64).encode(), encoder=B64))
    return box.decrypt(ciphertext)


def pack_bundle(files: list[tuple[Path, str]]) -> bytes:
    """tar.gz the (local_path, arcname) pairs into an in-memory blob. arcname
    is the final relative path the validator should see (e.g.
    ``training/checkpoint.pt``)."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for local, arcname in sorted(files, key=lambda t: t[1]):
            if Path(local).exists():
                tar.add(str(local), arcname=arcname)
    return buf.getvalue()


def _is_within(base: Path, target: Path) -> bool:
    try:
        target.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def unpack_bundle(blob: bytes, out_dir: Path) -> Path:
    """Extract a packed bundle into out_dir, rejecting path traversal / links."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        members = tar.getmembers()
        for m in members:
            name = m.name
            if name.startswith("/") or ".." in Path(name).parts:
                raise ValueError(f"unsafe path in bundle archive: {name!r}")
            if not (m.isfile() or m.isdir()):
                raise ValueError(f"unexpected member in bundle archive: {name!r} (links not allowed)")
            if not _is_within(out_dir, out_dir / name):
                raise ValueError(f"path escapes out_dir: {name!r}")
        try:
            tar.extractall(out_dir, filter="data")  # py>=3.12
        except TypeError:
            tar.extractall(out_dir)  # older py: members already validated above
    return out_dir


def gen_keypair() -> tuple[str, str]:
    """Return (pubkey_b64, privkey_b64). The owner runs this ONCE."""
    PrivateKey, _, _, B64 = _nacl()
    sk = PrivateKey.generate()
    return sk.public_key.encode(B64).decode(), sk.encode(B64).decode()


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Validator bundle-encryption key tools.")
    sub = p.add_subparsers(dest="cmd")
    kg = sub.add_parser("keygen", help="generate a validator keypair")
    kg.add_argument("--out", default="/root/.ralph_validator_enc_key.json")
    args = p.parse_args()

    if args.cmd == "keygen":
        pub, priv = gen_keypair()
        outp = Path(args.out)
        outp.write_text(json.dumps({"pubkey": pub, "privkey": priv, "scheme": ENC_SCHEME}, indent=2))
        try:
            outp.chmod(0o600)
        except OSError:
            pass
        # Print ONLY the public key; the private key stays in the (chmod-600) file.
        print(f"pubkey: {pub}")
        print(f"wrote keypair to {outp} — distribute privkey out of band; never commit it")
    else:
        p.print_help()


if __name__ == "__main__":
    main()

"""Tests for proof.bundle_crypto — the validator-key bundle encryption.

The core invariant the validator relies on: encrypt(pack(dir)) → decrypt →
unpack reproduces the bundle directory byte-for-byte, so the existing verify
path (hash recompute from disk) runs unchanged.
"""
from __future__ import annotations

import importlib.util
import io
import json
import tarfile

import pytest

from proof import bundle_crypto as BC

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("nacl") is None, reason="PyNaCl not installed"
)


def _set_keypair(monkeypatch):
    pub, priv = BC.gen_keypair()
    monkeypatch.setenv("RALPH_VALIDATOR_PUBKEY", pub)
    monkeypatch.setenv("RALPH_VALIDATOR_PRIVKEY", priv)
    return pub, priv


def test_encrypt_decrypt_roundtrip(monkeypatch):
    _set_keypair(monkeypatch)
    pt = b"the recipe diff + attestation tokens"
    ct = BC.encrypt(pt)
    assert ct != pt and len(ct) > len(pt)
    assert BC.decrypt(ct) == pt


def test_decrypt_with_wrong_key_fails():
    pub, _priv = BC.gen_keypair()
    ct = BC.encrypt(b"secret", pub)
    _other_pub, other_priv = BC.gen_keypair()
    with pytest.raises(Exception):
        BC.decrypt(ct, other_priv)


def test_decrypt_tampered_ciphertext_fails():
    pub, priv = BC.gen_keypair()
    ct = bytearray(BC.encrypt(b"secret payload", pub))
    ct[-1] ^= 0x01  # flip one bit
    with pytest.raises(Exception):
        BC.decrypt(bytes(ct), priv)


def test_pack_unpack_reproduces_bundle_dir(tmp_path, monkeypatch):
    _set_keypair(monkeypatch)
    src = tmp_path / "src"
    (src / "training").mkdir(parents=True)
    (src / "bundle_manifest.json").write_text('{"bundle_hash":"abc123"}')
    (src / "calibration.json").write_text("{}")
    (src / "attestation.json").write_text('{"gpu_token":"x","tdx_quote":"y"}')
    (src / "patch.diff").write_text("diff --git a/x b/x\n")
    (src / "training" / "checkpoint.pt").write_bytes(b"\x00\x01\x02weights\xff")
    (src / "training" / "training_log.jsonl").write_text('{"step":1}\n')
    files = [
        (src / "bundle_manifest.json", "bundle_manifest.json"),
        (src / "calibration.json", "calibration.json"),
        (src / "attestation.json", "attestation.json"),
        (src / "patch.diff", "patch.diff"),
        (src / "training" / "checkpoint.pt", "training/checkpoint.pt"),
        (src / "training" / "training_log.jsonl", "training/training_log.jsonl"),
    ]
    blob = BC.encrypt(BC.pack_bundle(files))
    out = tmp_path / "out"
    BC.unpack_bundle(BC.decrypt(blob), out)
    for local, arc in files:
        assert (out / arc).read_bytes() == local.read_bytes(), arc
    # training artifacts land under training/ where the validator looks
    assert (out / "training" / "checkpoint.pt").exists()


def test_unpack_rejects_path_traversal(tmp_path):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        data = b"pwned"
        info = tarfile.TarInfo(name="../escape.txt")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    with pytest.raises(ValueError):
        BC.unpack_bundle(buf.getvalue(), tmp_path / "out")
    assert not (tmp_path / "escape.txt").exists()


def test_unpack_rejects_symlink_member(tmp_path):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        info = tarfile.TarInfo(name="evil")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        tar.addfile(info)
    with pytest.raises(ValueError):
        BC.unpack_bundle(buf.getvalue(), tmp_path / "out")


def test_default_pubkey_is_embedded():
    assert BC.load_validator_pubkey()
    assert len(BC.DEFAULT_VALIDATOR_PUBKEY) == 44  # base64 of 32-byte X25519 key


def test_privkey_loads_from_file(tmp_path, monkeypatch):
    monkeypatch.delenv("RALPH_VALIDATOR_PRIVKEY", raising=False)
    pub, priv = BC.gen_keypair()
    f = tmp_path / "key.json"
    f.write_text(json.dumps({"pubkey": pub, "privkey": priv}))
    monkeypatch.setenv("RALPH_VALIDATOR_PRIVKEY_FILE", str(f))
    assert BC.load_validator_privkey() == priv


def test_missing_privkey_raises(monkeypatch):
    monkeypatch.delenv("RALPH_VALIDATOR_PRIVKEY", raising=False)
    monkeypatch.delenv("RALPH_VALIDATOR_PRIVKEY_FILE", raising=False)
    with pytest.raises(RuntimeError):
        BC.load_validator_privkey()

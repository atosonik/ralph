"""Ninja-style one-commitment-per-hotkey: content-addressed, readable, anti-spoof."""
from __future__ import annotations

import pytest

from chain_layer.submission_commitment import (
    COMMITMENT_RE,
    build_commitment,
    derive_submission_id,
    parse_commitment,
    signature_payload,
    verify_commitment,
)

HK = "5FCDTbBDka1WxcspxAjRMUeecp3vHnYiM1nenEsKGXysYqGE"
SHA = "d93b5c601bf339ebee19528b3e457dcbded75e3ef15f448c0a9b6bf9c524ce9d"
SHA2 = "a" * 64


def test_build_is_readable_and_content_addressed():
    c = build_commitment(SHA, hotkey=HK)
    assert c == f"ralph-submission:{HK[:16]}-{SHA[:16]}:{SHA}"
    assert COMMITMENT_RE.match(c)
    sid, sha = parse_commitment(c)
    assert sha == SHA and sid == derive_submission_id(HK, SHA)


def test_overwrite_is_update_not_race():
    # Same hotkey, new content -> a different, deterministic commitment. The
    # single on-chain slot just holds whichever was set last (the current one).
    c1 = build_commitment(SHA, hotkey=HK)
    c2 = build_commitment(SHA2, hotkey=HK)
    assert c1 != c2
    assert parse_commitment(c2)[1] == SHA2


def test_verify_matches_the_right_bundle():
    c = build_commitment(SHA, hotkey=HK)
    ok, _ = verify_commitment(c, hotkey=HK, bundle_sha256=SHA)
    assert ok


def test_verify_rejects_wrong_bundle():
    c = build_commitment(SHA, hotkey=HK)
    ok, reason = verify_commitment(c, hotkey=HK, bundle_sha256=SHA2)
    assert not ok and "content hash" in reason


def test_verify_rejects_spoofed_submission_id():
    # A commitment whose id was derived for a DIFFERENT hotkey but carries this
    # content must not verify for HK.
    other = "5H6mytgBYJbgTRNQv11gGfhFa1Dq91rd8TiSZQnKW7cmB1jb"
    spoof = f"ralph-submission:{derive_submission_id(other, SHA)}:{SHA}"
    assert COMMITMENT_RE.match(spoof)
    ok, reason = verify_commitment(spoof, hotkey=HK, bundle_sha256=SHA)
    assert not ok and "canonical" in reason


def test_parse_rejects_malformed():
    for bad in (
        "",
        "private-submission:x:" + SHA,           # wrong prefix
        f"ralph-submission:{HK[:16]}-{SHA[:16]}", # missing hash
        f"ralph-submission:bad id:{SHA}",         # space in id
        f"ralph-submission:x:{SHA.upper()}",      # uppercase hash
        f"ralph-submission:x:{SHA[:63]}",         # short hash
    ):
        with pytest.raises(ValueError):
            parse_commitment(bad)


def test_build_rejects_bad_sha():
    with pytest.raises(ValueError):
        build_commitment("nothex", hotkey=HK)


def test_case_insensitive_sha_normalizes():
    c = build_commitment(SHA.upper(), hotkey=HK)
    assert parse_commitment(c)[1] == SHA  # lowercased
    ok, _ = verify_commitment(c, hotkey=HK, bundle_sha256=SHA.upper())
    assert ok


def test_signature_payload_shape():
    sid = derive_submission_id(HK, SHA)
    assert signature_payload(HK, sid, SHA) == f"ralph-submission-v1:{HK}:{sid}:{SHA}".encode()

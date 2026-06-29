"""Stage 3 — HOSB wired into op4 (RALPH_HOSB off/shadow/enforce), entropy, and the
sandbox grid entrypoint. CPU only: the op4 tests inject an in-process NLL provider
(monkeypatching the container call) and use a LocalChain for the block-hash seed.
"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ralph_bootstrap  # noqa: F401
import validator.validator as vv
from chain_layer.local import LocalChain
from eval.val_bpb import build_blanked_grid, per_position_nlls_blanked, pinned_eval_seq_len

RECIPE_DIR = str(ralph_bootstrap.RECIPE_DIR)
if RECIPE_DIR not in sys.path:
    sys.path.insert(0, RECIPE_DIR)
try:
    from model import RalphBase, RalphConfig  # noqa: E402
    _HAVE_MODEL = True
except Exception:  # noqa: BLE001
    _HAVE_MODEL = False

pytestmark = pytest.mark.skipif(not _HAVE_MODEL, reason="canonical model package not importable")


def _tiny(tmp_path):
    torch.manual_seed(0)
    cfg = RalphConfig(vocab_size=64, dim=32, n_layers=2, n_heads=2, head_dim=16, ffn_mult=2.0, max_seq_len=16)
    model = RalphBase(cfg)
    ckpt = tmp_path / "checkpoint.pt"
    torch.save({"model": model.state_dict(), "config": dataclasses.asdict(cfg)}, ckpt)
    return cfg, model, ckpt


def _load_canonical(ckpt_path):
    saved = torch.load(ckpt_path, weights_only=True, map_location="cpu")
    cfg = RalphConfig(**{k: v for k, v in saved["config"].items() if k in RalphConfig.__dataclass_fields__})
    m = RalphBase(cfg)
    m.load_state_dict(saved["model"])
    return m


def _setup_submission(tmp_path):
    """A ralph_root with an eval shard + a proof_dir with the checkpoint."""
    cfg, model, _ = _tiny(tmp_path)
    ralph_root = tmp_path / "root"
    evdir = ralph_root / "eval" / "private"
    evdir.mkdir(parents=True)
    tokens = np.random.default_rng(11).integers(0, cfg.vocab_size, size=600, dtype=np.uint16)
    tokens.tofile(evdir / "active_tokens.bin")
    proof = tmp_path / "proof"
    (proof / "training").mkdir(parents=True)
    torch.save({"model": model.state_dict(), "config": dataclasses.asdict(cfg)}, proof / "training" / "checkpoint.pt")
    (proof / "submission.json").write_text(json.dumps({"bundle_hash": "bh"}))
    return ralph_root, proof, cfg, model


def _inprocess_provider(ralph_root, proof, *, bench=0.5):
    """A stand-in for _hosb_sandbox_nlls: runs the model in-process over the grid."""
    model = _load_canonical(proof / "training" / "checkpoint.pt")

    def provider(_rr, _pd, idx_grid, tgt_grid, _seed=b""):
        nlls = per_position_nlls_blanked(model, idx_grid, tgt_grid)
        return nlls, {"benchmark_accuracy": bench, "benchmark_examples": 3}

    return provider


# ---------------------------------------------------------------------------
# entropy + mode parsing
# ---------------------------------------------------------------------------


def test_derive_grid_seed_deterministic_and_entropic():
    a = vv.derive_grid_seed("blockhashA", "shardfp", "bundlefp", "hk")
    b = vv.derive_grid_seed("blockhashA", "shardfp", "bundlefp", "hk")
    c = vv.derive_grid_seed("blockhashB", "shardfp", "bundlefp", "hk")  # different block
    assert a == b and len(a) == 32
    assert a != c  # a fresh block hash → a fresh, unpredictable schedule


def test_hosb_mode_parsing(monkeypatch):
    for raw, want in [("off", "off"), ("shadow", "shadow"), ("ENFORCE", "enforce"), ("garbage", "off"), (None, "off")]:
        if raw is None:
            monkeypatch.delenv("RALPH_HOSB", raising=False)
        else:
            monkeypatch.setenv("RALPH_HOSB", raw)
        assert vv._hosb_mode() == want


# ---------------------------------------------------------------------------
# sandbox grid entrypoint (in-process, no container)
# ---------------------------------------------------------------------------


def test_sandbox_grid_eval_is_leak_free_and_matches_direct(tmp_path):
    """The container emits TOP-K logits (never the targets); the HOST computes CE,
    and it equals the direct in-process per-cell CE."""
    from eval.host_reduce import ce_from_topk_logits
    from validator.sandbox_eval import run_sandbox_grid_eval

    cfg, model, ckpt = _tiny(tmp_path)
    tokens = np.random.default_rng(5).integers(0, cfg.vocab_size, size=400, dtype=np.uint16)
    L = pinned_eval_seq_len(cfg.max_seq_len)
    idx, tgt, _layout = build_blanked_grid(tokens, tokens, L, b"seed-x", n_scored_per_window=6)
    rows = np.arange(idx.shape[0])
    scored_idx = (tgt != -100).argmax(axis=1)

    gdir = tmp_path / "grid"
    gdir.mkdir()
    np.save(gdir / "idx_grid.npy", idx)
    np.save(gdir / "scored_idx.npy", scored_idx)
    (gdir / "job.json").write_text(json.dumps({"top_k": cfg.vocab_size}))  # K=V → exact CE
    out = tmp_path / "out"
    run_sandbox_grid_eval(RECIPE_DIR, ckpt, gdir, out)

    # LEAK-FREE: the container was never given the targets.
    assert not (gdir / "tgt_grid.npy").exists()
    man = json.loads((out / "manifest.json").read_text())
    assert man["mode"] == "hosb_grid_topk" and man["rows"] == idx.shape[0]

    assert not (out / "logsumexp.npy").exists()  # no container partition function
    targets = tgt[rows, scored_idx]
    ce = ce_from_topk_logits(
        np.load(out / "topk_logits.npy"), np.load(out / "topk_indices.npy"), targets, cfg.vocab_size,
    )
    direct = per_position_nlls_blanked(model, idx, tgt)[rows, scored_idx]
    assert np.allclose(ce, direct, atol=1e-4)


def test_ce_from_topk_exact_when_target_in_topk():
    # full top-K (K=V): Z_hat = host logsumexp over all logits → exact CE.
    import torch

    from eval.host_reduce import ce_from_topk_logits
    logits = np.array([[2.0, 1.0, 0.0, -1.0, -2.0]])
    lse = torch.logsumexp(torch.tensor(logits), dim=-1).numpy()
    order = np.argsort(-logits, axis=1)
    tv = np.take_along_axis(logits, order, axis=1)
    ce = ce_from_topk_logits(tv, order, targets=np.array([2]), vocab_size=5)
    assert ce[0] == pytest.approx(lse[0] - logits[0, 2], rel=1e-6)


def test_ce_uniform_logit_shift_invariant():
    """CE is invariant to a uniform shift of all emitted logits (Z_hat and the
    target logit shift together) — kills the rescale forgery."""
    from eval.host_reduce import ce_from_topk_logits

    tv = np.array([[3.0, 1.0, 0.0]])
    ti = np.array([[7, 2, 5]])
    base = ce_from_topk_logits(tv, ti, np.array([7]), vocab_size=50)
    shifted = ce_from_topk_logits(tv + 12.5, ti, np.array([7]), vocab_size=50)
    assert np.allclose(base, shifted, atol=1e-9)


def test_ce_from_topk_host_owns_partition_no_logsumexp_param():
    """The container can no longer supply a logsumexp to deflate CE — the host
    computes Z_hat = logsumexp(emitted top-K) itself. A degenerate emission (one
    real token + filler) self-defeats: a missed target carries a HUGE boundary CE."""
    # honest near-full top-K → exact-ish CE on a hit
    import torch

    from eval.host_reduce import ce_from_topk_logits
    logits = np.array([[5.0, 4.0, 3.0, 2.0]])
    lse = torch.logsumexp(torch.tensor(logits), dim=-1).numpy()[0]
    ce_hit = ce_from_topk_logits(logits, np.array([[0, 1, 2, 3]]), np.array([1]), vocab_size=1000)
    assert ce_hit[0] == pytest.approx(lse - 4.0, rel=1e-6)
    # degenerate emission: rank-1 high + filler-low; a missed target → huge miss CE
    ce_miss = ce_from_topk_logits(np.array([[20.0, -30.0]]), np.array([[0, 1]]), np.array([999]), vocab_size=1000)
    assert ce_miss[0] > 40.0  # z_hat(~20) - min(-30) ~ 50 nats → self-defeating


def test_ce_from_topk_rejects_impossible_emissions():
    from eval.host_reduce import ce_from_topk_logits

    tgt = np.array([0])
    with pytest.raises(ValueError, match="outside"):  # index out of [0, V)
        ce_from_topk_logits(np.array([[1.0, 0.0]]), np.array([[0, 999]]), tgt, vocab_size=5)
    with pytest.raises(ValueError, match="duplicate"):  # repeated index inflates coverage
        ce_from_topk_logits(np.array([[1.0, 0.0]]), np.array([[2, 2]]), tgt, vocab_size=5)
    with pytest.raises(ValueError, match="non-finite"):
        ce_from_topk_logits(np.array([[np.inf, 0.0]]), np.array([[0, 1]]), tgt, vocab_size=5)


def test_hosb_benchmark_is_host_reduced_no_answer_key_mounted(tmp_path, monkeypatch):
    """The benchmark answer key is never mounted; the container gets only shuffled
    candidates and the HOST computes accuracy from the private correct slot."""
    import validator.sandbox as sbx
    from validator.sandbox import SandboxResult
    from validator.sandbox_eval import run_sandbox_grid_eval

    ralph_root, proof, cfg, _model = _setup_submission(tmp_path)
    monkeypatch.setenv("RALPH_SANDBOX_IMAGE", "ralph-eval-sandbox@sha256:" + "a" * 64)
    # A small whitened benchmark file; contexts kept short (< the tiny model's
    # max_seq_len=16; real models are 512+). Candidates from one exchangeable pool.
    rng = np.random.default_rng(1)
    examples = []
    for _ in range(8):
        cands = rng.choice(cfg.vocab_size, size=5, replace=False)
        examples.append({
            "context_ids": [int(t) for t in rng.integers(0, cfg.vocab_size, size=6)],
            "target_id": int(cands[0]),
            "distractors": [int(c) for c in cands[1:]],
        })
    (ralph_root / "eval" / "private" / "active_benchmark.json").write_text(json.dumps(examples))
    tokens = np.fromfile(ralph_root / "eval" / "private" / "active_tokens.bin", dtype=np.uint16)
    L = pinned_eval_seq_len(cfg.max_seq_len)
    idx, tgt, _layout = build_blanked_grid(tokens, tokens, L, b"bs", n_scored_per_window=4)

    captured = {}

    def fake(cfg_, *, container_argv, mounts, out_dir, timeout_s, **kw):
        gdir = next(Path(m.host) for m in mounts if m.container == "/grid")
        captured["files"] = sorted(p.name for p in gdir.iterdir())
        run_sandbox_grid_eval(RECIPE_DIR, proof / "training" / "checkpoint.pt", gdir, out_dir)
        return SandboxResult(returncode=0, stdout="ok", stderr="", timed_out=False)

    monkeypatch.setattr(sbx, "run_in_sandbox", fake)
    _nlls, manifest = vv._hosb_sandbox_nlls(ralph_root, proof, idx, tgt, b"bench-seed")

    # shuffled candidates are mounted; the answer key / correct-index never is.
    assert "bench_cands.npy" in captured["files"]
    assert "active_benchmark.json" not in captured["files"]
    assert not any("correct" in f for f in captured["files"])
    # the host computed a real accuracy (not a container-reported number).
    assert manifest["benchmark_examples"] == 8
    assert 0.0 <= manifest["benchmark_accuracy"] <= 1.0


def test_hosb_sandbox_pins_vocab_from_checkpoint_not_manifest(tmp_path, monkeypatch):
    """The container manifest's vocab is never used in scoring — V comes from the
    HOST checkpoint config (index-range validation only; CE uses the host-built
    Z_hat). A forged manifest vocab changes nothing."""
    import validator.sandbox as sbx
    from validator.sandbox import SandboxResult
    from validator.sandbox_eval import run_sandbox_grid_eval

    ralph_root, proof, cfg, _model = _setup_submission(tmp_path)
    monkeypatch.setenv("RALPH_SANDBOX_IMAGE", "ralph-eval-sandbox@sha256:" + "a" * 64)
    monkeypatch.setenv("RALPH_HOSB_TOPK", "8")  # K < vocab(64) → miss cells exist
    tokens = np.fromfile(ralph_root / "eval" / "private" / "active_tokens.bin", dtype=np.uint16)
    L = pinned_eval_seq_len(cfg.max_seq_len)
    idx, tgt, _layout = build_blanked_grid(tokens, tokens, L, b"s", n_scored_per_window=4)

    forged = {"v": 2}

    def fake(cfg_, *, container_argv, mounts, out_dir, timeout_s, **kw):
        gdir = next(Path(m.host) for m in mounts if m.container == "/grid")
        run_sandbox_grid_eval(RECIPE_DIR, proof / "training" / "checkpoint.pt", gdir, out_dir)
        mp = Path(out_dir) / "manifest.json"
        man = json.loads(mp.read_text())
        man["model_config"]["vocab_size"] = forged["v"]  # forge the reported vocab
        mp.write_text(json.dumps(man))
        return SandboxResult(returncode=0, stdout="ok", stderr="", timed_out=False)

    monkeypatch.setattr(sbx, "run_in_sandbox", fake)
    forged["v"] = 2
    a, _ = vv._hosb_sandbox_nlls(ralph_root, proof, idx, tgt)
    forged["v"] = 999999
    b, _ = vv._hosb_sandbox_nlls(ralph_root, proof, idx, tgt)
    # Identical despite wildly different forged manifest vocab → the manifest value
    # is IGNORED (host pins V from the checkpoint). If trusted, a != b.
    assert np.allclose(a, b)




# ---------------------------------------------------------------------------
# op4 dispatch: enforce / shadow / fail-closed
# ---------------------------------------------------------------------------


def test_hosb_sandbox_provider_never_mounts_targets(tmp_path, monkeypatch):
    """End-to-end leak check: _hosb_sandbox_nlls writes idx_grid + scored_idx to the
    container mount but NEVER tgt_grid; the host computes CE from the emitted top-K."""
    import validator.sandbox as sbx
    from validator.sandbox import SandboxResult
    from validator.sandbox_eval import run_sandbox_grid_eval

    ralph_root, proof, cfg, model = _setup_submission(tmp_path)
    monkeypatch.setenv("RALPH_SANDBOX_IMAGE", "ralph-eval-sandbox@sha256:" + "a" * 64)
    tokens = np.fromfile(ralph_root / "eval" / "private" / "active_tokens.bin", dtype=np.uint16)
    L = pinned_eval_seq_len(cfg.max_seq_len)
    idx, tgt, _layout = build_blanked_grid(tokens, tokens, L, b"seed", n_scored_per_window=4)

    captured = {}

    def fake_sandbox(cfg_, *, container_argv, mounts, out_dir, timeout_s, **kw):
        gdir = next(Path(m.host) for m in mounts if m.container == "/grid")
        captured["files"] = sorted(p.name for p in gdir.iterdir())
        run_sandbox_grid_eval(RECIPE_DIR, proof / "training" / "checkpoint.pt", gdir, out_dir)
        return SandboxResult(returncode=0, stdout="ok", stderr="", timed_out=False)

    monkeypatch.setattr(sbx, "run_in_sandbox", fake_sandbox)

    nlls2d, _manifest = vv._hosb_sandbox_nlls(ralph_root, proof, idx, tgt)

    assert "idx_grid.npy" in captured["files"] and "scored_idx.npy" in captured["files"]
    assert "tgt_grid.npy" not in captured["files"]  # the answer key is NEVER mounted
    rows = np.arange(idx.shape[0])
    sc = (tgt != -100).argmax(axis=1)
    direct = per_position_nlls_blanked(model, idx, tgt)[rows, sc]
    assert np.allclose(nlls2d[rows, sc], direct, atol=1e-4)


def test_op4_enforce_requires_ack_else_fails_closed(tmp_path, monkeypatch):
    ralph_root, proof, _cfg, _model = _setup_submission(tmp_path)
    monkeypatch.setenv("RALPH_HOSB", "enforce")
    monkeypatch.delenv("RALPH_HOSB_ENFORCE_ACK", raising=False)
    monkeypatch.setattr(vv, "_hosb_sandbox_nlls", lambda *a, **k: pytest.fail("HOSB ran without ack"))
    ok, detail, result = vv.op4_hidden_eval(ralph_root, proof, chain=LocalChain(tmp_path / "chain"))
    assert not ok and result is None and "calibration" in detail


def test_op4_enforce_crowns_hosb(tmp_path, monkeypatch):
    ralph_root, proof, cfg, model = _setup_submission(tmp_path)
    monkeypatch.setenv("RALPH_HOSB", "enforce")
    monkeypatch.setenv("RALPH_HOSB_ENFORCE_ACK", "1")
    monkeypatch.setattr(vv, "_hosb_sandbox_nlls", _inprocess_provider(ralph_root, proof, bench=0.5))
    chain = LocalChain(tmp_path / "chain")

    ok, detail, result = vv.op4_hidden_eval(ralph_root, proof, chain=chain)
    assert ok, detail
    assert "HOSB" in detail and result is not None
    assert result.val_seq_len == pinned_eval_seq_len(cfg.max_seq_len)
    assert result.benchmark_accuracy == 0.5
    # Sanity: the crowned HOSB val_bpb is a finite honest number, not collapsed.
    assert np.isfinite(result.val_bpb) and result.val_bpb > 0


def test_op4_enforce_fails_closed_without_chain_entropy(tmp_path, monkeypatch):
    ralph_root, proof, _cfg, _model = _setup_submission(tmp_path)
    monkeypatch.setenv("RALPH_HOSB", "enforce")
    monkeypatch.setenv("RALPH_HOSB_ENFORCE_ACK", "1")
    monkeypatch.setattr(vv, "_hosb_sandbox_nlls", _inprocess_provider(ralph_root, proof))

    ok, detail, result = vv.op4_hidden_eval(ralph_root, proof, chain=None)
    assert not ok and result is None
    assert "entropy" in detail.lower()


def test_op4_enforce_rejects_witness_failure(tmp_path, monkeypatch):
    ralph_root, proof, _cfg, _model = _setup_submission(tmp_path)
    monkeypatch.setenv("RALPH_HOSB", "enforce")
    monkeypatch.setenv("RALPH_HOSB_ENFORCE_ACK", "1")

    # A degenerate cheat producer: emit ~0 NLL everywhere → the wrong-target cells
    # are all sub-floor → the host's witness rejects (fail-closed).
    def zeros_provider(_rr, _pd, idx_grid, _tgt_grid, _seed=b""):
        return np.zeros_like(idx_grid, dtype=np.float32), {"benchmark_accuracy": 0.0, "benchmark_examples": 0}

    monkeypatch.setattr(vv, "_hosb_sandbox_nlls", zeros_provider)
    chain = LocalChain(tmp_path / "chain")

    ok, detail, result = vv.op4_hidden_eval(ralph_root, proof, chain=chain)
    assert not ok and result is None
    assert "REJECTED" in detail or "wrong" in detail.lower()


def test_op4_shadow_crowns_legacy_and_logs(tmp_path, monkeypatch):
    from eval import HiddenEvalResult

    ralph_root, proof, cfg, _model = _setup_submission(tmp_path)
    monkeypatch.setenv("RALPH_HOSB", "shadow")
    monkeypatch.setattr(vv, "_hosb_sandbox_nlls", _inprocess_provider(ralph_root, proof))

    legacy = HiddenEvalResult(val_bpb=2.0, benchmark_accuracy=0.4, tokens_evaluated=100,
                              benchmark_examples=3, eval_set_hash="x", val_seq_len=8, tail_val_bpb=2.1)
    monkeypatch.setattr(vv, "_legacy_hidden_eval", lambda rr, pd: (True, "legacy", legacy))
    chain = LocalChain(tmp_path / "chain")

    ok, detail, result = vv.op4_hidden_eval(ralph_root, proof, chain=chain)
    assert ok and result is legacy  # the CROWN is the legacy score in shadow mode
    # …and HOSB was logged for calibration.
    log = (ralph_root / "hosb_shadow.jsonl").read_text().strip().splitlines()
    assert len(log) == 1
    rec = json.loads(log[0])
    assert rec["legacy_val_bpb"] == 2.0 and rec["hosb_ok"] is True
    assert "hosb_val_bpb" in rec and "delta_vs_legacy" in rec


def test_op4_off_is_unchanged_legacy(tmp_path, monkeypatch):
    from eval import HiddenEvalResult

    ralph_root, proof, _cfg, _model = _setup_submission(tmp_path)
    monkeypatch.delenv("RALPH_HOSB", raising=False)
    legacy = HiddenEvalResult(val_bpb=1.5, benchmark_accuracy=0.5, tokens_evaluated=10,
                              benchmark_examples=1, eval_set_hash="x", val_seq_len=8, tail_val_bpb=None)
    monkeypatch.setattr(vv, "_legacy_hidden_eval", lambda rr, pd: (True, "legacy-detail", legacy))
    # No HOSB work should happen in off mode.
    monkeypatch.setattr(vv, "_hosb_sandbox_nlls", lambda *a, **k: pytest.fail("HOSB ran in off mode"))

    ok, detail, result = vv.op4_hidden_eval(ralph_root, proof, chain=LocalChain(tmp_path / "chain"))
    assert ok and result is legacy and detail == "legacy-detail"
    assert not (ralph_root / "hosb_shadow.jsonl").exists()

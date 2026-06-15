# H100 miner setup

How to launch a Ralph miner on a rented H100 (Shadeform / Hyperstack / any cloud) and submit proof bundles to validators via HuggingFace Hub.

## What you need on the box

- 1× H100 80GB
- Ubuntu 22.04+, CUDA 12.x driver
- Python 3.10+, git, build-essentials
- ~150 GB free disk (training data + checkpoint)

## 1. Rent the box

From this repo's machine (the one already holding wallets + Shadeform API key):

```bash
python scripts/gpu.py rent           # default: Hyperstack H100
python scripts/gpu.py status         # wait for "active"
python scripts/gpu.py ssh            # SSH in
```

## 2. Bootstrap on the H100

Inside the rented box, clone **both** repos side-by-side:

```bash
git clone https://github.com/RalphLabsAI/ralph.git
git clone https://github.com/RalphLabsAI/recipe.git

cd ralph
python -m venv .venv
source .venv/bin/activate
pip install -e '.[hub]'    # includes huggingface_hub + bittensor

# Prepare training data inside the recipe repo
(cd ../recipe && python -m data.prepare \
  --source fineweb-edu \
  --out data/shards \
  --total-tokens 1_000_000_000 \
  --eval-tokens 5_000_000)
```

The protocol auto-resolves the recipe via `../recipe` — no flag needed. To put it elsewhere, set `RALPH_RECIPE_DIR=/path/to/recipe` in `.env`.

## 3. Bittensor wallet + HF token

Two creds the miner needs:

| Secret              | Purpose                                |
| ------------------- | -------------------------------------- |
| `BT_WALLET_PASSWORD` | Decrypt the coldkey at startup        |
| `HF_TOKEN`          | Write proof bundles to the dataset repo|

Copy your already-registered miner hotkey to the box (the simplest path is `rsync ~/.bittensor/wallets/<name>` from your laptop), then create `.env`:

```bash
cat > .env <<'EOF'
RALPH_CHAIN=bittensor
BT_NETWORK=test
BT_NETUID=16
BT_WALLET=<your-miner-wallet>
BT_HOTKEY=default
BT_WALLET_PASSWORD=<password>

HF_TOKEN=<hf-write-token>
RALPH_HF_REPO=RalphLabsAI/proof-bundles
EOF
```

Verify registration:

```bash
set -a && source .env && set +a
python -c "
from chain_layer.config import get_chain
from pathlib import Path
c = get_chain(Path('.'))
hk = c.wallet.hotkey.ss58_address
print('hotkey', hk, 'registered:', c.is_hotkey_registered(hk))
"
```

## 4. Submit a bundle

Baseline (empty patch — establishes you as a participant):

```bash
.venv/bin/python scripts/miner_run.py \
  --baseline \
  --label round1_baseline \
  --config configs/proxy_h100.json \
  --tier unverified
```

A patch that tweaks the recipe — generate the diff from your edits in `../recipe`:

```bash
# Edit configs/h100_proxy.json in the recipe repo, then:
(cd ../recipe && git diff main) > patches/raise_lr.diff

.venv/bin/python scripts/miner_run.py \
  --patch patches/raise_lr.diff \
  --label round2_raise_lr \
  --config configs/h100_proxy.json \
  --tier unverified
```

What happens under the hood (printed in the run log):
1. Hash patch → `patch_hash`
2. Commit `(hotkey, patch_hash, nonce)` on Bittensor
3. Run canonical training (proxy_h100.json: ~30 min on a single H100)
4. Sign the bundle with the miner hotkey
5. Upload bundle to `RalphLabsAI/proof-bundles/submissions/<bundle_hash[:16]>/`

After upload, validators worldwide will discover the bundle on their next HF poll (~60s) and score it. Track progress from any validator host with:

```bash
tail -f chain_bt_test_16/events.jsonl
```

## 5. Multi-round iteration

The intent is repeated submissions — every new patch is a new round. Loop:

```bash
while true; do
  # ... write a new patch / config tweak ...
  .venv/bin/python scripts/miner_run.py --patch patches/$NAME.diff --label $NAME
  sleep $((60 * 30))  # leave room for validator scoring + weight set
done
```

## 6. When done, back up & shut down

From the orchestrator (not the H100):

```bash
python scripts/gpu.py backup       # rsync runs/ + chain* back home
python scripts/gpu.py delete       # terminate the instance
```

## Notes

- **Tier**: `unverified` for now (no Confidential Compute). Verified tier needs Intel TDX + NVIDIA CC enrollment — track [Targon's pattern](../memory_or_reference/reference_targon_attestation.md).
- **Cost**: ~$2/hr on Hyperstack H100. One proxy_h100.json proof test = ~30 min training + 1 min upload. Budget per submission ≈ $1.
- **Validators score for free** — they only run the cheap hidden-eval, not retraining. Your H100 is the expensive node, by design.

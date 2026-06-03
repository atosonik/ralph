#!/usr/bin/env bash
# Per-session sanity check + status print.
# Run at the top of a session — verifies that the validator host is in a
# state where a research round can be kicked off. Does NOT spend money;
# only reads state.
#
# Exit 0 if ready; non-zero if anything blocks (env, wallets, validator).
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

red()   { printf "\033[31m%s\033[0m\n" "$1"; }
green() { printf "\033[32m%s\033[0m\n" "$1"; }
yellow(){ printf "\033[33m%s\033[0m\n" "$1"; }

ok=1
fail() { red "  FAIL: $1"; ok=0; }
pass() { green "  ok:   $1"; }
warn() { yellow "  warn: $1"; }

echo "=== Karpa session setup check ==="

# 1. .env present + restrictive
echo
echo "1. environment"
if [ -f .env ]; then
  mode=$(stat -c '%a' .env 2>/dev/null || stat -f '%Lp' .env 2>/dev/null)
  if [ "$mode" = "600" ]; then pass ".env mode 600"; else fail ".env mode $mode (need 600 — chmod 600 .env)"; fi
  # source-and-probe presence; do not echo any values
  set -a; . ./.env 2>/dev/null; set +a
  for k in BT_NETWORK BT_NETUID BT_WALLET BT_HOTKEY BT_WALLET_PASSWORD HF_TOKEN KARPA_BOT_GH_TOKEN; do
    if [ -n "${!k:-}" ]; then pass "$k: set"; else fail "$k: missing"; fi
  done
else
  fail ".env not found at $ROOT/.env"
fi

# 2. wallets present
echo
echo "2. wallets"
for w in green-test green1 green2; do
  if [ -d "$HOME/.bittensor/wallets/$w" ]; then pass "wallet $w present"; else fail "wallet $w missing"; fi
done
for w in green3; do
  if [ -d "$HOME/.bittensor/wallets/$w" ]; then pass "wallet $w (spare) present"; else warn "wallet $w (spare) missing"; fi
done

# 3. shadeform credentials
echo
echo "3. shadeform"
if [ -f /root/.shadeform_api_key ]; then
  mode=$(stat -c '%a' /root/.shadeform_api_key 2>/dev/null || stat -f '%Lp' /root/.shadeform_api_key 2>/dev/null)
  if [ "$mode" = "600" ]; then pass "shadeform api key mode 600"; else fail "shadeform api key mode $mode (need 600)"; fi
else
  fail "shadeform api key missing at /root/.shadeform_api_key"
fi
if [ -f /root/.ssh/id_bitzic ]; then pass "ssh key id_bitzic present"; else fail "ssh key id_bitzic missing"; fi

# 4. recipe checkout
echo
echo "4. recipe"
if [ -d /workspace/unicorn/karpathian/recipe ]; then
  pass "recipe checkout present"
else
  fail "recipe checkout missing at /workspace/unicorn/karpathian/recipe"
fi

# 5. validator running?
echo
echo "5. validator"
val_pids=$(pgrep -f "validator.service" 2>/dev/null | head -3)
if [ -n "$val_pids" ]; then
  pass "validator running (PIDs: $val_pids)"
else
  warn "validator not running — start it before kicking off a round:"
  echo "         cd $ROOT && nohup .venv/bin/python -m validator.service --epoch-seconds 60 --hf-repo karpaai/proof-bundles > logs/validator.log 2>&1 &"
fi

# 6. orphan H100 sweep
echo
echo "6. shadeform orphan sweep"
if [ -x scripts/orphan_gpu_sweep.py ]; then
  .venv/bin/python scripts/orphan_gpu_sweep.py --quiet 2>&1 | sed 's/^/         /'
  pass "orphan sweep ran"
else
  fail "scripts/orphan_gpu_sweep.py not executable"
fi

# 7. current king + recent activity
echo
echo "7. chain state"
if [ -f chain/king.json ]; then
  king_bpb=$(.venv/bin/python -c "import json;print(json.load(open('chain/king.json'))['val_bpb'])" 2>/dev/null)
  king_hk=$(.venv/bin/python -c "import json;print(json.load(open('chain/king.json'))['miner_hotkey'][:16])" 2>/dev/null)
  pass "king: ${king_hk}… val_bpb=${king_bpb}"
else
  warn "no king yet (chain/king.json absent)"
fi
if [ -f chain/events.jsonl ]; then
  recent=$(tail -1 chain/events.jsonl | .venv/bin/python -c "import json,sys;e=json.loads(sys.stdin.read());print(e.get('type','?'))" 2>/dev/null)
  pass "last event: $recent"
fi

# 8. agents dir present
echo
echo "8. agents"
for a in a b; do
  if [ -d agents/$a ]; then
    n=$(ls agents/$a/memory.jsonl 2>/dev/null | wc -l)
    rounds=$(wc -l < agents/$a/memory.jsonl 2>/dev/null || echo 0)
    pass "agent_$a present ($rounds prior rounds)"
  else
    warn "agent_$a directory missing — will be created on first round"
  fi
done

echo
if [ $ok -eq 1 ]; then
  green "==> ready"
  exit 0
else
  red "==> not ready (fix issues above)"
  exit 1
fi

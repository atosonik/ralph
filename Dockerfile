# Ralph proof-test container
#
# This is the official miner container that runs the canonical training
# procedure for a submitted patch and produces the attested proof bundle.
#
# In Phase 0.5+ its image digest = the "container measurement" committed
# on-chain. No other workload can produce a valid attestation against this
# measurement. The validator rejects any submission whose attested measurement
# doesn't match.
#
# Build:
#   docker build -t ralph-proof:latest .
#
# Run a proof test:
#   docker run --gpus all --rm \
#     -v /path/to/data:/data:ro \
#     -v /path/to/submission:/submission:ro \
#     -v /path/to/output:/output \
#     ralph-proof:latest \
#       --submission /submission --out-dir /output
#
# For reproducible builds:
#   DOCKER_BUILDKIT=1 docker build \
#     --build-arg BUILDKIT_INLINE_CACHE=1 \
#     -t ralph-proof:$(git rev-parse --short HEAD) .
#
# The image digest (sha256) is the container measurement:
#   docker inspect --format='{{.RepoDigests}}' ralph-proof:latest

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    patch \
    && rm -rf /var/lib/apt/lists/*

RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies first for layer caching.
COPY pyproject.toml /app/pyproject.toml
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124 && \
    pip install --no-cache-dir -e /app

# Pin the source commits the measured tree is built from. The image has no
# .git, so these env vars are how proof/runner.py records ralph_source_commit /
# recipe_source_commit in the bundle manifest — letting the validator report a
# clear "built against the wrong version" error instead of an opaque measurement
# mismatch (see RALPH_CANONICAL_SOURCE_COMMITS on the validator).
#   docker build \
#     --build-arg RALPH_SOURCE_COMMIT=$(git -C ralph rev-parse HEAD) \
#     --build-arg RECIPE_SOURCE_COMMIT=$(git -C recipe rev-parse HEAD) .
# For a REPRODUCIBLE measurement: assemble the build context from those exact
# (clean) commits, and do NOT include eval/private/ — the validator's secret
# held-out eval, excluded from the measurement and never shipped to miners.
ARG RALPH_SOURCE_COMMIT=unknown
ARG RECIPE_SOURCE_COMMIT=unknown
ENV RALPH_SOURCE_COMMIT=${RALPH_SOURCE_COMMIT}
ENV RECIPE_SOURCE_COMMIT=${RECIPE_SOURCE_COMMIT}

# Flattened image layout: model/recipe/data/configs live directly under /app, so
# point ralph_bootstrap's recipe resolver at /app. Without this it falls back to
# /app/recipe and computes the measurement over an empty recipe tree (wrong hash).
ENV RALPH_RECIPE_DIR=/app

# Copy the protocol source — model, recipe, data, eval, calibration, proof.
# This is the "canonical training code" whose hash is the measurement.
COPY model/ /app/model/
COPY recipe/ /app/recipe/
COPY data/ /app/data/
COPY eval/ /app/eval/
COPY calibration/ /app/calibration/
COPY proof/ /app/proof/
COPY miner/ /app/miner/
COPY validator/ /app/validator/
COPY configs/ /app/configs/
COPY restricted_files.yaml /app/restricted_files.yaml
# README.md contributes to the container_measurement (proof/sources.py _PROTOCOL_FILES).
COPY README.md /app/README.md
# ralph_bootstrap.py is the top-level recipe-path resolver. It is not a package in
# pyproject [tool.setuptools.packages.find], so `pip install -e` does not vendor it —
# copy it explicitly or the runner crashes at `from ralph_bootstrap import RECIPE_DIR`.
COPY ralph_bootstrap.py /app/ralph_bootstrap.py

WORKDIR /app

# Verify the model code loads.
RUN python -c "from model import RalphBase, RalphConfig; print('model import ok')"
RUN python -c "from proof.runner import run_proof_test; print('proof runner import ok')"

# The entry point is the proof runner.
# Mounts expected at runtime:
#   /data       (ro)  training data shards + manifest
#   /submission (ro)  patch.diff + proof_request.json
#   /output           where the proof bundle is written
ENTRYPOINT ["python", "-m", "proof.runner"]
CMD ["--help"]

# --- Labels for traceability ---
ARG GIT_SHA="unknown"
LABEL org.opencontainers.image.source="https://github.com/RalphLabsAI/ralph"
LABEL org.opencontainers.image.revision="${GIT_SHA}"
LABEL org.opencontainers.image.description="Ralph proof-test container — canonical training + attestation"

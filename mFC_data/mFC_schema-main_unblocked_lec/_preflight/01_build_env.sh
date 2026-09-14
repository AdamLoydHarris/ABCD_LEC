#!/usr/bin/env bash
# Build the pinned `mfc_replication` environment.
#
# WHY A PINNED ENV RATHER THAN CODE EDITS: the deposited helpers build ragged arrays via
# bare `np.asarray([...])`, which numpy >= 1.24 raises on. Patching them with
# `dtype=object` is NOT equivalent -- `concatenate_complex2` is polymorphic: its inner
# call needs object dtype, its outer call receives per-bin scalars and must stay numeric
# so that `~np.isnan(phases_conc)` works. Forcing object makes np.isnan raise TypeError.
# Pinning numpy < 1.24 therefore costs ZERO notebook edits and is the only faithful route.
#
# Primary target = the paper's stated stack (numpy 1.22.0 / scipy 1.10.1 / sklearn 1.3.2).
# numba 0.56.4 is the pin that matters: last series accepting numpy<1.24 on py3.9, and
# umap-learn needs it.
set -euo pipefail

CONDA_BASE=/ceph/apps/ubuntu-20/packages/miniconda/23.10.0
ENV_NAME=mfc_replication
ENV_DIR=/nfs/nhome/live/aharris/.conda/envs/${ENV_NAME}
LOGDIR=/ceph/behrens/adam_harris/Taskspace_abstraction_lEC/mFC_data/lec_replication_run/logs
mkdir -p "$LOGDIR"

echo "=== [1/4] creating env ${ENV_NAME} (python 3.9) ==="
PY="$ENV_DIR/bin/python"
PIP="$ENV_DIR/bin/pip"

# Test for a WORKING interpreter, not merely the directory: a half-finished
# `conda create` leaves $ENV_DIR present but without bin/python, and reusing that
# silently skips creation and then fails every pip call.
if [ -x "$PY" ]; then
    echo "env already usable at $ENV_DIR -- reusing"
else
    if [ -d "$ENV_DIR" ]; then
        echo "removing incomplete env at $ENV_DIR"
        "$CONDA_BASE/bin/conda" env remove -y -n "$ENV_NAME" || rm -rf "$ENV_DIR"
    fi
    "$CONDA_BASE/bin/conda" create -y -n "$ENV_NAME" python=3.9
fi

[ -x "$PY" ] || { echo "FATAL: $PY still missing after create"; exit 1; }
echo "python: $($PY -V)"

echo "=== [2/4] probing wheel availability under py3.9 before committing ==="
# The plan flags this: `pip index versions scipy` under py3.12 does not list 1.10.x
# (a wheel-filter artefact). Re-check under 3.9, where it should be available.
for pkg in numpy scipy scikit-learn numba; do
    echo "--- $pkg"
    $PIP index versions "$pkg" 2>&1 | head -3 || echo "  (probe failed for $pkg)"
done

echo "=== [3/4] installing pinned stack ==="
$PIP install --no-input \
    "numpy==1.22.0" \
    "scipy==1.10.1" \
    "scikit-learn==1.3.2" \
    "pandas==2.0.3" \
    "matplotlib==3.7.3" \
    "seaborn==0.13.2" \
    "statsmodels==0.14.0" \
    "pingouin==0.5.4" \
    "numba==0.56.4" \
    "llvmlite==0.39.1" \
    "umap-learn==0.5.3" \
    joblib jupyter ipykernel nbclient nbconvert

echo "=== [4/4] registering kernel ==="
$PY -m ipykernel install --user --name "$ENV_NAME" --display-name "$ENV_NAME (numpy 1.22)"

$PIP freeze > "$LOGDIR/pip_freeze.txt"
echo "wrote $LOGDIR/pip_freeze.txt"
echo "ENV BUILD COMPLETE"

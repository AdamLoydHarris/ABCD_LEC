#!/usr/bin/env bash
# Build a WRITABLE mirror of the deposited data, so no notebook can write to the deposit.
#
# WHY A PHYSICAL COPY RATHER THAN LINKS: `data/Intermediate_objects/` is 218 unique
# objects plus 3841 HARDLINKS into the six type-subfolders under `data/`. Every notebook
# writes via `np.save(Input_folder + ...)`. Figure2 cell 20 saves Neuron_raw_/Location_raw_/
# XY_raw_/trialtimes_ under *single-day* recday names -- names hardlinked to the deposited
# raw arrays -- and it does not merely overwrite them, it RENUMBERS them (deposited
# me08_12092021 has session indices [0,1,3]; cell 20 would write contiguous 0,1,2).
# A copy makes that structurally impossible.
#
# `cp -a` preserves hardlinks *within* the copy, so the mirror costs ~51 GB rather than
# ~70 GB, and links inside the mirror are harmless (both ends are ours, not the deposit).
set -euo pipefail

ROOT=/ceph/behrens/adam_harris/Taskspace_abstraction_lEC/mFC_data
DEPOSIT="$ROOT/data"
MIRROR="$ROOT/lec_replication_run/data"
OUT="$ROOT/lec_replication_run/Output_folder"
LOGDIR="$ROOT/lec_replication_run/logs"

mkdir -p "$LOGDIR" "$OUT/ephys" "$OUT/behaviour"

[ "$DEPOSIT" != "$MIRROR" ] || { echo "FATAL: mirror == deposit"; exit 1; }

echo "=== [1/4] snapshotting the deposit (pre-run integrity baseline) ==="
find "$DEPOSIT" -type f -printf '%i\t%s\t%T@\t%p\n' | sort > "$LOGDIR/deposit_snapshot_pre.tsv"
wc -l < "$LOGDIR/deposit_snapshot_pre.tsv" | xargs echo "  files recorded:"
# md5 a deterministic 200-file sample for content verification later.
# NOTE: cap with awk, not `head` -- `head` closing the pipe sends SIGPIPE upstream, and
# with `set -o pipefail` that makes the whole pipeline exit 141 and `set -e` abort here.
find "$DEPOSIT" -type f | sort | awk 'NR%20==1 && ++n<=200' \
    | xargs -d '\n' md5sum > "$LOGDIR/deposit_md5_sample_pre.txt"
wc -l < "$LOGDIR/deposit_md5_sample_pre.txt" | xargs echo "  md5 sample:"

echo "=== [2/4] copying $DEPOSIT -> $MIRROR ==="
if [ -d "$MIRROR" ] && [ -f "$LOGDIR/mirror_complete.stamp" ]; then
    echo "  mirror already complete (stamp present) -- skipping copy"
else
    rm -f "$LOGDIR/mirror_complete.stamp"
    mkdir -p "$MIRROR"
    # -a: archive (preserves hardlinks within the copy). Trailing /. copies contents.
    cp -a "$DEPOSIT/." "$MIRROR/"
    touch "$LOGDIR/mirror_complete.stamp"
fi
du -sh --apparent-size "$MIRROR" | xargs echo "  mirror apparent size:"

echo "=== [3/4] verifying the mirror is independent of the deposit ==="
python3 - "$DEPOSIT" "$MIRROR" <<'PY'
import os, sys
dep, mir = sys.argv[1], sys.argv[2]
# Sample a few known-hardlinked raw arrays; their mirror copies must have DIFFERENT inodes
# from the deposit (otherwise a write would reach the deposit).
probes = [
    'Intermediate_objects/Neuron_raw_ah04_01122021_02122021_0.npy',
    'Intermediate_objects/XY_raw_ah04_01122021_02122021_0.npy',
    'Intermediate_objects/trialtimes_me11_05122021_06122021_0.npy',
    'Intermediate_objects/Task_data_ab03_01092023_02092023.npy',
]
bad = 0
for p in probes:
    dp, mp = os.path.join(dep, p), os.path.join(mir, p)
    if not os.path.exists(mp):
        print(f'  MISSING in mirror: {p}'); bad += 1; continue
    di, mi = os.stat(dp).st_ino, os.stat(mp).st_ino
    same = (di == mi)
    print(f'  {"SHARED INODE (BAD)" if same else "independent"}: {p}')
    if same: bad += 1
if bad:
    print(f'FATAL: {bad} probe(s) failed -- mirror is not independent'); sys.exit(1)
print('  all probes independent')

# File-count parity per directory
for sub in ['Intermediate_objects', 'Neuronal_activity/Awake', 'Maze location',
            'XY position', 'Trial_times', 'Tasks', 'MetaData']:
    d, m = os.path.join(dep, sub), os.path.join(mir, sub)
    if not os.path.isdir(d):
        print(f'  (no such deposit dir: {sub})'); continue
    nd, nm = len(os.listdir(d)), len(os.listdir(m))
    flag = 'OK' if nd == nm else 'MISMATCH'
    print(f'  {flag:8s} {sub}: deposit {nd} vs mirror {nm}')
    if nd != nm: sys.exit(1)
PY

echo "=== [4/4] deposit write-protection (opt-in) ==="
# `mFC_data/data` sits in the shared `behrens` group and is group-writable, so locking it
# blocks collaborators, not just this run. Step [3/4] already proves the mirror holds
# independent inodes and every notebook's Input_folder points at the mirror, so the lock
# is belt-and-braces rather than load-bearing. It is therefore OPT-IN.
#
# Enable with LOCK_DEPOSIT=1. To undo a lock, restore the ORIGINAL modes from the mirror
# (which `cp -a` preserved) rather than guessing at them -- the tree has four distinct
# modes (files 664/770, dirs 2775/2770), so a blanket chmod is not exact:
#
#   python3 - <<'EOF'
#   import os, stat
#   DEP, MIR = 'data', 'lec_replication_run/data'
#   for root, dirs, files in os.walk(MIR):
#       rel = os.path.relpath(root, MIR)
#       for n in dirs + files:
#           dp = os.path.join(DEP, rel, n) if rel != '.' else os.path.join(DEP, n)
#           mp = os.path.join(root, n)
#           if os.path.lexists(dp) and not os.path.islink(mp):
#               os.chmod(dp, stat.S_IMODE(os.stat(mp).st_mode))
#   EOF
if [ "${LOCK_DEPOSIT:-0}" = "1" ]; then
    chmod -R a-w "$DEPOSIT"
    echo "  LOCK_DEPOSIT=1 -> chmod -R a-w applied to $DEPOSIT"
    echo "  NOTE: this also removes GROUP write on a shared behrens directory"
else
    echo "  skipped (set LOCK_DEPOSIT=1 to enable)"
    echo "  the mirror + repointed Input_folder are what actually protect the deposit"
fi

echo "MIRROR BUILD COMPLETE"

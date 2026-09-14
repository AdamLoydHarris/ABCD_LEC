#!/usr/bin/env bash
# One compact progress line per invocation, for periodic status reporting.
# Progress inside long cells is read from the per-recday files the notebooks save as
# they go -- that is the only way to see inside a cell, because nbclient buffers a
# cell's stdout until the cell completes.
D=/ceph/behrens/adam_harris/Taskspace_abstraction_lEC/mFC_data
M=$D/replication_run/data/Intermediate_objects
L=$D/replication_run/logs

cur () {  # current cell of a run, or DONE/DEAD
  local log=$1 pat=$2
  [ -f "$log" ] || { echo "notstarted"; return; }
  if grep -q "cells run" "$log" 2>/dev/null; then echo "DONE"; return; fi
  if ! pgrep -f "40_run_notebook.py $pat" >/dev/null 2>&1; then echo "DEAD"; return; fi
  awk '/^\[ *[0-9]+\] RUN/{c=$2} END{gsub(/[][]/,"",c); print "c"c}' "$log"
}
n () { ls $M/$1 2>/dev/null | wc -l | tr -d ' '; }

P_C21=$(n 'Poisson_GLM_anchoring_coeffs_all_*.npy')
P_C26=$(n 'Poisson_Predicted_Actual_correlation_mean_*.npy')
E_C21=$(n 'GLM_anchoring_coeffs_all_*.npy')
E_C26=$(n 'Predicted_Actual_correlation_mean_*.npy')
F3_X=$(n 'Xneuron_correlations_*.npy')
F3_S=$(ls $M/sigma_goalprogress.npy 2>/dev/null | wc -l | tr -d ' ')

echo "F5R-poisson[$(cur $L/run_Figure5_Regression.log Figure5_Regression.ipynb) c21:$P_C21/25 c26:$P_C26/25] \
F5R-enet[$(cur $L/run_Figure5_Regression_elasticnet.log Figure5_Regression_elasticnet) c21:$E_C21/25 c26:$E_C26/25] \
F3fast[$(cur $L/run_Figure3_fast.log Figure3_fast) Xneuron:$F3_X sigma:$F3_S] \
load=$(cut -d' ' -f1 /proc/loadavg)"

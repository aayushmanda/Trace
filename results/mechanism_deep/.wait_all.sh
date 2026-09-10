#!/usr/bin/env bash
base=/home/hariguru/aayus/trace/results/mechanism_deep/runs
depths="2 4 6 8"
seeds="42 43 44 45 46"
while true; do
  done_count=0; total=0
  for d in $depths; do
    for s in $seeds; do
      total=$((total+1))
      [ -f "$base/D${d}_seed${s}/persist.json" ] && done_count=$((done_count+1))
    done
  done
  echo "progress: $done_count/$total"
  if [ "$done_count" -eq "$total" ]; then
    echo ALL_MECHANISM_RUNS_DONE
    break
  fi
  if ! pgrep -f "run_mechanism_study.sh" >/dev/null; then
    echo "DRIVER_EXITED_progress_${done_count}_of_${total}"
    break
  fi
  sleep 20
done

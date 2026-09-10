# E2 week-2 freeze

`protocol.json` is the 5-rate × 10-seed confirmation protocol (40 calibration jobs, 80 confirm jobs). PDF Table 6 was not overwritten.

**This week:** `plan` + two CPU `single` cells (`calibration/process_process_s7001_lr0.0005_clip1`, `calibration/outcome_process_s7001_lr0.0005_clip1`, 2 steps). Pipeline proof only.

**Blocked:** full `calibrate` (40 jobs) and `confirm` (80 jobs). GPU 2/3 had free memory but ~100% util from other users. Commands: [RUN.md leftover GPU](../../../RUN.md).

# T-maze Coordinate Alignment

This project contains scripts for processing DeepLabCut T-maze tracking exports.

## Workflow

1. `analyze_tmaze_normalization.py`
   - Earlier frame-level rigid normalization prototype.
   - Kept for reference.

2. `align_tmaze_coordinate_system.py`
   - Current workflow.
   - Excludes the `20250703_191708979` dataset.
   - Aligns the remaining sessions to the `20250608_160522529` reference session.
   - Filters plots to the main connected T-maze body.
   - Calibrates aligned coordinates to centimeters using the 89 cm stem reference.

## Run

Use the Anaconda `data` environment:

```powershell
& "C:\Users\17809\anaconda3\envs\data\python.exe" "C:\work\Tmaze\align_tmaze_coordinate_system.py"
```

Outputs are written to:

```text
C:\work\Tmaze\tmaze_coordinate_aligned
```

The output CSV files include raw pixel coordinates, aligned pixel coordinates, centimeter coordinates, and QC flags:

- `center_valid`
- `center_in_reference_maze`
- `center_ref_distance_px`
- `*_x_cm`
- `*_y_cm`

For downstream analysis, use rows where:

```text
center_valid == 1
center_in_reference_maze == 1
```

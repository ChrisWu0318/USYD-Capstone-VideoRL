# Before (buggy):
if gt_number is None:
    ...  # no return or else
if out_number is None:
    ...  # no return or else
rel_diff = abs(gt_number - out_number) / ...  # TypeError when None

# After (fixed):
if gt_number is None or out_number is None:
    return 0.0
rel_diff = abs(gt_number - out_number) / max(abs(gt_number), 1e-6)
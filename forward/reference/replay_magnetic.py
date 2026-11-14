"""Replay the archived finite-cylinder field using saved rigid magnet motion."""
from pathlib import Path
import json

import numpy as np
from magnetic_core import contributions
from coupling_core import hall_delta

folder = Path(__file__).resolve().parent
with np.load(folder / "nominal.npz", allow_pickle=False) as a:
    nominal = {k: a[k] for k in a.files}
with np.load(folder / "rigid_source_motion.npz", allow_pickle=False) as a:
    np.testing.assert_array_equal(a["source_centers_mm"], nominal["centers_mm"])
    np.testing.assert_array_equal(a["nominal_frames"], nominal["frames"])
    base = contributions(nominal["centers_mm"], nominal["frames"], nominal["sensors_mm"], order=96)
    moved = contributions(a["moved_centers_mm"], a["moved_frames"], nominal["sensors_mm"], order=96)
delta = hall_delta(nominal, moved, base)
with np.load(folder / "reference_fields.npz", allow_pickle=False) as a:
    reference = a["order96"]
np.testing.assert_allclose(delta, reference, rtol=1e-10, atol=1e-7)
print(json.dumps({"shape": list(delta.shape), "max_abs_uT_per_T": float(np.max(np.abs(delta-reference)))}, indent=2))

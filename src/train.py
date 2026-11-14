"""Train one or all frozen-protocol configurations on the published development split."""
import argparse
from pathlib import Path
import json

import numpy as np
from paper1_static_fit import ARCHITECTURES, fit
from projection import project

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("output", type=Path, help="A new directory to hold the trained checkpoints.")
p.add_argument("--seed", type=int, choices=(20261110,20261111,20261112), default=20261110)
p.add_argument("--model", choices=("small","double","both"), default="both")
args = p.parse_args()
root = Path(__file__).resolve().parent.parent
with np.load(root/"data/fields.npz",allow_pickle=False) as d:
    region = [r.replace("dev_","").replace("seam_side","seam").replace("seam_cap","seam") for r in d["dev_region"]]
    train = (d["train_fields"], d["train_xyz_mm"], d["dev_fields"], d["dev_xyz_mm"])
args.output.mkdir()
for name,widths in zip(("small","double"),ARCHITECTURES,strict=True):
    if args.model not in ("both",name):
        continue
    result = fit(*train, region, widths, args.seed, 0.008, project, limit=12000)
    best = result["best"]
    np.savez_compressed(args.output/f"{name}-{args.seed}.npz",*best["params"],
                        center=result["center"],spread=result["spread"])
    del best["params"]
    (args.output/f"{name}-{args.seed}.json").write_text(json.dumps({
        "history":result["history"],"best":best,"stable":result["stable"]},indent=2)+"\n")

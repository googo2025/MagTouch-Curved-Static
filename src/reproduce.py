"""Recompute the paper's Table 1, Table 2 and regional errors from the release."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.interpolate import RBFInterpolator
from paper1_evaluate_pairs import evaluate, load_predictor
from paper1_paired_metrics import compare_inputs, outside_interpolator, summarize, position_errors
from projection import project

ROOT = Path(__file__).resolve().parent.parent


def load_json(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def tps_blend(xyz, fields, config):
    linear = outside_interpolator(xyz, fields)
    period = 40 * math.pi

    def unroll(point):
        x, y, z = point
        return (20 * math.atan2(y, x) % period,
                z if z <= 100 else 100 + 20 * math.atan2(z - 100, math.hypot(x, y)))

    uv = np.asarray([unroll(p) for p in xyz])

    def interpolate(ref):
        field, _ = linear(ref)
        if field is None or ref[2] >= 65:
            return field
        phi, height = unroll(ref)
        offsets = np.column_stack(((uv[:, 0] - phi + period / 2) % period - period / 2,
                                   config["axial_scale"] * (uv[:, 1] - height)))
        distance = np.linalg.norm(offsets, axis=1)
        near = np.argsort(distance, kind="stable")[:config["neighbors"]]
        if distance[near[0]] < 1e-10:
            return fields[near[0]]
        spline = RBFInterpolator(offsets[near] / distance[near[-1]], fields[near],
                                 smoothing=config["smoothing"], kernel="thin_plate_spline", degree=1)
        t = np.clip((ref[2] - 35.) / 30., 0, 1)
        return field + (1 - t * t * (3 - 2 * t)) * (spline(np.zeros((1, 2)))[0] - field)

    return interpolate


def check_distance(expected, computed):
    for name in ("direct", "interpolated"):
        a, b = expected[name], computed[name]
        assert (a is None) == (b is None)
        if a is not None:
            np.testing.assert_allclose(a["distance_mm"], b["distance_mm"], rtol=0, atol=1e-8)


def demo_predictions(data):
    norm = np.linalg.norm(data["train_fields"], axis=1, keepdims=True)
    normalized = data["train_fields"] / norm
    models = {}
    for depth, widths in ((1, [51, 16, 3]), (2, [51, 64, 64, 3])):
        for seed in (20261110, 20261111, 20261112):
            label = f"static_{depth}_{seed}"
            models[label] = load_predictor(ROOT / "models" / f"{label}.npz", widths, project)

    def template(field):
        return data["train_xyz_mm"][np.argmax(normalized @ (field / np.linalg.norm(field)))]

    models["template"] = template
    return models


def primary(data, models):
    rows = [dict(load=s, xyz=x, region=r) for s, x, r in
            zip(data["final_site"], data["final_xyz_mm"], data["final_region"], strict=True)]
    keys = [(region, int(phi), int(level)) for region, phi, level in data["grid_keys"]]
    grid = dict(zip(keys, data["grid_fields"], strict=True))
    allowed = {key for key, ok in zip(keys, data["strict_allowed"], strict=True) if ok}
    computed = evaluate(rows, data["final_fields"], grid, allowed, data["outside_xyz_mm"], data["outside_fields"], models)
    frozen = load_json("results/primary-evaluation.json")
    key = lambda row: (row["site"], row["model"])
    for expected, actual in zip(sorted(frozen["details"], key=key), sorted(computed["details"], key=key), strict=True):
        assert key(expected) == key(actual)
        for mode, record in expected["comparisons"].items():
            check_distance(record, actual["comparisons"][mode])
    historic = {(r["site"], r["model"]): r for r in frozen["historical_direct"]}
    for s, ref, field in zip(data["historical_site"], data["historical_xyz_mm"], data["historical_fields"], strict=True):
        for model, predict in models.items():
            value = position_errors(ref, predict(field))["distance_mm"]
            np.testing.assert_allclose(historic[(s, model)]["distance_mm"], value, rtol=0, atol=1e-8)
    table = {}
    for architecture, prefix in (("small", "static_1"), ("double", "static_2"), ("template", "template")):
        records = [r for r in computed["details"] if r["model"].startswith(prefix)]
        table[architecture] = {}
        for mode in ("direct", "engineering", "outside"):
            values = [r["comparisons"]["engineering" if mode=="direct" else mode]
                      ["direct" if mode=="direct" else "interpolated"]["distance_mm"] for r in records]
            table[architecture][mode] = {"mean": float(np.mean(values)), "rmse": float(np.sqrt(np.mean(np.square(values)))),
                                        "p95": float(np.quantile(values, .95))}
    return table, computed["summary"]


def supplementary(data, models):
    layouts = load_json("data/anchor-layouts.json")
    rows = {}
    for role in ("train", "boundary_development"):
        rows.update({s: (x, f) for s, x, f in zip(data[f"{role}_site"], data[f"{role}_xyz_mm"],
                                                data[f"{role}_fields"], strict=True)})
    groups = layouts["boundary"]
    selected = groups[layouts["selected"]]

    def anchors(ids):
        return tuple(map(np.stack, zip(*(rows[i] for i in ids))))

    def linear(ids):
        fn = outside_interpolator(*anchors(ids))
        return lambda p: fn(p)[0]

    pairs = {
        "equal-count": {"uniform147": linear(layouts["equal_count"]["uniform147"]),
                        "targeted147": linear(layouts["equal_count"]["targeted147"])},
        "boundary": {"uniform147": linear(groups["uniform147"]), "boundary147": linear(selected)},
    }
    for role in ("spline", "transition"):
        settings = load_json(f"data/{role}-protocol.json")["rbf_config"]
        pairs[role] = {"boundary147": linear(selected), "local_rbf": tps_blend(*anchors(selected), settings)}
    result = {}
    for role, variants in pairs.items():
        archive = load_json(f"results/{role}.json")
        frozen = {(p["site"], p["seed"]): p.get("values", p.get("metrics")) for p in archive["details"]}
        key = role.replace("-", "_")
        verified = {k: [] for k in variants}
        for site, ref, field in zip(data[f"{key}_site"], data[f"{key}_xyz_mm"], data[f"{key}_fields"], strict=True):
            inputs = {variant: method(ref) for variant, method in variants.items()}
            for seed, model in models.items():
                if not seed.startswith("static_2"):
                    continue
                for variant, values in compare_inputs(ref, field, inputs, model).items():
                    check_distance(frozen[(site, seed)][variant], values)
                    verified[variant].append(values)
        result[role] = {variant: summarize(records) for variant, records in verified.items()}
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, help="Optional output JSON path.")
    options = cli.parse_args()
    with np.load(ROOT / "data/fields.npz", allow_pickle=False) as input_data:
        data = {key: input_data[key] for key in input_data.files}
    models = demo_predictions(data)
    table, regions = primary(data, models)
    bundle = dict(primary=table, regions=regions, supplementary=supplementary(data, models))
    if options.output is not None:
        options.output.write_text(json.dumps(bundle, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"primary":table, "supplementary":{k:{m: s["interpolated"]["distance_mm"]["p95"]
                          for m,s in p.items()} for k,p in bundle["supplementary"].items()}}, indent=2, allow_nan=False))


if __name__=="__main__":
    main()

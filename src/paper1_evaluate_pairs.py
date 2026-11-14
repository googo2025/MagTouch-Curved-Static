"""Compare frozen localization models on matched direct and interpolated fields."""

import math
from collections import defaultdict

import numpy as np

from paper1_paired_metrics import (
    BETAS, HEIGHTS, compare_inputs, field_inputs, outside_interpolator, summarize,
)
from paper1_static_fit import OFFSET, SCALE, forward


def engineering_grid(rows, load_features, role, excluded_ids):
    definitions = {}
    for phi in range(0, 360, 45):
        a = math.radians(phi)
        for z in HEIGHTS:
            definitions['side', phi, z] = (20*math.cos(a), 20*math.sin(a), z)
        for beta in BETAS[1:-1]:
            b = math.radians(beta)
            r = 20*math.sin(b)
            definitions['cap', phi, beta] = (r*math.cos(a), r*math.sin(a), 100+20*math.cos(b))
    definitions['apex', 0, 0] = (0, 0, 120)
    matched = []
    for xyz in definitions.values():
        [row] = [row for row in rows if math.dist(row['xyz'], xyz) <= 1e-7]
        matched.append(row)
    assert len({row['load'] for row in matched}) == len(definitions) == 121
    fields, _ = load_features(matched)
    return dict(zip(definitions, fields)), {
        key for (key, xyz), row in zip(definitions.items(), matched)
        if role(xyz) == 'train' and row['load'] not in excluded_ids
    }


def load_predictor(weights, widths, project):
    with np.load(weights, allow_pickle=False) as archive:
        center = archive['center']
        spread = archive['spread']
        params = [archive[f'arr_{i}'] for i in range(2*(len(widths)-1))]
    assert center.shape == spread.shape == (51,)
    assert np.isfinite(center).all() and np.isfinite(spread).all() and (spread > 0).all()
    for n, m, weight, bias in zip(widths, widths[1:], params[::2], params[1::2]):
        assert weight.shape == (n, m) and bias.shape == (m,)
        assert np.isfinite(weight).all() and np.isfinite(bias).all()

    def predict(field):
        raw = forward((field-center)/spread, params)[-1] * SCALE + OFFSET
        return project(raw)

    return predict


def evaluate(rows, direct_fields, grid, allowed, anchor_xyz, anchor_fields, predictors):
    assert len(rows) == len(direct_fields) and predictors
    outside = outside_interpolator(anchor_xyz, anchor_fields)
    details = []
    grouped = defaultdict(list)
    for row, direct in zip(rows, direct_fields):
        reference = np.asarray(row['xyz'])
        fields = field_inputs(reference, grid, allowed)
        fields['outside'], span = outside(reference)
        for label, predict in predictors.items():
            comparisons = compare_inputs(reference, direct, fields, predict)
            for mode, result in comparisons.items():
                grouped[row['region'], label, mode].append(result)
            details.append(dict(site=row['load'], region=row['region'], model=label,
                                outside_span_mm=span, comparisons=comparisons))
    summary = [dict(region=region, model=model, mode=mode, **summarize(records))
               for (region, model, mode), records in sorted(grouped.items())]
    return dict(details=details, summary=summary)


"""Paired static localization metrics on the 20 mm cylinder and cap."""

import bisect
import math

import numpy as np

HEIGHTS = (3, 12, 30, 50, 70, 88, 97, 100)
BETAS = (0, 15, 30, 45, 60, 75, 85, 88, 90)
CENTER = np.array([0., 0., 100.])


def coordinates(xyz):
    x, y, z = xyz
    r = math.hypot(x, y)
    phi = None if r <= 1e-7 else math.degrees(math.atan2(y, x)) % 360
    cap = z > 100 + 1e-7
    beta = math.degrees(math.atan2(r, z - 100)) if cap else None
    return phi, beta, cap


def position_errors(reference, prediction):
    ref_phi, ref_beta, ref_cap = coordinates(reference)
    pred_phi, pred_beta, pred_cap = coordinates(prediction)
    phi = (abs((pred_phi - ref_phi + 180) % 360 - 180)
           if ref_phi is not None and pred_phi is not None else None)
    result = dict(distance_mm=float(np.linalg.norm(prediction - reference)),
                  azimuth_deg=phi, crosses_seam=bool(ref_cap != pred_cap))
    if ref_cap and pred_cap:
        a, b = reference - CENTER, prediction - CENTER
        result.update(polar_deg=abs(pred_beta - ref_beta),
                      cap_angle_deg=math.degrees(math.atan2(np.linalg.norm(np.cross(a, b)), a @ b)))
    elif not ref_cap:
        result['height_mm'] = float(abs(prediction[2] - reference[2]))
    ref_meridian = 100 + 20 * math.radians(90 - ref_beta) if ref_cap else reference[2]
    pred_meridian = 100 + 20 * math.radians(90 - pred_beta) if pred_cap else prediction[2]
    result['meridian_mm'] = float(abs(pred_meridian - ref_meridian))
    return result


def interpolation_stencil(xyz):
    phi, beta, cap = coordinates(xyz)
    if cap and phi is None:
        return {('apex', 0, 0): 1.}
    if phi is None:
        return {}
    levels = BETAS if cap else HEIGHTS
    level = beta if cap else float(xyz[2])
    if level < levels[0] - 1e-7 or level > levels[-1] + 1e-7:
        return {}
    level = min(levels[-1], max(levels[0], level))
    index = min(len(levels) - 2, bisect.bisect_right(levels, level) - 1)
    lo, hi = levels[index:index+2]
    vertical = (level - lo) / (hi - lo)
    start = 45 * int(phi // 45)
    fraction = (phi - start) / 45
    weights = {}
    for az, u in ((start, 1-fraction), ((start+45) % 360, fraction)):
        for height, v in ((lo, 1-vertical), (hi, vertical)):
            if u*v <= 1e-14:
                continue
            if cap and height == 0:
                key = ('apex', 0, 0)
            elif cap and height != 90:
                key = ('cap', az, height)
            else:
                key = ('side', az, 100 if cap else height)
            weights[key] = weights.get(key, 0.) + u*v
    return weights


def paired_errors(reference, direct_prediction, interpolated_prediction):
    direct = position_errors(reference, direct_prediction)
    if interpolated_prediction is None:
        return dict(direct=direct, interpolated=None, delta=None)
    interpolated = position_errors(reference, interpolated_prediction)
    delta = {key: interpolated[key] - direct[key] for key in direct.keys() & interpolated.keys()
             if key != 'crosses_seam' and direct[key] is not None and interpolated[key] is not None}
    return dict(direct=direct, interpolated=interpolated, delta=delta)


def field_inputs(xyz, grid, allowed_anchors):
    """Use a common stencil and reject incomplete support without extrapolating."""
    stencil = interpolation_stencil(xyz)
    if not stencil or not stencil.keys() <= grid.keys():
        return dict(engineering=None, strict=None)
    engineering = sum(weight * grid[key] for key, weight in stencil.items())
    strict = engineering if stencil.keys() <= allowed_anchors else None
    return dict(engineering=engineering, strict=strict)


def compare_inputs(reference, direct_field, interpolated_fields, predict):
    direct = predict(direct_field)
    norm = np.linalg.norm(direct_field)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError('Invalid direct field')
    results = {}
    for mode, field in interpolated_fields.items():
        prediction = None if field is None else predict(field)
        paired = paired_errors(reference, direct, prediction)
        paired['relative_field_error'] = (
            None if field is None else float(np.linalg.norm(field - direct_field) / norm)
        )
        fitted = None
        if field is not None:
            energy = float(np.dot(field, field))
            if energy > 0:
                scale = max(0.0, float(np.dot(field, direct_field)) / energy)
                fitted = float(np.linalg.norm(scale * field - direct_field) / norm)
        paired['nonnegative_fit_relative_error'] = fitted
        results[mode] = paired
    return results


def summarize(rows):
    """Keep the direct, supported-interpolation and matched denominators visible."""
    def stats(records):
        keys = sorted(set().union(*(r.keys() for r in records)) - {'crosses_seam'})
        values = {}
        for key in keys:
            valid = [r[key] for r in records if r.get(key) is not None]
            if valid:
                values[key] = dict(n=len(valid), mean=float(np.mean(valid)),
                                   p50=float(np.quantile(valid, .5)),
                                   p95=float(np.quantile(valid, .95)),
                                   maximum=float(max(valid)))
        if any('crosses_seam' in r for r in records):
            values['crosses_seam'] = sum(r.get('crosses_seam', False) for r in records)
        return values

    supported = [row for row in rows if row['interpolated'] is not None]
    return dict(n=len(rows), paired_n=len(supported),
                direct=stats([row['direct'] for row in rows]),
                direct_matched=stats([row['direct'] for row in supported]),
                interpolated=stats([row['interpolated'] for row in supported]),
                delta=stats([row['delta'] for row in supported]),
                field=stats([
                    dict(relative=row['relative_field_error'],
                         nonnegative_fit_relative=row['nonnegative_fit_relative_error'])
                    for row in supported]))


def outside_interpolator(xyz, fields):
    """Build a periodic interpolator from the frozen outside-anchor cohort."""
    from scipy.spatial import Delaunay

    def surface(point):
        x, y, z = point
        if math.hypot(x, y) <= 1e-7:
            return None
        phi = math.atan2(y, x) % (2*math.pi)
        meridian = z if z <= 100 else 100 + 20*math.atan2(z-100, math.hypot(x, y))
        return 20*phi, meridian

    if len(xyz) != len(fields) or not np.isfinite(fields).all():
        raise ValueError('Invalid outside-anchor fields')
    usable = [(pos, i) for i, point in enumerate(xyz) if (pos := surface(point)) is not None]
    period = 40*math.pi
    points = np.array([(u+turn*period, v) for turn in (-1, 0, 1) for (u, v), _ in usable])
    source = np.tile([i for _, i in usable], 3)
    mesh = Delaunay(points)

    def evaluate(point):
        position = surface(point)
        if position is None:
            return None, None
        position = np.asarray(position)
        simplex = int(mesh.find_simplex(position))
        if simplex < 0:
            return None, None
        indices = mesh.simplices[simplex]
        transform = mesh.transform[simplex]
        partial = transform[:2] @ (position-transform[2])
        weights = np.r_[partial, 1-partial.sum()]
        if not np.isfinite(weights).all() or weights.min() < -1e-10:
            raise ValueError('Invalid interpolation weights')
        span = max(math.dist(a, b) for a in points[indices] for b in points[indices])
        return weights @ fields[source[indices]], span

    return evaluate

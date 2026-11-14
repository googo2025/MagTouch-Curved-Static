"""Shared CPU fitting for the two frozen MagTouch static architectures."""

import math

import numpy as np

ARCHITECTURES = ((51, 16, 3), (51, 64, 64, 3))
OFFSET = np.array([0, 0, 60])
SCALE = np.array([20, 20, 60])


def forward(features, params):
    layers = [features]
    for weight, bias in zip(params[::2], params[1::2]):
        value = layers[-1] @ weight + bias
        layers.append(np.tanh(value) if len(layers) < len(params) // 2 else value)
    return layers


def fit(train_x, train_xyz, dev_x, dev_xyz, regions, widths, seed, lr, project,
        limit=12000):
    """Return the best development checkpoint and the stability evidence."""
    if widths not in ARCHITECTURES or train_x.shape[1] != 51 or dev_x.shape[1] != 51:
        raise ValueError("Unexpected architecture or Hall channel count")
    groups = [np.asarray(regions) == name for name in ("side", "seam", "cap")]
    if not all(group.any() for group in groups) or not np.logical_or.reduce(groups).all():
        raise ValueError("Development data must cover all three regions")
    center = train_x.mean(0)
    spread = train_x.std(0)
    if not np.isfinite(spread).all() or spread.max() <= 0:
        raise ValueError("Invalid training feature scale")
    spread = np.maximum(spread, spread.max() * 1e-6)
    train = (train_x - center) / spread
    dev = (dev_x - center) / spread
    target = (train_xyz - OFFSET) / SCALE

    rng = np.random.default_rng(seed)
    params = []
    for n, m in zip(widths, widths[1:]):
        params.extend((rng.normal(0, np.sqrt(2 / (n + m)), (n, m)), np.zeros(m)))
    first = [np.zeros_like(param) for param in params]
    second = [np.zeros_like(param) for param in params]
    history = []
    best = None
    stable = False
    for step in range(1, limit + 1):
        progress = max(0., (step - 6000) / (limit - 6000))
        step_rate = lr * (.1 + .45 * (1. + math.cos(math.pi * progress)))
        layers = forward(train, params)
        error = layers[-1] - target
        grad = 2 * error / error.size
        gradients = [None] * len(params)
        for i in range(len(layers) - 2, -1, -1):
            gradients[2*i:2*i+2] = [layers[i].T @ grad, grad.sum(0)]
            if i:
                grad = (grad @ params[2*i].T) * (1 - layers[i] ** 2)
        for i, grad in enumerate(gradients):
            first[i] = .9 * first[i] + .1 * grad
            second[i] = .999 * second[i] + .001 * grad * grad
            params[i] -= step_rate * (first[i] / (1 - .9**step)) / (
                np.sqrt(second[i] / (1 - .999**step)) + 1e-8
            )
        if step % 50:
            continue
        predicted = np.array([project(q) for q in forward(dev, params)[-1] * SCALE + OFFSET])
        errors = np.linalg.norm(predicted - dev_xyz, axis=1)
        if not np.isfinite(errors).all():
            raise ValueError("Non-finite development errors")
        mean = float(np.mean([errors[group].mean() for group in groups]))
        p95 = [float(np.quantile(errors[group], .95)) for group in groups]
        record = dict(step=step, mean=mean, p95=p95)
        history.append(record)
        if best is None or mean < best["mean"]:
            best = dict(record, params=[param.copy() for param in params])
        if step >= 600 and step % 200 == 0:
            windows = [min(history[i:i+4], key=lambda r: r["mean"])
                       for i in range(len(history)-12, len(history), 4)]
            stable = (
                all(a["mean"] - b["mean"] < max(.05, .01 * a["mean"])
                    for a, b in zip(windows, windows[1:]))
                and all(np.all(np.asarray(w["p95"]) <= np.asarray(windows[0]["p95"]) + .20)
                        for w in windows[1:])
            )
    return dict(best=best, history=history, stable=stable, center=center, spread=spread)

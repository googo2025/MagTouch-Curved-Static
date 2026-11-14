"""FP64 exterior finite-cylinder magnetic kernel for the new cylindrical study.

Lengths are millimetres.  Fields are microtesla per unit polarization
J = mu0*M = 1 tesla.  Every cylinder is a uniformly, axially magnetized rigid
body; its local +Z axis is its positive magnetization direction.  This is a
magnetic decoder, not a silicone/contact/FEM model.  Magnetic source fields
may be summed; independently solved contact deformations may not be summed
and presented as a shared mechanical multiple-contact state.

No numerical validation runs at import.  ``validate`` is intended to be called
explicitly on the authorized Ubuntu host with the pinned original source file.
"""

from __future__ import annotations

from functools import lru_cache
import hashlib
import importlib.util
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.special import ellipk, ellipe


CROSS_ATOL_UT_PER_T = 1e-3
CROSS_RTOL = 1e-6
INTERNAL_ATOL_UT_PER_T = 1e-4
INTERNAL_RTOL = 1e-8


def _fp64(value: Any, name: str) -> np.ndarray:
    """Convert finite real coordinates/rotations to FP64; reject complex data."""
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real")
    result = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


@lru_cache(maxsize=16)
def _gauss(order: int) -> tuple[np.ndarray, np.ndarray]:
    """Return cached FP64 Gauss--Legendre nodes and weights; no physical units."""
    if isinstance(order, bool) or not isinstance(order, int) or order < 2:
        raise ValueError("Axial quadrature order must be an integer >= 2")
    nodes, weights = np.polynomial.legendre.leggauss(order)
    nodes.setflags(write=False)
    weights.setflags(write=False)
    return nodes, weights


def _dimensions(radius_mm: float, height_mm: float) -> tuple[float, float]:
    radius, height = float(radius_mm), float(height_mm)
    if not math.isfinite(radius) or not math.isfinite(height) or radius <= 0 or height <= 0:
        raise ValueError("Finite positive cylinder radius and height in mm required")
    return radius, height


def _exterior(points_mm: np.ndarray, radius_mm: float, height_mm: float) -> None:
    """Reject locations inside or on the finite magnetic source, in local mm."""
    radial = np.linalg.norm(points_mm[..., :2], axis=-1)
    if np.any((radial <= radius_mm) & (np.abs(points_mm[..., 2]) <= height_mm / 2)):
        raise ValueError("Exterior magnetic field only: sensor inside/on source")


def cylinder_local_uT_np(
    local_mm: Any,
    radius_mm: float = 2.5,
    height_mm: float = 1.0,
    order: int = 24,
) -> np.ndarray:
    """Return exterior local B (...,3), in microtesla per J=1 tesla.

    Input (...,3) is relative to the cylinder center, in millimetres.  The
    source has uniform axial polarization along local +Z.  The formula uses
    analytic complete elliptic loop integrals and axial Gauss--Legendre
    quadrature, matching ``finite_cylinder_field_r393.cylinder_local_uT``.
    It is not a point dipole.  FP64 is mandatory internally.  Inside/on-source
    coordinates are rejected; the near-axis limit matches the original code.
    """
    points = _fp64(local_mm, "local_mm")
    if points.ndim < 1 or points.shape[-1] != 3:
        raise ValueError("Expected coordinates (...,3), millimetres")
    radius, height = _dimensions(radius_mm, height_mm)
    _exterior(points, radius, height)
    original_shape = points.shape
    p = points.reshape(-1, 3)
    rho = np.linalg.norm(p[:, :2], axis=-1)
    half_height = height / 2
    nodes, weights = _gauss(order)
    weights_mm = weights * half_height
    axial = p[:, 2, None] - half_height * nodes[None, :]
    radial = rho[:, None]
    denominator = np.sqrt((radius + radial) ** 2 + axial**2)
    distance_squared = (radius - radial) ** 2 + axial**2
    parameter = np.clip(4 * radius * radial / denominator**2, 0.0, 1.0 - 1e-16)
    k, e = ellipk(parameter), ellipe(parameter)
    bz = (
        (k + (radius**2 - radial**2 - axial**2) / distance_squared * e)
        / denominator
    ) @ weights_mm / (2 * math.pi)
    br = (
        axial / np.maximum(radial, 1e-30)
        * (-k + (radius**2 + radial**2 + axial**2) / distance_squared * e)
        / denominator
    ) @ weights_mm / (2 * math.pi)
    bx_by = br[:, None] * p[:, :2] / np.maximum(rho[:, None], 1e-30)
    zp, zm = p[:, 2] + half_height, p[:, 2] - half_height
    axial_bz = 0.5 * (
        zp / np.sqrt(zp**2 + radius**2) - zm / np.sqrt(zm**2 + radius**2)
    )
    axial_derivative = 0.5 * radius**2 * (
        (zp**2 + radius**2) ** -1.5 - (zm**2 + radius**2) ** -1.5
    )
    axial_bx_by = -0.5 * p[:, :2] * axial_derivative[:, None]
    on_axis = rho < radius * 1e-5
    bx_by = np.where(on_axis[:, None], axial_bx_by, bx_by)
    bz = np.where(on_axis, axial_bz, bz)
    result = np.concatenate((bx_by, bz[:, None]), axis=-1).reshape(original_shape) * 1e6
    if not np.isfinite(result).all():
        raise FloatingPointError("Nonfinite finite-cylinder output; no clipping/substitution")
    return result


def contributions(
    centers: Any,
    rotations: Any,
    sensors: Any,
    order: int = 24,
    *,
    radius_mm: float = 2.5,
    height_mm: float = 1.0,
) -> np.ndarray:
    """Return individual world-axis fields (M,S,3), microtesla per J=1 tesla.

    ``centers`` (M,3) and ``sensors`` (S,3) share one world frame, in mm.
    ``rotations`` (M,3,3) map magnet-local column vectors into world axes.
    Sources share the explicitly supplied radius/height.  Fields are neither
    summed nor projected into Hall-chip axes here.  For a Hall local-to-world
    matrix Q_s, its chip-axis vector is Q_s.T @ B_world.  Signed J values,
    baseline subtraction, saturation and sensor noise belong to the caller.
    """
    centers_np = _fp64(centers, "centers")
    rotations_np = _fp64(rotations, "rotations")
    sensors_np = _fp64(sensors, "sensors")
    if centers_np.ndim != 2 or centers_np.shape[-1] != 3 or len(centers_np) == 0:
        raise ValueError("centers must have shape (M,3), M >= 1")
    if rotations_np.shape != (len(centers_np), 3, 3):
        raise ValueError("rotations must have shape (M,3,3)")
    if sensors_np.ndim != 2 or sensors_np.shape[-1] != 3 or len(sensors_np) == 0:
        raise ValueError("sensors must have shape (S,3), S >= 1")
    gram = np.swapaxes(rotations_np, -1, -2) @ rotations_np
    if (
        np.max(np.abs(gram - np.eye(3, dtype=np.float64))) > 2e-5
        or np.max(np.abs(np.linalg.det(rotations_np) - 1.0)) > 2e-5
    ):
        raise ValueError("Each source rotation must be proper rigid: R.T R=I, det R=+1")
    offsets = sensors_np[None, :, :] - centers_np[:, None, :]
    local = np.einsum("msi,mij->msj", offsets, rotations_np)
    local_field = cylinder_local_uT_np(local, radius_mm, height_mm, order)
    return np.einsum("msj,mij->msi", local_field, rotations_np)


def _disk_charge_uT_per_T(
    local_mm: Any,
    radius_mm: float,
    height_mm: float,
    radial_order: int,
    angular_order: int,
) -> np.ndarray:
    """Independent top/bottom magnetic-charge surface integral, µT per J=1 T.

    Uniform axial polarization is represented by opposite surface charges on
    the two circular faces.  Direct Coulomb-vector integration uses radial
    Gauss--Legendre and periodic midpoint azimuth quadrature.  It uses neither
    elliptic integrals nor the axial-loop expression.  Exterior points only;
    validate() deliberately chooses moderate separation from source surfaces.
    """
    points = _fp64(local_mm, "charge-kernel points")
    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError("Charge-kernel coordinates must have shape (N,3), mm")
    radius, height = _dimensions(radius_mm, height_mm)
    _exterior(points, radius, height)
    if isinstance(angular_order, bool) or not isinstance(angular_order, int) or angular_order < 8:
        raise ValueError("Angular quadrature order must be integer >= 8")
    gn, gw = _gauss(radial_order)
    rr = (gn + 1) * radius / 2
    radial_weights = gw * radius / 2 * rr
    theta = (np.arange(angular_order, dtype=np.float64) + 0.5) * (2 * math.pi / angular_order)
    xx = (rr[:, None] * np.cos(theta)[None, :]).ravel()
    yy = (rr[:, None] * np.sin(theta)[None, :]).ravel()
    area_weights = np.repeat(radial_weights, angular_order) * (2 * math.pi / angular_order)
    total = np.zeros((len(points), 3), dtype=np.float64)
    for sign in (1.0, -1.0):
        source = np.column_stack((xx, yy, np.full_like(xx, sign * height / 2)))
        delta = points[:, None, :] - source[None, :, :]
        norm_squared = np.sum(delta**2, axis=-1)
        if np.any(norm_squared == 0):
            raise ValueError("Singular magnetic-charge integration coordinate")
        total += sign * np.sum(
            delta * (area_weights[None, :] / norm_squared**1.5)[..., None], axis=1
        )
    return total * (1e6 / (4 * math.pi))


def _comparison(name: str, actual: Any, reference: Any, atol: float, rtol: float) -> dict:
    """Return a JSON-safe fixed-tolerance comparison; never relax tolerances."""
    a, b = np.asarray(actual, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    result = {"name": name, "atol_uT_per_T": atol, "rtol": rtol, "passed": False}
    if a.shape != b.shape:
        return {**result, "shape_actual": list(a.shape), "shape_reference": list(b.shape)}
    finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
    result.update(shape=list(a.shape), finite=finite)
    if not finite:
        return result
    error = np.abs(a - b)
    allowed = atol + rtol * np.abs(b)
    result.update(
        passed=bool(np.all(error <= allowed)),
        max_absolute_error_uT_per_T=float(np.max(error)),
        max_allowed_ratio=float(np.max(error / allowed)),
        failing_components=int(np.count_nonzero(error > allowed)),
    )
    return result


def _rotation_xyz(ax: float, ay: float, az: float) -> np.ndarray:
    """Create a proper local-to-world Rz@Ry@Rx rotation from radians (fixtures)."""
    cx, sx, cy, sy, cz, sz = math.cos(ax), math.sin(ax), math.cos(ay), math.sin(ay), math.cos(az), math.sin(az)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def validate(original_module_path: str | Path) -> dict:
    """Explicit remote-only numerical qualification; return every gate and passed.

    No training, FEM, contact model, data archive or hardware is accessed.
    ``original_module_path`` must point to the uploaded pinned original
    finite_cylinder_field_r393.py.  Its generic CPU torch operators are an
    implementation cross-check; the independent disk-charge integral provides
    a second physical representation.  FP64, cross-kernel ATOL=1e-3 µT/T /
    RTOL=1e-6, internal ATOL=1e-4 µT/T / RTOL=1e-8 are fixed.  Any failed gate
    remains failed.  This qualification does not validate actual magnets,
    silicone deformation, sensor coverage or physical localization accuracy.
    """
    gates: list[dict] = []
    result: dict = {
        "schema": "cylindrical-v03-magnetic-kernel-validation-v1",
        "dtype": "float64",
        "length_unit": "mm",
        "field_unit": "uT_per_J_T",
        "polarization_definition": "J=mu0*M in tesla; uniform axial magnetization",
        "cross_atol_uT_per_T": CROSS_ATOL_UT_PER_T,
        "cross_rtol": CROSS_RTOL,
        "internal_atol_uT_per_T": INTERNAL_ATOL_UT_PER_T,
        "internal_rtol": INTERNAL_RTOL,
        "gates": gates,
        "passed": False,
        "physical_deformation_or_device_validation": False,
    }

    def compare(name: str, actual: Any, reference: Any, *, cross: bool = False) -> None:
        gates.append(_comparison(
            name, actual, reference,
            CROSS_ATOL_UT_PER_T if cross else INTERNAL_ATOL_UT_PER_T,
            CROSS_RTOL if cross else INTERNAL_RTOL,
        ))

    radius, height = 2.5, 1.0
    try:
        axis_z = np.array([-8.0, -4.0, -1.5, 1.5, 4.0, 8.0], dtype=np.float64)
        axis_points = np.column_stack((np.zeros_like(axis_z), np.zeros_like(axis_z), axis_z))
        zp, zm = axis_z + height / 2, axis_z - height / 2
        analytical = np.zeros_like(axis_points)
        analytical[:, 2] = 0.5e6 * (
            zp / np.sqrt(zp**2 + radius**2) - zm / np.sqrt(zm**2 + radius**2)
        )
        compare("axis_analytical", cylinder_local_uT_np(axis_points, radius, height, 24), analytical)
    except Exception as exc:
        gates.append({"name": "axis_analytical", "passed": False, "error": f"{type(exc).__name__}: {exc}"})

    probe_points = np.array([
        [0.0, 0.0, 4.0], [1.0, 2.0, 5.0], [-5.0, 1.0, 3.0],
        [4.0, 0.0, -3.0], [3.8, 1.2, 1.8], [1.0, -0.4, -2.5],
    ], dtype=np.float64)
    try:
        fields = {order: cylinder_local_uT_np(probe_points, radius, height, order) for order in (24, 48, 96)}
        compare("axial_order_24_vs_48", fields[24], fields[48])
        compare("axial_order_48_vs_96", fields[48], fields[96])
        disk_coarse = _disk_charge_uT_per_T(probe_points, radius, height, 64, 128)
        disk_fine = _disk_charge_uT_per_T(probe_points, radius, height, 96, 256)
        compare("independent_disk_order_convergence", disk_coarse, disk_fine)
        compare("elliptic_vs_independent_two_disk_integral", fields[96], disk_fine, cross=True)
        result["independent_kernel_points_local_mm"] = probe_points.tolist()
    except Exception as exc:
        gates.append({"name": "quadrature_and_independent_disk", "passed": False, "error": f"{type(exc).__name__}: {exc}"})

    centers = np.array([[0.0, 0.0, 0.0], [8.0, -3.0, 1.0], [-4.0, 6.0, -2.0]], dtype=np.float64)
    rotations = np.stack([
        _rotation_xyz(0.0, 0.0, 0.0),
        _rotation_xyz(0.2, -0.3, 0.4),
        _rotation_xyz(-0.4, 0.15, -0.2),
    ])
    sensors = np.array([[1.0, -1.0, 7.0], [12.0, 6.0, 5.0], [-10.0, -6.0, 4.0], [2.0, 11.0, 9.0]], dtype=np.float64)
    try:
        native = contributions(centers, rotations, sensors, order=48)
        shift = np.array([13.0, -9.0, 4.0], dtype=np.float64)
        compare("common_translation_covariance", contributions(centers + shift, rotations, sensors + shift, order=48), native)
        q = _rotation_xyz(0.7, -0.45, 0.25)
        rotated = contributions(centers @ q.T, q[None, :, :] @ rotations, sensors @ q.T, order=48)
        compare("common_rigid_rotation_covariance", rotated, native @ q.T)
        # Polarity is an explicitly signed scalar multiplier, never a pose change.
        signed_j = np.array([0.7, -1.1, 0.3], dtype=np.float64)
        zero = np.zeros((len(sensors), 3), dtype=np.float64)
        positive = np.einsum("m,msi->si", signed_j, native)
        compare("polarity_reversal_and_zero", np.einsum("m,msi->si", -signed_j, native), -positive)
        compare("zero_polarization", np.einsum("m,msi->si", np.zeros(3, dtype=np.float64), native), zero)
    except Exception as exc:
        gates.append({"name": "covariance_and_signed_polarization", "passed": False, "error": f"{type(exc).__name__}: {exc}"})

    try:
        original_path = Path(original_module_path).resolve(strict=True)
        original_bytes = original_path.read_bytes()
        digest = hashlib.sha256(original_bytes).hexdigest()
        result["original_module_path"] = str(original_path)
        result["original_module_sha256"] = digest
        module_name = f"_cylindrical_original_finite_cylinder_{digest[:16]}"
        spec = importlib.util.spec_from_file_location(module_name, original_path)
        if spec is None or spec.loader is None:
            raise ImportError("Cannot load pinned original magnetic module")
        original = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = original
        spec.loader.exec_module(original)
        import torch

        result["numpy_version"] = np.__version__
        import scipy
        result["scipy_version"] = scipy.__version__
        result["torch_version"] = torch.__version__

        def tensor(x: Any):
            return torch.as_tensor(np.asarray(x, dtype=np.float64).copy(), dtype=torch.float64, device="cpu")

        with torch.no_grad():
            torch_local = original.cylinder_local_uT(tensor(probe_points), radius, height, 48).cpu().numpy()
            compare("numpy_scipy_vs_original_torch_local", cylinder_local_uT_np(probe_points, radius, height, 48), torch_local, cross=True)
            reference_parts = []
            for index in range(len(centers)):
                one = original.fields_external_matrices(
                    tensor(centers[index:index + 1]), tensor(rotations[index:index + 1]),
                    tensor(sensors), tensor([1.0]), radius, height, 48,
                )
                reference_parts.append(one.cpu().numpy())
            reference_parts_np = np.stack(reference_parts)
            actual_parts = contributions(centers, rotations, sensors, order=48)
            compare("numpy_vs_original_torch_individual_world_contributions", actual_parts, reference_parts_np, cross=True)
            for scale in (1.0, -1.0, 2.3, 0.0):
                weights = np.array([0.7, -1.1, 0.3], dtype=np.float64) * scale
                reference_sum = original.fields_external_matrices(
                    tensor(centers), tensor(rotations), tensor(sensors), tensor(weights), radius, height, 48,
                ).cpu().numpy()
                compare(
                    f"signed_linear_combination_vs_original_torch_scale_{scale}",
                    np.einsum("m,msi->si", weights, actual_parts), reference_sum, cross=True,
                )
    except Exception as exc:
        gates.append({"name": "original_torch_cross_kernel", "passed": False, "error": f"{type(exc).__name__}: {exc}"})

    result["passed"] = bool(gates) and all(bool(gate["passed"]) for gate in gates)
    result["gate_count"] = len(gates)
    result["failed_gate_names"] = [gate["name"] for gate in gates if not gate["passed"]]
    return result

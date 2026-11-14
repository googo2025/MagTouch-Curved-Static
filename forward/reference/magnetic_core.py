"""Independent numerical revision; old operator remains immutable.

Units mm and microtesla per J=1 T. Maxwell vacuum axial expansion is used
only for D5x1 mm cylinders, rho <= .0025 mm and |z| >= 1 mm. The unchanged
elliptic operator supplies all other exterior coordinates. This deliberately
limited numerical branch does not model silicone or actual magnetic grade.
"""
import numpy as np
import legacy_core as old


def branch_mask(points, radius=2.5, height=1.):
    return ((np.linalg.norm(points[..., :2], axis=-1) <= radius * 1e-3)
            & (np.abs(points[..., 2]) >= height)
            & (radius == 2.5) & (height == 1.))


def cylinder_local_uT_np(local_mm, radius_mm=2.5, height_mm=1., order=24):
    points = old._fp64(local_mm, 'local_mm')
    if points.ndim < 1 or points.shape[-1] != 3:
        raise ValueError('Expected (...,3)')
    a, height = old._dimensions(radius_mm, height_mm)
    old._exterior(points, a, height)
    old._gauss(order)  # Retain original order validation even on the axis.
    shape = points.shape
    p = points.reshape(-1, 3)
    use = branch_mask(p, a, height)
    result = np.empty_like(p)
    if np.any(~use):
        result[~use] = old.cylinder_local_uT_np(p[~use], a, height, order)
    if np.any(use):
        q = p[use]
        up, um = q[:, 2] + height/2, q[:, 2] - height/2
        dp, dm = up*up+a*a, um*um+a*a
        b0 = .5*(up/np.sqrt(dp)-um/np.sqrt(dm))
        d1 = .5*a*a*(dp**-1.5-dm**-1.5)
        d2 = -1.5*a*a*(up*dp**-2.5-um*dm**-2.5)
        d3 = -1.5*a*a*((a*a-4*up*up)*dp**-3.5-(a*a-4*um*um)*dm**-3.5)
        d4 = 7.5*a*a*(up*(3*a*a-4*up*up)*dp**-4.5-um*(3*a*a-4*um*um)*dm**-4.5)
        r2 = np.sum(q[:, :2]**2, axis=1)
        bxy = q[:, :2]*(-.5*d1+r2*d3/16)[:, None]
        bz = b0-r2*d2/4+r2*r2*d4/64
        result[use] = np.column_stack((bxy, bz))*1e6
    if not np.isfinite(result).all():
        raise FloatingPointError('Nonfinite revised magnetic output')
    return result.reshape(shape)


def contributions(centers, rotations, sensors, order=24, *, radius_mm=2.5, height_mm=1.):
    centers = old._fp64(centers, 'centers')
    rotations = old._fp64(rotations, 'rotations')
    sensors = old._fp64(sensors, 'sensors')
    if centers.ndim != 2 or centers.shape[-1] != 3 or len(centers) == 0:
        raise ValueError('centers shape')
    if rotations.shape != (len(centers), 3, 3) or sensors.ndim != 2 or sensors.shape[-1] != 3 or len(sensors) == 0:
        raise ValueError('rotation/sensor shape')
    gram = np.swapaxes(rotations, -1, -2) @ rotations
    if np.max(np.abs(gram-np.eye(3))) > 2e-5 or np.max(np.abs(np.linalg.det(rotations)-1)) > 2e-5:
        raise ValueError('Proper rigid rotations required')
    local = np.einsum('msi,mij->msj', sensors[None]-centers[:, None], rotations)
    return np.einsum('msj,mij->msi', cylinder_local_uT_np(local, radius_mm, height_mm, order), rotations)

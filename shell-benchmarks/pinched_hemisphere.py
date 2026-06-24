# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: 'defaultInterpreterPath: 3.12.3.final.0'
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Pinched Hemispherical Shell Benchmark (18° hole)
#
# A hemispherical shell with an **18° hole at the top**, free along both edges (hole and
# equator), loaded by **two inward and two outward radial point forces 90° apart** on the
# equator. It is the classic test of an element's **inextensional bending** behaviour and its
# ability to represent rigid-body motion, and — because the loading is nearly inextensional —
# a sharp test of **membrane locking**. The 18° hole removes the pole, so the mid-surface is a
# smooth spherical *zone* with no degenerate apex.
#
# The target quantity is the **radial displacement at a loaded point**, normalised by the
# reference value. To probe membrane locking the problem is run at three slendernesses
# $R/t=250,\,2500,\,25000$; with the load scaled as $P\propto t^{3}$ the reference displacement
# is **thickness-independent**, so locking shows directly as a drop below 1 that worsens as the
# shell thins (and is much milder for the mixed element).
#
# **Reference**
#
# J.C. Simo, D.D. Fox and M.S. Rifai, *"On a stress resultant geometrically exact shell model.
# Part II: the linear theory; computational aspects"*, Comput. Methods Appl. Mech. Engrg., 1989,
# **73**(1), pp. 53-92. — Section 7.5.1, Table 12. For the standard $t=0.04$ ($R/t=250$) case
# Simo normalises by **0.093** (an asymptotic value); MacNeal & Harder (1985) published 0.094.
#
# The displacement is *not* exactly thickness-independent (the point-load response carries
# local indentation / transverse-shear terms that don't scale as $t^3$), so each slenderness has
# its own reference. The geometry, the $t^{3}$ load scaling ($P=31250\,t^{3}$; $R=10$,
# $E=6.825\times10^{7}$, $\nu=0.3$; $R/t=250,2500,25000$), and the per-slenderness reference
# values $|u_A| = 0.093521,\,0.091594,\,0.090817$ are taken from
#
# H. Casquero and K.D. Mathews, *"Overcoming membrane locking in quadratic NURBS-based
# discretizations of linear Kirchhoff-Love shells: CAS elements"*, arXiv:2311.00101 (2023),
# Section 5.2 — where the reference displacements are an **overkill solution** (2562 classical
# shell elements of degree 9), not an analytical value. The $R/t=250$ value $0.093521$ is within
# 0.6% of Simo's 0.093 / 0.4% of MacNeal-Harder's 0.094.

# %%
import csv
import os
import time

import numpy as np
import scipy.sparse as sp
import pyck as ck
from pyck.basis import NURBS

HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
OUT_DIR = os.path.join(HERE, "pinched_hemisphere")
os.makedirs(OUT_DIR, exist_ok=True)

# %% [markdown]
# ## Problem setup
#
# Hemisphere of radius $R=10$ with an $18°$ hole at the pole. One quadrant is modelled with
# symmetry on the two cut meridians; the hole and equator edges are free. Two point loads sit on
# the equator at the symmetry planes: an **outward** force along $+X$ at $(R,0,0)$ and an
# **inward** force along $-Y$ at $(0,R,0)$. The quadrant load is $P=(t/t_\mathrm{ref})^{3}$ with
# $t_\mathrm{ref}=0.04$ (half of the full pair, since each load lies on a symmetry plane), so the
# normalised displacement is thickness-independent.
#
# $$
# \begin{aligned}
# \text{material:}\quad & E = 6.825\times10^{7},\quad \nu = 0.3,\quad R = 10 \\[2pt]
# \text{slenderness:}\quad & R/t = 250,\,2500,\,25000,\qquad \text{hole half-angle } 18°
# \end{aligned}
# $$

# %%
R    = 10.0            # radius
E    = 6.825e7         # Young's modulus
NU   = 0.3
T_REF = 0.04           # reference thickness (R/t = 250); quadrant load is 1.0 here
HOLE_DEG = 18.0        # hole half-angle (colatitude of the hole edge)
PEN_SCALE = 1.0e6      # penalty scale (× E t / E t^3); thickness-scaled to stay well-conditioned

# Per-slenderness reference |u_A| — Casquero & Mathews, arXiv:2311.00101 (2023) Sec. 5.2,
# overkill solution (2562 CS elements, degree 9). R/t=250 value is within 0.6% of Simo's 0.093.
REFERENCE_U = {250: 0.093521, 2500: 0.091594, 25000: 0.090817}


def quadrant_load(t: float) -> float:
    """Quadrant point-load magnitude P = (t/t_ref)^3 (full pair P = 2 (t/t_ref)^3 = 31250 t^3),
    scaled so the normalised loaded-point displacement is thickness-independent."""
    return (t / T_REF) ** 3


# %% [markdown]
# ## Geometry — exact NURBS spherical zone
#
# The quadrant mid-surface is the spherical zone $\varphi\in[18°,90°]$ swept $90°$ about $Z$.
# It is an **exact** rational (NURBS) surface: a tensor product of a rational-quadratic meridian
# arc (the $72°$ zone, weight $\cos 36°$) and a rational-quadratic $90°$ revolution arc (weight
# $1/\sqrt2$). Parametrisation: $u=0$ is the (free) hole edge, $u=1$ the (free) equator; $v=0$
# the $y=0$ symmetry meridian, $v=1$ the $x=0$ symmetry meridian. The two equator corners
# $(u{=}1,v{=}0)=(R,0,0)$ and $(u{=}1,v{=}1)=(0,R,0)$ are the loaded points.

# %%
def zone_patch(nu: int, nv: int, deg: int, name: str = "hemi") -> ck.SurfacePatch:
    """Exact NURBS spherical zone (hole half-angle ``HOLE_DEG`` to the equator) over one quadrant.

    ``nu``/``nv`` are the basis counts (>= deg+1); the base rational-quadratic patch is degree
    elevated to ``deg`` and knot-refined to ``nu``/``nv`` uniform elements per direction.
    """
    phi1, phi2 = np.radians(HOLE_DEG), np.radians(90.0)
    am, phim = 0.5 * (phi2 - phi1), 0.5 * (phi1 + phi2)        # meridian half-angle, bisector
    rho = [R * np.sin(phi1), (R / np.cos(am)) * np.sin(phim), R * np.sin(phi2)]   # dist. from Z
    zz  = [R * np.cos(phi1), (R / np.cos(am)) * np.cos(phim), R * np.cos(phi2)]   # height
    w_u = [1.0, np.cos(am), 1.0]                               # meridian arc weights
    cxy = [(1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]                 # 90° revolution control polygon
    w_v = [1.0, 1.0 / np.sqrt(2.0), 1.0]

    cps = [[rho[i] * cx, rho[i] * cy, zz[i]]                   # u-fastest: g = j*3 + i
           for (cx, cy) in cxy for i in range(3)]
    patch = ck.SurfacePatch(NURBS.clamped_uniform(2, 3, w_u),
                            NURBS.clamped_uniform(2, 3, w_v), np.array(cps, float), name=name)
    if deg > 2:
        patch = patch.elevate_degree(0, deg - 2).elevate_degree(1, deg - 2)
    for u in np.arange(1, nu - deg) / (nu - deg):
        patch = patch.insert_knot(0, float(u))
    for v in np.arange(1, nv - deg) / (nv - deg):
        patch = patch.insert_knot(1, float(v))
    return patch


# %% [markdown]
# ## Boundary conditions and loads
#
# Symmetry on the two cut meridians ($v=0$: $U_Y=\theta_n=0$; $v=1$: $U_X=\theta_n=0$); the hole
# and equator edges are free. The only unconstrained rigid mode left is the **vertical $U_Z$
# translation** (the horizontal loads and the two symmetry planes pin everything else), removed
# by a single $U_Z$ pin at a load corner. The point loads land exactly on the two equator corner
# control points (interpolatory), so the consistent nodal force is just $\pm P$ on those DOFs.
# The constant-$\psi$ null mode of the rotation-free 4p family is pinned at one control point.

# %%
def solve_pinched(
    deg: int, nel: int, t: float,
    element_cls: type = ck.ShellReissnerMindlinHier4p,
    assumed_strain: bool = False,
):
    """Solve the quadrant pinched hemisphere and return ``(u_load, ndof, t_asm, t_solve)``.

    ``u_load`` is the radial displacement $U_X$ at the loaded point $(R,0,0)$ (compare with
    ``REFERENCE_U[R/t]``). The point load is ``quadrant_load(t)`` (scaled $\\propto t^3$). With
    ``assumed_strain=True`` the shell is wrapped in a ``MixedMembraneStrainShell`` (relieves
    membrane locking; far more accurate as the shell thins).

    Penalties are **thickness-scaled** (``PEN_SCALE * E * t`` for displacement, ``* E * t^3``
    for rotation): absolute penalties over-stiffen the bending-dominated thin shells by orders
    of magnitude and destroy the solution at high ``R/t``.
    """
    n = nel + deg
    patch = zone_patch(n, n, deg)
    base = element_cls(ck.PlaneStress2d(E, NU, t))
    gauss2 = ck.GaussLegendre(deg + 1, dim=2)
    gauss1 = ck.GaussLegendre(deg + 1, dim=1)
    element = ck.MixedMembraneStrainShell(patch, base, gauss2) if assumed_strain else base

    # Symmetry on the two cut meridians (penalty; full-size system so loads/pins add directly).
    A_DISP, A_ROT = PEN_SCALE * E * t, PEN_SCALE * E * t ** 3
    prob = ck.LinearElasticProblem([patch], element, gauss2)
    c = ck.PenaltyBoundaryCondition(patch.boundary(1, True), gauss1)    # v=0: y=0 plane
    c.add(ck.Field.U_Y, A_DISP).add(ck.Field.ROT_N, A_ROT)
    prob.add_condition(c, patch="hemi")
    c = ck.PenaltyBoundaryCondition(patch.boundary(1, False), gauss1)   # v=1: x=0 plane
    c.add(ck.Field.U_X, A_DISP).add(ck.Field.ROT_N, A_ROT)
    prob.add_condition(c, patch="hemi")

    t0 = time.perf_counter()
    K, f = prob.assemble()
    K = sp.csr_matrix(K).tolil()
    f = np.asarray(f, dtype=np.float64).copy()
    t_asm = time.perf_counter() - t0

    nd = base.num_node_dofs
    g_x = n - 1                       # corner (u=1, v=0) = (R,0,0): +X outward load
    g_y = (n - 1) * n + (n - 1)       # corner (u=1, v=1) = (0,R,0): -Y inward load
    p = quadrant_load(t)
    f[g_x * nd + 0] += p
    f[g_y * nd + 1] -= p

    pen = PEN_SCALE * abs(K.diagonal()).max()
    K[g_x * nd + 2, g_x * nd + 2] += pen     # pin U_Z (vertical rigid-body mode)
    K[0 * nd + 3, 0 * nd + 3] += pen         # pin one psi (constant null mode of the 4p family)

    t0 = time.perf_counter()
    u_full = np.asarray(ck.solve(K.tocsr(), f, full=True))
    t_solve = time.perf_counter() - t0

    disp = ck.Function(u_full[:patch.num_control_pts * nd], element, patch,
                       ck.FieldType.DISPLACEMENT)
    u_load = float(np.asarray(disp([[1.0, 0.0]])).reshape(3)[0])    # U_X at (R,0,0)
    ndof = patch.num_control_pts * nd
    return u_load, ndof, t_asm, t_solve


# %% [markdown]
# ## Studies

# %%
def save_rows(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {path}")


# %% [markdown]
# ## Study 1 — Membrane-locking sweep: loaded-point displacement vs slenderness
#
# Uniform mesh ($p=4$), sweep elements per side for the three slendernesses $R/t=250,2500,25000$
# (load $\propto t^3$, so the normalised displacement is thickness-independent). Two formulations:
# the displacement **RM-Hier-4p** and its mixed variant **RM-Hier-4p-MMS** (assumed membrane
# strain). Membrane locking shows as a drop below 1 that deepens as the shell thins; the mixed
# element resists it far better.

# %%
nel_sweep = (2, 3, 4, 6, 8, 12, 16, 24, 32, 48)
deg = 4
FORMULATIONS = (
    ("ShellReissnerMindlinHier4p",    ck.ShellReissnerMindlinHier4p, False),
    ("ShellReissnerMindlinHier4pMMS", ck.ShellReissnerMindlinHier4p, True),
)

rows = []
print(f"\nStudy 1 - Pinched hemisphere locking sweep (p={deg})")
for ratio in (250, 2500, 25000):
    t = R / ratio
    u_ref = REFERENCE_U[ratio]
    print(f"\n######## R/t = {ratio}   (t = {t:g}, P_quad = {quadrant_load(t):.4g}, "
          f"u* = {u_ref}) ########")
    for ename, ecls, use_mms in FORMULATIONS:
        print(f"  -- {ename}")
        print(f"  {'nel':>5} {'ndof':>8} {'u_load':>12} {'u/u*':>9}")
        for nel in nel_sweep:
            try:
                u_load, ndof, t_asm, t_sol = solve_pinched(deg, nel, t, ecls, use_mms)
            except Exception as exc:
                print(f"  {nel:>5}   skipped ({type(exc).__name__})", flush=True)
                continue
            u_ratio = u_load / u_ref
            print(f"  {nel:>5} {ndof:>8} {u_load:>12.6f} {u_ratio:>9.5f}", flush=True)
            rows.append({
                "element": ename, "ratio": ratio, "t": t, "deg": deg, "nel": nel, "ndof": ndof,
                "u_load": u_load, "u_ref": u_ref, "u_ratio": u_ratio,
            })

save_rows(rows, os.path.join(OUT_DIR, "pinched_hemisphere_results.csv"))

# %% [markdown]
# <img src="pinched_hemisphere/pinched_hemisphere_consistency.pdf" width="520" align="center" alt="Pinched hemisphere with 18-degree hole: normalized loaded-point radial displacement u^h/u^ref vs dofs on a uniform mesh, three slendernesses R/t=250,2500,25000 (colour) for RM-Hier-4p (displacement, dashed) and RM-Hier-4p-MMS (mixed strain, solid); membrane locking deepens as the shell thins, and the mixed element resists it far better than the displacement element.">

# %% [markdown]
# ## ParaView export
#
# Solve the pinched hemisphere at the reference thickness ($R/t=250$, $p=4$, 12 elements per
# side) and write the quadrant to a Bézier `.vtu`. The zone has no degenerate pole, so all
# fields export cleanly. Open in ParaView, warp by `displacement`.

# %%
deg_v, nel_v, t_v = 4, 6, T_REF
n_v = nel_v + deg_v
patch_v  = zone_patch(n_v, n_v, deg_v)
elem_v   = ck.ShellReissnerMindlinHier4p(ck.PlaneStress2d(E, NU, t_v))
gauss2_v = ck.GaussLegendre(deg_v + 1, dim=2)
gauss1_v = ck.GaussLegendre(deg_v + 1, dim=1)

a_disp_v, a_rot_v = PEN_SCALE * E * t_v, PEN_SCALE * E * t_v ** 3
prob_v = ck.LinearElasticProblem([patch_v], elem_v, gauss2_v)
c = ck.PenaltyBoundaryCondition(patch_v.boundary(1, True), gauss1_v)
c.add(ck.Field.U_Y, a_disp_v).add(ck.Field.ROT_N, a_rot_v)
prob_v.add_condition(c, patch="hemi")
c = ck.PenaltyBoundaryCondition(patch_v.boundary(1, False), gauss1_v)
c.add(ck.Field.U_X, a_disp_v).add(ck.Field.ROT_N, a_rot_v)
prob_v.add_condition(c, patch="hemi")

K_v, f_v = prob_v.assemble()
K_v = sp.csr_matrix(K_v).tolil(); f_v = np.asarray(f_v, dtype=np.float64).copy()
nd_v = elem_v.num_node_dofs
gx_v, gy_v = n_v - 1, (n_v - 1) * n_v + (n_v - 1)
p_v = quadrant_load(t_v)
f_v[gx_v * nd_v + 0] += p_v
f_v[gy_v * nd_v + 1] -= p_v
pen_v = PEN_SCALE * abs(K_v.diagonal()).max()
K_v[gx_v * nd_v + 2, gx_v * nd_v + 2] += pen_v
K_v[0 * nd_v + 3, 0 * nd_v + 3] += pen_v

u_v = np.asarray(ck.solve(K_v.tocsr(), f_v, full=True))[:patch_v.num_control_pts * nd_v]

# Displacement from control-point coefficients (point_data); moments sampled (functions=).
arr = u_v.reshape(-1, nd_v)
moment_v = ck.Function(u_v, elem_v, patch_v, ck.FieldType.MOMENT)
fields = {"m11": moment_v[0], "m22": moment_v[1], "m12": moment_v[2]}
out = os.path.join(OUT_DIR, "pinched_hemisphere.vtu")
with ck.BezierVtuWriter(out) as writer:
    writer.add(patch_v, point_data={"displacement": arr[:, :3], "psi": arr[:, 3:4]},
               functions=fields)
print(f"Wrote {out}")

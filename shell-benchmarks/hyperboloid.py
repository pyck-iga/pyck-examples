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
# # Hyperboloid-of-Revolution Benchmark
#
# A single-sheet hyperboloid of revolution loaded by a self-equilibrated normal
# pressure with a $\cos 2\varphi$ circumferential variation — a demanding test of a
# shell element on a doubly-curved surface (nonzero Gaussian curvature). The top circular
# edge is **clamped**, giving a **membrane-dominated** response with a boundary layer along
# that edge — the hard case for thin-shell convergence.
#
# The target quantity is the **strain energy** of the (one-eighth) model, reported
# against the converged references for three slendernesses $t=1/100,\,1/1000,\,1/10000$.
#
# **References**
#
# 1. J.-F. Hiller and K.-J. Bathe, *"Measuring convergence of mixed finite element
#    discretizations: an application to shell structures"*, Computers & Structures,
#    2003, **81**(8-11), pp. 639-654. — the original hyperboloid shell benchmark.
# 2. P. Krysl, *"Benchmarking Computational Shell Models"*, Archives of
#    Computational Methods in Engineering, 2023, **30**(1), pp. 301-315.
#    — compiles the reference strain energies (Tables 19/20) used here.

# %%
import csv
import os
import time

import numpy as np
import pyck as ck

HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
OUT_DIR = os.path.join(HERE, "hyperboloid")
os.makedirs(OUT_DIR, exist_ok=True)

# %% [markdown]
# ## Problem setup
#
# The mid-surface is the single-sheet hyperboloid of revolution
# $$ x^2 + z^2 = 1 + y^2, \qquad -1 \le y \le 1 $$
# (axis along $Y$, throat radius $1$), loaded by a normal pressure
# $p = p_0\cos 2\varphi$ with $\varphi=\operatorname{atan2}(z,x)$, applied along the
# outward unit normal $\hat n \propto (x,-y,z)$. The load is self-equilibrated and the
# geometry triply symmetric, so **one eighth** is modelled: $\varphi\in[0,\tfrac\pi2]$,
# $y\in[0,1]$, with symmetry planes $z=0$, $x=0$ and the throat $y=0$.
#
# $$
# \begin{aligned}
# \text{material:}\quad & E = 2\times10^{11}, \quad \nu = \tfrac13 \\[4pt]
# \text{load:}\quad & p_0 = 1\ \text{MPa}, \quad p = p_0\cos 2\varphi
# \end{aligned}
# $$
#
# **Note on $\nu$:** the benchmark uses $\nu=\tfrac13$ — the common typo $\nu=0.3$ shifts the
# energy and breaks the comparison against the reference values, so keep $\nu=\tfrac13$.
#
# <img src="hyperboloid/hyperboloid_geom.svg" width="760" align="center" alt="Hyperboloid of revolution: one-eighth model, symmetry planes, and the cos 2phi pressure.">


# %%
E = 2.0e11
NU = 1.0 / 3.0
P0 = 1.0e6

LAYER_C = 6.0                            # boundary-layer band width along the clamped edge (~ c*sqrt(t))
C_NIT = 300.0                            # Nitsche stabilisation constant (KL element only)


def _meridian_param_at_y(y_target: float) -> float:
    """Parameter v in [0,1] on the meridian (rational quadratic) where the axial
    coordinate equals `y_target`. Bisection on the closed-form y(v) (monotone in v)."""
    s2 = np.sqrt(2.0)
    wt = (1.0, np.sqrt(2.0 + 2.0 * s2) / 2.0, 1.0)       # control weights
    yc = (0.0, s2 - 1.0, 1.0)                            # control y-coordinates

    def y_of(v):
        b = (wt[0] * (1 - v) ** 2, wt[1] * 2 * v * (1 - v), wt[2] * v * v)
        return (b[0] * yc[0] + b[1] * yc[1] + b[2] * yc[2]) / sum(b)

    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if y_of(mid) < y_target else (lo, mid)
    return 0.5 * (lo + hi)


def hyperboloid_octant(
    deg: int, nel: int, layer_width: float = 0.0,
) -> ck.SurfacePatch:
    """Exact NURBS octant of x^2 + z^2 = 1 + y^2, degree-elevated to `deg`.

    With ``layer_width > 0`` the meridian is a **classical two-region boundary-layer mesh**
    (Hiller-Bathe): ``nel//2`` uniform elements in the band ``y in [1 - layer_width, 1]``
    and ``nel//2`` uniform elements over the rest ``[0, 1 - layer_width]``. Otherwise the
    meridian is uniform.
    """
    s2 = np.sqrt(2.0)
    w1 = np.sqrt(2.0 + 2.0 * s2) / 2.0                   # meridian-hyperbola middle weight
    meridian = [(1.0, 0.0), (1.0, s2 - 1.0), (s2, 1.0)]  # (r, y), conic r^2 - y^2 = 1
    arc = [(1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]           # quarter circle (x, z), radius 1
    cps = np.array([[r * ax, y, r * az]
                    for (r, y) in meridian for (ax, az) in arc], dtype=float)
    bu = ck.NURBS.clamped_uniform(2, 3, [1.0, 1.0 / s2, 1.0])  # u = circumferential arc
    bv = ck.NURBS.clamped_uniform(2, 3, [1.0, w1, 1.0])        # v = meridian (y)
    patch = ck.SurfacePatch(bu, bv, cps, name="hyp")
    patch = patch.elevate_degree(0, deg - 2).elevate_degree(1, deg - 2)

    for k in range(1, nel):                              # circumferential: uniform
        patch = patch.insert_knot(0, k / nel)

    if layer_width > 0.0:                                # meridian: two-region layer mesh
        layer_width = min(layer_width, 0.5)              # cap: c*sqrt(t) exceeds the meridian for thick shells
        v_split = _meridian_param_at_y(1.0 - layer_width)
        m = nel // 2
        knots = [j * v_split / m for j in range(1, m + 1)]                 # [0, v_split]
        knots += [v_split + j * (1.0 - v_split) / m for j in range(1, m)]  # (v_split, 1)
        for v in knots:
            patch = patch.insert_knot(1, v)
    else:                                                # meridian: uniform
        for k in range(1, nel):
            patch = patch.insert_knot(1, k / nel)
    return patch


def pressure(P: np.ndarray) -> np.ndarray:
    """Normal pressure p0*cos(2 phi) along the outward unit normal — the self-equilibrated n=2
    load of the classic Hiller-Bathe benchmark; matches the Krysl REFERENCE table."""
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    phi = np.arctan2(z, x)
    nrm = np.sqrt(x * x + y * y + z * z)
    n_hat = np.column_stack([x / nrm, -y / nrm, z / nrm])
    return (P0 * np.cos(2.0 * phi))[:, None] * n_hat

# %% [markdown]
# ## Reference values
#
# Strain energies of the one-eighth clamped model (Hiller-Bathe via Krysl 2023, Tables 19/20):
#
# | $1/t$ | clamped $U^{\ast}$ [Nm] |
# |--:|--:|
# | $100$    | $0.539187\times10^{3}$ |
# | $1000$   | $0.600115\times10^{4}$ |
# | $10000$  | $0.618988\times10^{5}$ |

# %%
REFERENCE = {
    100:   0.539187e3,
    1000:  0.600115e4,
    10000: 0.618988e5,
}

# %% [markdown]
# ## Boundary conditions
#
# - **Symmetry planes** ($z=0$, $x=0$, throat $y=0$) — zero normal displacement +
#   zero bending rotation $\text{ROT}_N$.
# - **Top edge** $y=1$ — *clamped* ($U_X=U_Y=U_Z=0,\ \text{ROT}_N=\text{ROT}_S=0$).
#
# Imposed with **Lagrange multipliers** (constraints enforced exactly, no penalty weight to
# tune). The `ROT_N` trace is the recovered bending rotation, so the same conditions apply to
# every element. NOTE: this yields a symmetric **indefinite** saddle-point system.

# %%
def symmetry_planes(prob, patch, gauss1, element, t, nel) -> None:
    """Three symmetry planes: zero normal displacement + zero bending rotation. Lagrange (exact)
    for RM; Nitsche for the rotation-free KL element (ROT_N is a derived trace, so a Lagrange
    multiplier on it is inf-sup unstable)."""
    is_kl = isinstance(element, ck.ShellKirchhoffLove3p)
    w_u, w_rot = C_NIT * E * t * nel, C_NIT * E * t**3 * nel
    for boundary, normal_field in (
        (patch.boundary(0, True), ck.Field.U_Z),    # phi=0    -> z=0 plane
        (patch.boundary(0, False), ck.Field.U_X),   # phi=pi/2 -> x=0 plane
        (patch.boundary(1, True), ck.Field.U_Y),    # y=0 throat plane
    ):
        if is_kl:
            c = ck.NitscheBoundaryCondition(boundary, gauss1)
            c.add(normal_field, w_u).add(ck.Field.ROT_N, w_rot)
        else:
            c = ck.LagrangeBoundaryCondition(boundary, gauss1)
            c.add(normal_field)
            c.add(ck.Field.ROT_N)
        prob.add_condition(c, patch="hyp")


def clamp_top_edge(prob, patch, gauss1, element, t, nel) -> None:
    """Clamp the top edge y=1: all displacements + bending rotations. Lagrange (exact) for RM;
    Nitsche for the rotation-free KL element (see ``symmetry_planes``)."""
    boundary = patch.boundary(1, False)
    if isinstance(element, ck.ShellKirchhoffLove3p):
        w_u, w_rot = C_NIT * E * t * nel, C_NIT * E * t**3 * nel
        c = ck.NitscheBoundaryCondition(boundary, gauss1)
        c.add(ck.Field.U_X, w_u).add(ck.Field.U_Y, w_u).add(ck.Field.U_Z, w_u)
        c.add(ck.Field.ROT_N, w_rot).add(ck.Field.ROT_S, w_rot)
    else:
        c = ck.LagrangeBoundaryCondition(boundary, gauss1)
        c.add(ck.Field.U_X)
        c.add(ck.Field.U_Y)
        c.add(ck.Field.U_Z)
        c.add(ck.Field.ROT_N)
        c.add(ck.Field.ROT_S)
    # c = ck.PenaltyBoundaryCondition(boundary, gauss1)   # penalty alternative (tune the weights)
    prob.add_condition(c, patch="hyp")

# %% [markdown]
# ## Solver

# %%
def solve_hyperboloid(
    deg: int, nel: int, t: float,
    element_cls: type[ck.Element] = ck.ShellReissnerMindlinHier4p,
    layer_width: float = 0.0,
):
    """Solve the one-eighth clamped hyperboloid; return ``(U, disp, ndof, t_asm, t_solve)`` — the
    strain energy, the midsurface displacement field (a ``ck.Function``, for the L2 norm), the DOF
    count, and the assembly / linear-solve wall-clock times [s].

    ``layer_width`` > 0 selects the two-region boundary-layer mesh (see ``hyperboloid_octant``).
    """
    patch = hyperboloid_octant(deg, nel, layer_width)
    element = element_cls(ck.PlaneStress2d(E, NU, t))
    gauss2 = ck.GaussLegendre(deg + 1, dim=2)
    prob = ck.LinearElasticProblem([patch], element, gauss2)
    prob.add_domain_load(pressure)

    gauss1 = ck.GaussLegendre(deg + 1, dim=1)
    symmetry_planes(prob, patch, gauss1, element, t, nel)
    clamp_top_edge(prob, patch, gauss1, element, t, nel)

    # Pin the hierarchic constant-psi null mode (RM-Hier-4p only)
    if isinstance(element, ck.ShellReissnerMindlinHier4p):
        prob.add_constraint(ck.DirectConstraint([3], value=0.0))

    t0 = time.perf_counter()
    K, f = prob.assemble()
    t_asm = time.perf_counter() - t0

    t0 = time.perf_counter()
    u = ck.solve(K, f, physical_dofs=prob.num_physical_dofs)
    t_solve = time.perf_counter() - t0

    K_energy, _ = ck.LinearElasticProblem([patch], element, gauss2).assemble()
    U = 0.5 * float(u @ (K_energy @ u))
    disp = ck.Function(u, element, patch, ck.FieldType.DISPLACEMENT)  # midsurface displacement field
    ndof = patch.num_control_pts * element.num_node_dofs
    return U, disp, ndof, t_asm, t_solve


# %% [markdown]
# ## Studies
#
# Two studies follow (both with **Lagrange-multiplier** BCs):
# - **Study 1** — normalized strain energy $U/U^{\ast}$ on the **uniform** mesh.
# - **Study 2** — energy convergence upon **polynomial refinement** ($p=3,4,5$): relative
#   energy error $|U/U^{\ast}-1|$ (log-log) on the **graded** mesh, self-converged per degree;
#   the slope steepens with $p$ (reduced rate $O(h^{2(p-1)})$).

# %%
def save_rows(rows, path):
    """Write study result rows to `path` as CSV (columns from the first row's keys)."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {path}")


# %% [markdown]
# ## Study 1 — Energy Consistency Study on Uniform mesh
#
# Per thickness, sweep elements per side on a **uniform** mesh ($p=3$), starting from the coarsest
# single-element mesh (`nel=1`, $N=\deg+1$), with the hierarchic **RM-Hier-5p** (independent
# transverse shear, locking-free by construction). The membrane-dominated clamped case converges
# ever slower as the shell thins — its boundary layer is unresolved on a uniform mesh, so the energy
# approaches the reference from below (over-stiff, negative err %) and the gap widens as $t\to0$.

# %%
nel_sweep = (1, 3, 5, 7, 9, 12, 16, 20, 24, 32, 64, 96)
deg = 3

uniform_rows = []
print(f"\nStudy 1 - Uniform-mesh convergence (ShellReissnerMindlinHier4p, p={deg})")

for ratio in (100, 1000, 10000):
    t = 1.0 / ratio
    ref = REFERENCE[ratio]
    print(f"\n######## 1/t = {ratio}   (ref U* = {ref:.6e} Nm) ########")
    print(f"{'nel':>5} {'ndof':>8} {'energy [Nm]':>16} {'err %':>9} "
            f"{'asm[s]':>8} {'solve[s]':>9}")
    for nel in nel_sweep:
        try:
            U, _disp, ndof, t_asm, t_solve = solve_hyperboloid(deg, nel, t, ck.ShellKirchhoffLove3p)
        except Exception as exc:             # singular on the coarsest meshes
            print(f"{nel:>5}   skipped ({type(exc).__name__})", flush=True)
            continue
        err = 100.0 * (U - ref) / ref
        print(f"{nel:>5} {ndof:>8} {U:>16.6e} {err:>8.3f}% "
                f"{t_asm:>8.2f} {t_solve:>9.2f}", flush=True)
        uniform_rows.append({"element": "ShellReissnerMindlinHier5p",
                                "ratio": ratio, "t": t, "deg": deg,
                                "nel": nel, "ndof": ndof, "energy": U,
                                "energy_ref": ref, "energy_err_pct": err})

save_rows(uniform_rows, os.path.join(OUT_DIR, "results_uniform.csv"))

# %% [markdown]
# <img src="hyperboloid/clamped_convergence.svg" width="560" align="center" alt="Clamped hyperboloid: normalized strain energy vs elements per side (uniform mesh), one curve per thickness.">

# %% [markdown]
# ## Study 2 — Convergence upon polynomial refinement (graded mesh)
#
# For the thinnest case $1/t=10000$, on the boundary-layer mesh (band of width $c\,\sqrt{t}$ along
# the clamped edge $y=1$; even `nel>=2`), sweep the polynomial degree $p=3,4,5$ (RM-Hier-5p,
# Lagrange BCs). Each degree is measured against its **own over-refined solution** ($n_{el}=$
# `NEL_REF`, same degree and graded mesh), so the slope is a clean per-degree convergence rate.
# Two error measures are tracked: the **relative energy norm** $|U/U^{\ast}-1|$ and the
# **relative $L^2$ displacement error** $\|u-u^{\ast}\|_{L^2}/\|u^{\ast}\|_{L^2}$.
#
# The membrane-dominated clamped shell is the **discriminating** test. On this doubly-curved
# geometry the strain energy converges at the **reduced $O(h^{2(p-1)})$** rate (the membrane
# response + clamped-edge boundary layer cap it below the optimal $2p$), so the slope steepens
# with the polynomial order — $\approx 4,\,6,\,8$ for $p=3,4,5$ in the energy norm.

# %%
DEG_SWEEP = (3, 4, 5)
NEL_SWEEP_GRADED = (4, 6, 8, 12, 16, 24, 32, 40, 48)
NEL_REF = 96

graded_rows = []
print(f"\nStudy 2 - p-refinement convergence (energy + L2 displacement norms; "
      f"ShellReissnerMindlinHier4p, Lagrange); boundary-layer width = {LAYER_C}*sqrt(t)")
for ratio in (10000,):                          # thinnest case only
    t = 1.0 / ratio
    layer = LAYER_C * np.sqrt(t)
    for deg in DEG_SWEEP:
        ref_quad = ck.GaussLegendre(deg + 1, dim=2)   # integrate the L2 norm on the reference mesh
        # Self-convergence reference: same degree on an over-refined (nel=NEL_REF) graded mesh.
        try:
            U_ref, disp_ref, ndof_ref, _, _ = solve_hyperboloid(
                deg, NEL_REF, t, ck.ShellReissnerMindlinHier4p, layer_width=0)
        except Exception as exc:
            print(f"  1/t={ratio} p={deg}: reference solve failed ({type(exc).__name__}); skipped",
                  flush=True)
            continue
        disp_ref_l2 = np.sqrt(ck.inner_product(disp_ref, disp_ref, ref_quad))
        print(f"\n######## 1/t = {ratio}, p = {deg}   (U_ref = {U_ref:.6e} Nm "
              f"[self, nel={NEL_REF}], layer = {layer:.4g}) ########")
        print(f"{'nel':>5} {'ndof':>8} {'energy [Nm]':>16} {'U err %':>9} {'L2 err %':>10}")
        for nel in NEL_SWEEP_GRADED:
            if nel < 2 or nel % 2 != 0:           # graded needs an even nel >= 2
                continue
            try:
                U, disp_h, ndof, t_asm, t_solve = solve_hyperboloid(
                    deg, nel, t, ck.ShellReissnerMindlinHier4p, layer_width=layer)
            except Exception as exc:             # singular on the coarsest meshes
                print(f"{nel:>5}   skipped ({type(exc).__name__})", flush=True)
                continue
            err = 100.0 * (U - U_ref) / U_ref
            # Relative L2 displacement error vs the self-reference. The coarse and reference
            # patches share the exact same geometry (refinement is geometry-preserving), so
            # (disp_ref - disp_h) evaluated at common parametric points is the true pointwise
            # error; integrate it on the reference mesh.
            err_fn = disp_ref - disp_h
            l2_rel = np.sqrt(ck.inner_product(err_fn, err_fn, ref_quad)) / disp_ref_l2
            print(f"{nel:>5} {ndof:>8} {U:>16.6e} {err:>8.3f}% {100.0 * l2_rel:>9.3f}%", flush=True)
            graded_rows.append({"ratio": ratio, "t": t, "deg": deg,
                                "element": "ShellReissnerMindlinHier5p", "layer_width": layer,
                                "nel": nel, "ndof": ndof, "energy": U,
                                "energy_ref": U_ref, "energy_err_pct": err,
                                "disp_l2_rel": l2_rel})

save_rows(graded_rows, os.path.join(OUT_DIR, "hyperboloid_convergence_results.csv"))

# %% [markdown]
# Clamped hyperboloid ($1/t=10000$, RM-Hier-5p, polynomial refinement) — energy norm (left) and
# $L^2$ displacement norm (right) side by side; each curve carries a small reference-slope guide
# labelled inline as $\mathcal{O}(h^k)$ (energy: $k=4,6,8$; $L^2$: $k=4,5,6$):
#
# <img src="hyperboloid/clamped_rate_10000.svg" width="860" align="center" alt="Clamped hyperboloid 1/t=10000: relative energy error (left) and relative L2 displacement error (right) vs elements per side (graded mesh) for p=3,4,5, with inline reference slopes.">

# %% [markdown]
# ## ParaView export
#
# Solve the clamped shell ($t=1/1000$) on both a **uniform** and a **boundary-layer-graded**
# mesh and write each to an exact rational-Bézier `.vtu` (ParaView 5.9+). The hierarchic
# **RM-Hier-4p** element is used here (it carries the twist potential $\psi$), so the export
# can show $\psi$ and $\operatorname{curl}\psi$ as in the Scordelis notebook.
# Open in ParaView, **warp by `displacement`**, colour by e.g. `psi` or `curl_psi`.

# %%
deg, nel, t = 4, 16, 1.0 / 100
element = ck.ShellReissnerMindlinHier4p(ck.PlaneStress2d(E, NU, t))
layer = LAYER_C * np.sqrt(t)
gauss1 = ck.GaussLegendre(deg + 1, dim=1)

for tag, lw in (("uniform", 0.0), ("graded", layer)):
    patch = hyperboloid_octant(deg, nel, lw)
    prob = ck.LinearElasticProblem([patch], element, ck.GaussLegendre(deg + 1, dim=2))
    prob.add_domain_load(pressure)
    symmetry_planes(prob, patch, gauss1, element, t, nel)
    clamp_top_edge(prob, patch, gauss1, element, t, nel)
    prob.add_constraint(ck.DirectConstraint([3], value=0.0))   # pin the constant-psi null mode
    u = ck.solve(prob)

    disp     = ck.Function(u, element, patch, ck.FieldType.DISPLACEMENT)
    traction = ck.Function(u, element, patch, ck.FieldType.TRACTION)    # [n11, n22, n12, q1, q2]
    moment   = ck.Function(u, element, patch, ck.FieldType.MOMENT)      # [m11, m22, m12]
    primal   = ck.Function(u, element, patch, ck.FieldType.PRIMAL)      # [v_x, v_y, v_z, psi]
    rotation = ck.Function(u, element, patch, ck.FieldType.ROTATION)    # [rot1, rot2]

    # curl(psi) alone: zero displacement DOFs, keep only psi slot (3)
    u_psi = np.asarray(u).reshape(-1, 4)
    u_psi = np.column_stack([np.zeros((u_psi.shape[0], 3)), u_psi[:, 3]]).ravel()
    curl_psi = ck.Function(u_psi, element, patch, ck.FieldType.ROTATION)

    fields = {
        "displacement": disp,
        "n11": traction[0], "n22": traction[1], "n12": traction[2],
        "m11": moment[0],   "m22": moment[1],   "m12": moment[2],
        "psi":      primal[3],
        "curl_psi": curl_psi,
        "rot1":     rotation[0], "rot2":   rotation[1],
    }
    with ck.BezierVtuWriter(os.path.join(OUT_DIR, f"hyperboloid_clamped_{tag}.vtu")) as writer:
        writer.add(patch, functions=fields)

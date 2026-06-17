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
# # Scordelis-Lo Roof Benchmark
#
# **References**
#
# 1. A. C. Scordelis and K. S. Lo, *"Computer analysis of cylindrical shells"*,
#    ACI Journal Proceedings, 1964, **61**(5), pp. 539-561. — the original problem.
# 2. R. H. MacNeal and R. L. Harder, *"A proposed standard set of problems to test
#    finite element accuracy"*, Finite Elements in Analysis and Design, 1985,
#    **1**(1), pp. 3-20. — adopted it into the standard benchmark suite.
# 3. P. Krysl, *"Benchmarking Computational Shell Models"*, Archives of
#    Computational Methods in Engineering, 2023, **30**(1), pp. 301-315.
#    — re-derives the converged references used here ($0.3020$ shear-flexible,
#    $0.3006$ shear-rigid) and shows the long-quoted $0.3024$ to be spurious.

# %%
import csv
import os

import numpy as np
import pyck as ck

PATH = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
OUT_DIR = os.path.join(PATH, "scordelis")
os.makedirs(OUT_DIR, exist_ok=True)

# %% [markdown]
# ## Problem setup
#
# A short cylindrical roof of radius $R$ and length $L$, supported at its curved
# ends by **rigid diaphragms** and free along its two straight longitudinal edges,
# under its own weight (a uniform body load $q$ per unit mid-surface area along
# gravity). Exploiting the two planes of symmetry (midspan $y=0$, crown $x=0$),
# **one quarter** is modelled. The target is the vertical deflection
# $|u_\text{ref}|$ at the **mid-span of the free longitudinal edge**.
#
# $$
# \begin{aligned}
# \text{geometry:}\quad & R = 25, \quad L = 50, \quad
#   \phi_0 = 40^\circ\ (\text{span } 80^\circ), \quad t = 0.25 \\[4pt]
# \text{material \& load:}\quad & E = 4.32\times10^{8}, \quad
#   \nu = 0, \quad q = 90
# \end{aligned}
# $$
#
# The quarter sits in the natural textbook frame: $Z$ vertical, cylinder axis along
# $Y$, the whole patch in $x\ge0,\,y\ge0$. Gravity acts in $-Z$.
#
# <img src="scordelis/scordelis_lo_geom.svg" width="480" align="center" alt="Scordelis-Lo roof: quarter model in the textbook frame.">


# %%
R = 25.0
L = 50.0
ARC_DEG = 80.0
T = 0.25
E = 4.32e8
NU = 0.0
Q = np.array([0.0, 0.0, -90.0])


def scordelis_quarter_roof(n: int, deg: int, layer_width: float = 0.0) -> ck.SurfacePatch:
    """Build the quarter-roof NURBS patch: a quarter-cylinder reoriented into the
    textbook frame (Z up, axis along Y).

    With ``layer_width > 0`` the mesh is a **Bathe two-region boundary-layer mesh**
    graded toward the two forced boundaries — the free edge (u=1) and the diaphragm
    (v=1): ``(n-deg)//2`` elements fill a band of physical width ``layer_width`` next
    to each boundary, the rest span the interior. The same physical width maps to a
    different parametric fraction per direction (arc u vs. axial v). Otherwise the
    mesh is uniform.

    Parameters:
        n: Number of control points per parametric direction.
        deg: Polynomial degree of the patch in both directions.
        layer_width: Physical width of the graded boundary band; 0 for a uniform mesh.

    Returns:
        The quarter-roof patch in the textbook frame, named ``"roof"``.
    """
    nu = nv = n if layer_width <= 0.0 else deg + 1   # graded: start minimal, refine below
    patch = ck.SurfacePatch.quarter_cylinder(
        nu=nu, nv=nv, deg=deg, radius=R, height=L / 2.0, angle=ARC_DEG / 2.0,
        name="roof",
    )
    bu, bv = patch.basis
    cps = np.asarray(patch.control_points)
    # reorient (X,Y,Z) -> (Y,Z,X); weights live in the basis, so the arc is exact
    new = np.column_stack([cps[:, 1], cps[:, 2], cps[:, 0]])
    patch = ck.SurfacePatch(bu, bv, new, name=patch.name)

    if layer_width > 0.0:
        m = (n - deg) // 2
        arc_len = R * np.deg2rad(ARC_DEG / 2.0)
        su = 1.0 - layer_width / arc_len
        sv = 1.0 - layer_width / (L / 2.0)
        for u in ([j * su / m for j in range(1, m + 1)] +
                  [su + j * (1.0 - su) / m for j in range(1, m)]):
            patch = patch.insert_knot(0, u)
        for v in ([j * sv / m for j in range(1, m + 1)] +
                  [sv + j * (1.0 - sv) / m for j in range(1, m)]):
            patch = patch.insert_knot(1, v)
    return patch

# %% [markdown]
# ## Reference values
#
# The traditional $0.3024$ target is **spurious** (Krysl 2023): never actually
# computed, only historically *proposed* and propagated by citation error. That work
# re-establishes the converged value for a shear-flexible (Reissner-Mindlin) shell,
# $$ |u_\text{ref}| = 0.3020. $$
# RM-Hier-4p is shear-flexible, so this is the appropriate reference.

# %%
REFERENCE_RM = 0.3020  # Reissner-Mindlin (shear-flexible)
REFERENCE_KL = 0.3006  # Kirchhoff-Love (shear-rigid)
U_REF_POINT = np.array([[1.0, 0.0]])

# %% [markdown]
# ## Boundary conditions
#
# - **Crown plane** $x=0$ ($u=0$) — symmetry: $U_X=0,\ \text{ROT}_N=0$.
# - **Midspan plane** $y=0$ ($v=0$) — symmetry: $U_Y=0,\ \text{ROT}_N=0$ (the
#   $U_Y$ condition also anchors the axial rigid-body mode).
# - **Curved end** $y=L/2$ ($v=1$) — rigid diaphragm: $U_X=U_Z=0$, axial $U_Y$ free.
# - **Free longitudinal edge** ($u=1$) — left free; where $u_\text{ref}$ is measured.
#
# Imposed weakly through Nitsche conditions (variationally consistent):
#
# - **RM-Hier-4p / 5p → symmetric Nitsche.** Lagrange multipliers are rank-deficient here
#   (the crown and diaphragm both pin $U_X$ at their shared corner, so the multiplier
#   blocks are linearly dependent — an inf-sup failure that worsens with refinement);
#   Nitsche adds no multiplier DOFs and is consistent. Per-field weights scale the
#   displacement traces ($\beta_u\sim E\,t\,n_{el}$) and the bending rotation
#   ($\beta_\text{rot}\sim E\,t^3 n_{el}$); on the uniform mesh $n_{el}\sim 1/h$.
# - **KL-3p → penalty.** The rotation-free element has no `ROT_N` flux for Nitsche (it
#   crashes), and Lagrange is rank-deficient, so it falls back to a penalty
#   ($\beta_u=P\,E\,t$, $\beta_\text{rot}=P\,E\,t^3$). The `ROT_N` trace is the recovered
#   bending rotation, so the same three functions apply.

# %%
C_NIT = 300.0  # Nitsche stabilisation


def crown_symmetry(
    prob: ck.LinearElasticProblem, patch: ck.SurfacePatch, gauss1: ck.QuadratureRule, 
    nel: int
) -> None:
    """Crown symmetry plane."""
    w_u = C_NIT * E * T * nel
    w_rot = C_NIT * E * T**3 * nel
    c = ck.NitscheBoundaryCondition(patch.boundary(0, True), gauss1)
    c.add(ck.Field.U_X, w_u).add(ck.Field.ROT_N, w_rot)
    prob.add_condition(c, patch="roof")


def midspan_symmetry(
    prob: ck.LinearElasticProblem, patch: ck.SurfacePatch, gauss1: ck.QuadratureRule,
    nel: int
) -> None:
    """Midspan symmetry plane."""
    w_u = C_NIT * E * T * nel
    w_rot = C_NIT * E * T**3 * nel
    c = ck.NitscheBoundaryCondition(patch.boundary(1, True), gauss1)
    c.add(ck.Field.U_Y, w_u).add(ck.Field.ROT_N, w_rot)
    prob.add_condition(c, patch="roof")


def rigid_diaphragm(
    prob: ck.LinearElasticProblem, patch: ck.SurfacePatch, gauss1: ck.QuadratureRule, 
    nel: int
) -> None:
    """Rigid diaphragm at the curved end."""
    w_u = C_NIT * E * T * nel
    c = ck.NitscheBoundaryCondition(patch.boundary(1, False), gauss1)
    c.add(ck.Field.U_X, w_u).add(ck.Field.U_Z, w_u)
    prob.add_condition(c, patch="roof")

# %% [markdown]
# ## Solver

# %%
def solve_roof(prob: ck.LinearElasticProblem):
    """Solve the quarter roof Scordelis-Lo Roof problem."""
    patch = prob.patches[0]
    bu = patch.basis[0]
    deg, nel = bu.degree, bu.num_intervals

    # Self-weight body traction (force per unit mid-surface area)
    prob.add_domain_load(Q)

    # Weak boundary conditions (Nitsche); weights scale with nel ~ 1/h
    gauss1 = ck.GaussLegendre(deg + 1, dim=1)
    crown_symmetry(prob, patch, gauss1, nel)
    midspan_symmetry(prob, patch, gauss1, nel)
    rigid_diaphragm(prob, patch, gauss1, nel)

    # Solve linear problem
    u = ck.solve(prob)
    return u

# %% [markdown]
# ## Studies
#
# Each study sweeps the mesh, prints a convergence table, and writes the rows to a CSV
# for postprocessing (the convergence figures are generated from these by `plot.py`).

# %%
def save_rows(rows, path):
    """Write study result rows to `path` as CSV (columns from the first row's keys)."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {path}")

# %% [markdown]
# ### Polynomial refinement
#
# RM-Hier-4p at $p=3,4,5,6$ on the **uniform** mesh, against control points per
# direction $N$. Higher $p$ approaches the reference faster. Each degree starts 
# at its coarsest single-element mesh ($N=\deg+1$).
# %%
n_sweep = (4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 20, 24, 32)
print(f"\nStudy 1 - polynomial refinement (RM-Hier-4p, uniform, ref u* = {REFERENCE_RM:.4f})")

rows = []
for deg in (3, 4, 5, 6):
    print(f"\n######## p={deg} ########")
    print(f"{'elems':>6} {'N':>4} {'ndof':>8} {'|w_A|':>12} {'err %':>9}")
    for n in n_sweep:
        if n - deg < 1:                  # >= 1 element/direction (min N = deg+1)
            continue

        try:
            patch = scordelis_quarter_roof(n, deg)
            material = ck.PlaneStress2d(E, NU, T)
            element = ck.ShellReissnerMindlinHier4p(material)
            gauss2 = ck.GaussLegendre(deg + 1, dim=2)
            prob = ck.LinearElasticProblem([patch], element, gauss2)

            u = solve_roof(prob)
            fn = ck.Function(u, element, patch, ck.FieldType.DISPLACEMENT)
            disp_A = np.asarray(fn(U_REF_POINT)).reshape(3)
            ndof = patch.num_control_pts * element.num_node_dofs
            w = disp_A @ np.array([0.0, 0.0, 1.0])

        except Exception as exc:         # rank-deficient on coarsest meshes
            print(f"{n - deg:>6} {n:>4}   skipped ({type(exc).__name__})", flush=True)
            continue

        err = 100.0 * (abs(w) - REFERENCE_RM) / REFERENCE_RM
        print(f"{n - deg:>6} {n:>4} {ndof:>8} {abs(w):>12.6f} {err:>8.3f}%", flush=True)
        rows.append({"mesh": "uniform", "element": "ShellReissnerMindlinHier4p",
                     "n": n, "deg": deg, "ndof": ndof, "w_signed": w, "w_abs": abs(w),
                     "reference": REFERENCE_RM, "err_pct": err})

print()
save_rows(rows, os.path.join(OUT_DIR, "results_polynomial.csv"))

# %% [markdown]
#
# <img src="scordelis/convergence_polynomial.svg" width="620" align="center" alt="RM-Hier-4p polynomial refinement: normalized deflection vs control points per direction.">


# %% [markdown]
# ### Element comparison
#
# KL-3p, RM-Hier-4p and RM-Hier-5p at $p=3$ on the **uniform** mesh, comparing the
# **normalized** deflection $|u_\text{ref}|/u^\ast$ per total DOF. The shear-flexible
# RM elements approach $1$; the shear-rigid KL-3p settles about 0.5% below it.
#
# Free-edge deflection $\lvert u_\text{ref}\rvert$ at $p=3$ for increasing refinement
# ($N$ = control points per direction; reference $u^\ast = 0.3020$):
#
# | formulation | $N=8$ | $N=16$ | $N=24$ | $N=32$ | $N=48$ |
# |:--|--:|--:|--:|--:|--:|
# | KL-3p (Kiendl) | 0.30045 | 0.30059 | 0.30059 | 0.30059 | 0.30059 |
# | RM-Hier-4p | 0.30066 | 0.30108 | 0.30131 | 0.30150 | 0.30174 |
# | RM-Hier-5p (Oesterle) | 0.30076 | 0.30131 | 0.30159 | 0.30177 | 0.30193 |
#
# The shear-flexible RM elements converge up toward $u^\ast = 0.3020$; KL-3p plateaus at
# the shear-rigid Kirchhoff-Love value $0.3006$.

# %%
n_sweep = (5, 6, 7, 8, 9, 10, 12, 14, 16, 20, 24, 32, 40, 48)
deg = 3
elements = (
    ("ShellReissnerMindlinHier4p", ck.ShellReissnerMindlinHier4p),
    ("ShellReissnerMindlinHier5p", ck.ShellReissnerMindlinHier5p),
    ("ShellKirchhoffLove3p",       ck.ShellKirchhoffLove3p),
)
print(f"\nStudy 2 - element comparison (p={deg}, uniform, ref u* = {REFERENCE_RM:.4f})")

rows = []
for ename, ecls in elements:
    print(f"\n######## {ename} | p={deg} ########")
    print(f"{'elems':>6} {'N':>4} {'ndof':>8} {'|w_A|':>12} {'err %':>9}")
    for n in n_sweep:
        if n - deg < 2:  # >= 2 elements/direction
            continue
        try:
            patch = scordelis_quarter_roof(n, deg)
            material = ck.PlaneStress2d(E, NU, T)
            element = ecls(material)
            gauss2 = ck.GaussLegendre(deg + 1, dim=2)
            prob = ck.LinearElasticProblem([patch], element, gauss2)
            
            u = solve_roof(prob)
            fn = ck.Function(u, element, patch, ck.FieldType.DISPLACEMENT)
            disp_A = np.asarray(fn(U_REF_POINT)).reshape(3)
            ndof = patch.num_control_pts * element.num_node_dofs
            w = disp_A @ np.array([0.0, 0.0, 1.0])

        except Exception as exc:
            print(f"{n - deg:>6} {n:>4}   skipped ({type(exc).__name__})", flush=True)
            continue

        err = 100.0 * (abs(w) - REFERENCE_RM) / REFERENCE_RM
        print(f"{n - deg:>6} {n:>4} {ndof:>8} {abs(w):>12.6f} {err:>8.3f}%", flush=True)
        rows.append({"mesh": "uniform", "element": ename, "n": n, "deg": deg,
                     "ndof": ndof, "w_signed": w, "w_abs": abs(w),
                     "reference": REFERENCE_RM, "err_pct": err})

print()
save_rows(rows, os.path.join(OUT_DIR, "results_element.csv"))

# %% [markdown]
#
# <img src="scordelis/convergence_element.svg" width="620" align="center" alt="Element comparison (KL-3p, RM-Hier-4p, RM-Hier-5p, p=3): normalized deflection vs DOFs.">

# %% [markdown]
# ### Energy convergence (uniform mesh)
#
# The same self-convergence test used for the hyperboloid: relative strain-energy error
# $|U/U^{\ast}-1|$ vs mesh size (RM-Hier-4p, $p=3$).
#
# Both boundary layers (diaphragm at $v=1$, free edge at $u=1$) keep convergence
# pre-asymptotic on a **uniform** mesh; a **graded** mesh that resolves them with a
# Bathe two-region layer recovers the optimal $O(h^{2p})$ rate. Both are run below
# against a common over-refined reference.

# %%
DEG = 3
NEL_REF = 192
LAYER_W = 0.7 * np.sqrt(R * T)

def roof_energy(n, deg, layer_width=0.0):
    patch = scordelis_quarter_roof(n, deg, layer_width)
    material = ck.PlaneStress2d(E, NU, T)
    element = ck.ShellReissnerMindlinHier4p(material)
    gauss2 = ck.GaussLegendre(deg + 1, dim=2)
    prob = ck.LinearElasticProblem([patch], element, gauss2)

    u = solve_roof(prob)
    K, _ = ck.LinearElasticProblem([patch], element, gauss2).assemble()
    U = 0.5 * float(u @ (K @ u))
    ndof = patch.num_control_pts * element.num_node_dofs
    return U, ndof

print(f"\nStudy 3 - energy convergence (RM-Hier-4p, p={DEG}) — uniform vs graded")

rows = []
for label, lw in (("uniform", 0.0), ("graded", LAYER_W)):
    # Each mesh family is measured against its own over-refined reference.
    U_ref, ndof_ref = roof_energy(NEL_REF + DEG, DEG, lw)
    print(f"\n-- {label} mesh --   reference: nel={NEL_REF}, ndof={ndof_ref}, U* = {U_ref:.8e}")
    print(f"{'nel':>5} {'N':>4} {'ndof':>8} {'energy':>15} {'err %':>10}")
    for nel in (4, 6, 8, 12, 16, 24, 32, 48, 64):
        n = nel + DEG
        try:
            U, ndof = roof_energy(n, DEG, lw)
        except Exception as exc:           # rank-deficient on the coarsest meshes
            print(f"{nel:>5} {n:>4}   skipped ({type(exc).__name__})", flush=True)
            continue

        err = 100.0 * (U - U_ref) / U_ref
        print(f"{nel:>5} {n:>4} {ndof:>8} {U:>15.6e} {err:>9.4f}%", flush=True)
        rows.append({"mesh": label, "element": "ShellReissnerMindlinHier4p", "nel": nel,
                     "n": n, "deg": DEG, "ndof": ndof, "energy": U, "energy_ref": U_ref,
                     "energy_err_pct": err, "layer_width": lw})

save_rows(rows, os.path.join(OUT_DIR, "results_energy.csv"))


# %% [markdown]
#
# <img src="scordelis/convergence_energy_graded.svg" width="620" align="center" alt="Scordelis-Lo roof: uniform vs graded mesh energy error (RM-Hier-4p, p=3), against the optimal O(h^2p) slope.">

# %% [markdown]
# ## ParaView export
#
# The export models the **whole roof** — the full $80^\circ$ arc centred on the $Z$ axis
# ($\pm40^\circ$ either side, crown on top) over the full length $L$, with a rigid
# diaphragm on each curved end and both longitudinal edges free, **no symmetry
# exploited** — so the artifact is the complete shell, not a quarter. A uniform and a
# boundary-layer-graded mesh are each solved and written, with **displacement**,
# membrane forces $n^{ab}$, bending moments $m^{ab}$, the hierarchic shear potential
# $\psi$, the director tilt $w_\alpha$ (rotation), and the transverse-shear strain
# $\gamma_\alpha$, to an exact rational-Bézier `.vtu` (ParaView 5.9+). Open in ParaView,
# **warp by `displacement`**, and colour by e.g. `m11` or `psi`.

# %%
def _layer_knots(n: int, deg: int, phys: float, lw: float) -> list[float]:
    """Interior knots for a symmetric boundary-layer mesh: thin bands of physical width
    ``lw`` refined at **both** ends of the [0,1] direction, uniform in the interior."""
    nel = n - deg
    nb = max(1, nel // 4)                  # elements per end band
    ni = max(1, nel - 2 * nb)              # interior elements
    f = lw / phys                          # parametric band width
    left = [f * j / nb for j in range(1, nb + 1)]                    # 0 .. f
    inter = [f + (1.0 - 2.0 * f) * j / ni for j in range(1, ni + 1)]  # f .. 1-f
    right = [(1.0 - f) + f * j / nb for j in range(1, nb)]           # 1-f .. 1
    return left + inter + right


def scordelis_full_roof(n: int, deg: int, layer_width: float = 0.0) -> ck.SurfacePatch:
    """The whole roof as one patch: the full ``ARC_DEG`` arc x full length L, crown and
    midspan centred. Symmetry is not exploited, so both longitudinal edges are free and
    both curved ends carry a diaphragm. With ``layer_width > 0`` the mesh is graded into
    thin boundary bands of that physical width at **all four** forced edges."""
    nu = nv = n if layer_width <= 0.0 else deg + 1   # graded: start minimal, refine below
    patch = ck.SurfacePatch.quarter_cylinder(
        nu=nu, nv=nv, deg=deg, radius=R, height=L, angle=ARC_DEG, name="roof",
    )
    # Reorient into textbook frame and center
    cps = np.asarray(patch.control_points)
    reo = np.column_stack([cps[:, 1], cps[:, 2], cps[:, 0]])
    a = np.radians(ARC_DEG / 2.0)
    ca, sa = np.cos(a), np.sin(a)
    x, y, z = reo[:, 0], reo[:, 1], reo[:, 2]
    rot = np.column_stack([x * ca - z * sa, y - L / 2.0, x * sa + z * ca])
    patch = ck.SurfacePatch(patch.basis[0], patch.basis[1], rot, name=patch.name)

    if layer_width > 0.0:
        for u in _layer_knots(n, deg, R * np.deg2rad(ARC_DEG), layer_width):
            patch = patch.insert_knot(0, u)
        for v in _layer_knots(n, deg, L, layer_width):
            patch = patch.insert_knot(1, v)
    return patch


def solve_full_roof(patch: ck.SurfacePatch, deg: int):
    """Solve the whole roof (no symmetry). Rigid diaphragms on both curved ends fix
    U_X, U_Z; both longitudinal edges are free. Two DirectConstraints pin the only
    nullspaces: the constant psi and the axial rigid-body translation (constant U_Y,
    which both diaphragms leave free — an inconsequential rigid shift)."""
    element = ck.ShellReissnerMindlinHier4p(ck.PlaneStress2d(E, NU, T))
    prob = ck.LinearElasticProblem([patch], element, ck.GaussLegendre(deg + 1, dim=2))
    prob.add_domain_load(Q)
    gauss1 = ck.GaussLegendre(deg + 1, dim=1)
    for end in (True, False):                  # both curved ends y = 0 and y = L
        c = ck.LagrangeBoundaryCondition(patch.boundary(1, end), gauss1)
        c.add(ck.Field.U_X)
        c.add(ck.Field.U_Z)
        prob.add_condition(c, patch="roof")
    prob.add_constraint(ck.DirectConstraint([3], value=0.0))   # constant psi
    prob.add_constraint(ck.DirectConstraint([1], value=0.0))   # axial U_Y rigid body
    return element, ck.solve(prob)


# %%
N, DEG = 20, 3

for tag, lw in (("uniform", 0.0), ("graded", np.sqrt(R * T))):
    patch = scordelis_full_roof(N, DEG, lw)
    element, u = solve_full_roof(patch, DEG)

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

    with ck.BezierVtuWriter(os.path.join(OUT_DIR, f"scordelis_lo_roof_{tag}.vtu")) as writer:
        writer.add(patch, functions=fields)

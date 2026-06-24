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
# (ref-macneal)=
# **[1]** R. H. MacNeal and R. L. Harder, *"A proposed standard set of problems to test
# finite element accuracy"*, Finite Elements in Analysis and Design, 1985, **1**(1),
# pp. 3-20. — standardized the benchmark and proposed the canonical $0.3024$ target.
#
# (ref-belytschko)=
# **[2]** T. Belytschko et al., *"Stress projection for membrane and shear locking in shell
# finite elements"*, Computer Methods in Applied Mechanics and Engineering, 1985,
# **51**(1-3), pp. 221-258. — the "obstacle course" set (pinched sphere, pinched cylinder,
# Scordelis-Lo roof); for the roof it reinforces [1] rather than deriving it independently.
#
# (ref-krysl)=
# **[3]** P. Krysl, *"Benchmarking Computational Shell Models"*, Archives of Computational
# Methods in Engineering, 2023, **30**(1), pp. 301-315. — re-derives the converged references
# used here ($0.3020$ shear-flexible, $0.3006$ shear-rigid) and shows the long-quoted $0.3024$
# to be spurious.

# %%
import csv
import os

import numpy as np
import pyck as ck

PATH = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
OUT_DIR = os.path.join(PATH, "scordelis")

# %% [markdown]
# ## Problem setup
#
# A short cylindrical roof of radius $R$ and length $L$, supported at its curved
# ends by rigid diaphragms and free along its two straight longitudinal edges,
# under its own weight (a uniform body load $q$ per unit mid-surface area). 
#
# $$
# \begin{aligned}
# \mathrm{geometry:}\quad & R = 25, \quad L = 50, \quad
#   \phi_0 = 40^\circ\ (\text{span } 80^\circ), \quad t = 0.25 \\[4pt]
# \mathrm{material/load:}\quad & E = 4.32\times10^{8}, \quad
#   \nu = 0, \quad q = 90
# \end{aligned}
# $$
#
# <img src="scordelis/scordelis_lo_geom.svg" width="480" align="center" alt="Scordelis-Lo roof: quarter model in the textbook frame.">
#
# The quarter sits in the natural textbook frame: $Z$ vertical, cylinder axis along
# $Y$, the whole patch in $x\ge0,\,y\ge0$. Gravity acts in $-Z$. Exploiting the two planes of symmetry (midspan $y=0$, crown $x=0$), one quarter is modelled. The target is the vertical deflection $|u_\text{ref}|$ at the mid-span of the free longitudinal edge.

# %%
R = 25.0
L = 50.0
ARC_DEG = 80.0
T = 0.25
E = 4.32e8
NU = 0.0
Q = np.array([0.0, 0.0, -90.0])


def scordelis_quarter_roof(
    n: int, deg: int, layer_width: float = 0.0
) -> ck.SurfacePatch:
    """Build the quarter-roof NURBS patch: a quarter-cylinder reoriented into the
    textbook frame (Z up, axis along Y).

    Parameters:
        n: Number of control points per parametric direction.
        deg: Polynomial degree of the patch in both directions.
        layer_width: Physical width of the graded boundary band; 0 for a uniform mesh.

    Returns:
        The quarter-roof patch in the textbook frame.
    """
    nu = nv = n if layer_width <= 0.0 else deg + 1
    patch = ck.SurfacePatch.quarter_cylinder(
        nu=nu, nv=nv, deg=deg, radius=R, height=L / 2.0, angle=ARC_DEG / 2.0,
        name="roof",
    )
    bu, bv = patch.basis
    cps = np.asarray(patch.control_points)
    
    new = np.column_stack([cps[:, 1], cps[:, 2], cps[:, 0]])  # (X,Y,Z) -> (Y,Z,X)
    patch = ck.SurfacePatch(bu, bv, new, name=patch.name)

    if layer_width > 0.0:
        # Boundary graded refinement
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
# The traditional $0.3024$ target is shown to be spurious [[3]](ref-krysl): it was proposed by
# MacNeal & Harder [[1]](ref-macneal) without a derivation and then propagated by citation (e.g.
# the "obstacle course" set [[2]](ref-belytschko)). Because the deflection depends on the kinematic
# model, Krysl [[3]](ref-krysl) establishes **two** references and shows $0.3024$ sits above both —
# roughly $0.15\%$ above the shear-flexible value and $0.6\%$ above the shear-rigid one:
#
# - **Shear-flexible (Reissner-Mindlin):** $|u_\text{ref}| = 0.3020$. Krysl's proposed interval
#   $0.30200$–$0.30204$.
# - **Shear-rigid (Kirchhoff-Love):** $|u_\text{ref}| = 0.3006$, from discrete-Kirchhoff (DKT/STRI3)
#   and rotation-free isogeometric models.
#
# RM-Hier-4p is shear-flexible, so $0.3020$ is the appropriate reference --- the shear-rigid $0.3006$
# is the target for the rotation-free KL-3p element.

# %%
REFERENCE_RM = 0.3020  # Reissner-Mindlin (shear-flexible)
REFERENCE_KL = 0.3006  # Kirchhoff-Love (shear-rigid)
U_REF_POINT = np.array([[1.0, 0.0]])


# %% [markdown]
# ## Boundary conditions
#
# - **Crown plane** ($u=1$) — symmetry: $U_X=0,\ \text{ROT}_N=0$.
# - **Midspan plane** ($v=1$) — symmetry: $U_Y=0,\ \text{ROT}_N=0$ ($U_Y$ also anchors the axial RBM).
# - **Curved end** ($v=0$) — rigid diaphragm: $U_X=U_Z=0$, axial $U_Y$ free.
# - **Free longitudinal edge** ($u=0$) — left free; $u_\text{ref}$ is measured here.
#
# All BCs use **symmetric Nitsche** (variationally consistent, no saddle-point system).
# Stabilisation weights scale as $\alpha E t \cdot n_\text{el}$ (displacement) and
# $\alpha E t^3 \cdot n_\text{el}$ (rotation), following the hyperboloid benchmark convention.
# The hierarchic constant-$\psi$ null mode of RM-Hier-4p is removed with one DirectConstraint.

# %%
C_NIT = 10.0   # Nitsche stabilisation prefactor (dimensionless; scales with E·t·nel)


def crown_symmetry(prob: ck.LinearElasticProblem, patch,
                   gauss1: ck.QuadratureRule, nel: int) -> None:
    """Crown symmetry plane (Nitsche): U_X = 0, ROT_N = 0."""
    w_u, w_rot = C_NIT * E * T * nel, C_NIT * E * T**3 * nel
    c = ck.NitscheBoundaryCondition(patch.boundary(0, True), gauss1)
    _ = c.add(ck.Field.U_X, w_u).add(ck.Field.ROT_N, w_rot)
    prob.add_condition(c, patch="roof")


def midspan_symmetry(prob: ck.LinearElasticProblem, patch,
                     gauss1: ck.QuadratureRule, nel: int) -> None:
    """Midspan symmetry plane (Nitsche): U_Y = 0, ROT_N = 0 (U_Y also anchors the axial RBM)."""
    w_u, w_rot = C_NIT * E * T * nel, C_NIT * E * T**3 * nel
    c = ck.NitscheBoundaryCondition(patch.boundary(1, True), gauss1)
    _ = c.add(ck.Field.U_Y, w_u).add(ck.Field.ROT_N, w_rot)
    prob.add_condition(c, patch="roof")


def rigid_diaphragm(prob: ck.LinearElasticProblem, patch,
                    gauss1: ck.QuadratureRule, nel: int) -> None:
    """Rigid diaphragm at the curved end (Nitsche): U_X = U_Z = 0, axial U_Y free."""
    w_u = C_NIT * E * T * nel
    c = ck.NitscheBoundaryCondition(patch.boundary(1, False), gauss1)
    _ = c.add(ck.Field.U_X, w_u).add(ck.Field.U_Z, w_u)
    prob.add_condition(c, patch="roof")


# %% [markdown]
# ## Solver

# %%
def solve_roof(prob: ck.LinearElasticProblem, pin_psi_all: bool = False):
    """Solve the quarter roof (Nitsche BCs for all elements).

    ``pin_psi_all=True`` pins ψ globally on Hier-4p, reducing it to KL/rotation-free
    behaviour (target ≈ 0.3006).
    """
    patch = prob.patches[0]
    bu = patch.basis[0]
    deg, nel = bu.degree, bu.num_intervals

    prob.add_domain_load(Q)
    base = prob.element.base if isinstance(prob.element, ck.MixedMembraneStrainShell) else prob.element

    gauss1 = ck.GaussLegendre(deg + 1, dim=1)
    crown_symmetry(prob, patch, gauss1, nel)
    midspan_symmetry(prob, patch, gauss1, nel)
    rigid_diaphragm(prob, patch, gauss1, nel)

    if isinstance(base, ck.ShellReissnerMindlinHier4p):
        if pin_psi_all:
            n_cp = patch.num_control_pts
            prob.add_constraint(ck.DirectConstraint(
                [cp * 4 + 3 for cp in range(n_cp)], value=0.0))
        else:
            prob.add_constraint(ck.DirectConstraint([3], value=0.0))
    elif isinstance(base, ck.ShellReissnerMindlinHierDisp5p):
        edge = {int(c) for d in (0, 1) for a in (True, False)
                for c in patch.boundary(d, a).displacement_dofs}
        prob.add_constraint(ck.DirectConstraint(
            [cp * 5 + 3 for cp in edge] + [cp * 5 + 4 for cp in edge], value=0.0))

    u = ck.solve(prob)
    return u

# %% [markdown]
# ## Convergence study
#
# The study sweeps the uniform mesh, prints a convergence table, and writes the rows to a CSV
# for postprocessing (the convergence figure is generated from it by `plot.py`).

# %%
def save_rows(rows, path):
    """Write study result rows to `path` as CSV (columns from the first row's keys)."""
    if not rows:
        print(f"No rows to write to {path} (all solves skipped)")
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {path}")

# %% [markdown]
# ### Convergence under refinement
#
# At the roof's single slenderness $R/t = 100$ ($t = 0.25$), refine the **uniform** mesh and compare
# the normalized free-edge deflection $|u_\text{ref}|/u^\ast$ against DOFs for five formulations,
# all with Lagrange multiplier BCs:
#
# | Element | Description | Target $u^\ast$ |
# |---------|-------------|-----------------|
# | **RM-5p** | Standard 5p (independent rotations) | 0.3020 |
# | **RM-Hier-5p** | Echter difference-vector 5p | 0.3020 |
# | **RM-HierDisp-5p** | Hierarchic-displacement 5p | 0.3020 |
# | **RM-Hier-4p** | Rotation-free hierarchic 4p | 0.3020 |
# | **RM-Hier-4p (ψ=0)** | Hier-4p with ψ globally pinned → KL | 0.3006 |
#
# $N_\text{dof}$ counts the physical control-point DOFs only.

# %%
DEG = 3
RATIO = int(round(R / T))   # the roof's single slenderness, R/t = 100

# Per-family n sweeps chosen so that DOF counts (n² × dofs/node) approximately coincide
# across families at each shared tier. Pairs within ~2%:
#   3p/5p: (9,7)→~244, (18,14)→~976, (27,21)→~2196, (36,28)→~3904, (42,32)→~5120-5292
#   4p/5p: (9,8)→~322, (18,16)→~1288, (27,24)→~2898, (36,32)→~5120-5184
#   3p/4p: (7,6)→~144-147, (14,12)→~576-588, (21,18)→~1296-1323, (28,24)→~2304-2352, (42,36)→~5184-5292
# Finest tier: 5p n=32 → 5120, 4p n=36 → 5184, 3p n=42 → 5292  (all within 3%, all > 5000)
N_5P = (4, 5, 6, 7,  8, 10, 14, 16, 21, 24, 28)
N_4P = (4, 5, 6,     8,  9, 12, 16, 18, 24, 27, 32)
N_3P = (4, 5, 6, 7,  9, 14, 18, 21, 27, 28, 36)

# Entry: (CSV label, element class, pin_psi_all, reference value, n_sweep).
LAYER_WIDTH = (R * T) ** 0.5   # physical boundary layer width for graded meshes

FORMULATIONS = (
    ("ShellReissnerMindlin5p",              ck.ShellReissnerMindlin5p,         False, REFERENCE_RM, N_5P, 0.0),
    ("ShellReissnerMindlin5p_graded",       ck.ShellReissnerMindlin5p,         False, REFERENCE_RM, N_5P, LAYER_WIDTH),
    ("ShellReissnerMindlinHier5p",          ck.ShellReissnerMindlinHier5p,     False, REFERENCE_RM, N_5P, 0.0),
    ("ShellReissnerMindlinHier5p_graded",   ck.ShellReissnerMindlinHier5p,     False, REFERENCE_RM, N_5P, LAYER_WIDTH),
    ("ShellReissnerMindlinHierDisp5p",      ck.ShellReissnerMindlinHierDisp5p, False, REFERENCE_RM, N_5P, 0.0),
    ("ShellReissnerMindlinHierDisp5p_graded", ck.ShellReissnerMindlinHierDisp5p, False, REFERENCE_RM, N_5P, LAYER_WIDTH),
    ("ShellReissnerMindlinHier4p",          ck.ShellReissnerMindlinHier4p,     False, REFERENCE_RM, N_4P, 0.0),
    ("ShellReissnerMindlinHier4p_graded",   ck.ShellReissnerMindlinHier4p,     False, REFERENCE_RM, N_4P, LAYER_WIDTH),
    ("ShellReissnerMindlinHier4p_psi0",     ck.ShellReissnerMindlinHier4p,     True,  REFERENCE_KL, N_3P, 0.0),
    ("ShellReissnerMindlinHier4p_psi0_graded", ck.ShellReissnerMindlinHier4p,  True,  REFERENCE_KL, N_3P, LAYER_WIDTH),
    ("ShellKirchhoffLove3p",                ck.ShellKirchhoffLove3p,           False, REFERENCE_KL, N_3P, 0.0),
    ("ShellKirchhoffLove3p_graded",         ck.ShellKirchhoffLove3p,           False, REFERENCE_KL, N_3P, LAYER_WIDTH),
)


def solve_convergence(n: int, deg: int, element_cls: type[ck.Element],
                      pin_psi: bool = False,
                      layer_width: float = 0.0) -> tuple[float, int]:
    """Solve the quarter roof; return ``(|w_A|, physical DOF count)``.

    ``pin_psi=True`` pins ψ globally on Hier-4p (KL behaviour).
    ``layer_width > 0`` uses a boundary-graded mesh with bands of that physical width."""
    patch = scordelis_quarter_roof(n, deg, layer_width=layer_width)
    element = element_cls(ck.PlaneStress2d(E, NU, T))
    gauss2 = ck.GaussLegendre(deg + 1, dim=2)
    prob = ck.LinearElasticProblem([patch], element, gauss2)
    u = solve_roof(prob, pin_psi_all=pin_psi)
    disp_A = np.asarray(
        ck.Function(u, element, patch, ck.FieldType.DISPLACEMENT)(U_REF_POINT)).reshape(3)
    dofs_per_node = 3 if pin_psi else element.num_node_dofs
    ndof = patch.num_control_pts * dofs_per_node
    return abs(disp_A[2]), ndof


print(f"\nConvergence under refinement (R/t={RATIO}, p={DEG})")

rows = []
for ename, ecls, pin_psi, wref, n_sweep, lw in FORMULATIONS:
    print(f"\n######## {ename} | p={DEG} | ref={wref} ########")
    print(f"{'elems':>6} {'N':>4} {'ndof':>8} {'|w_A|':>12} {'|w|/wref':>10}")
    for n in sorted(set(n_sweep)):
        if n - DEG < 1:
            continue
        try:
            w, ndof = solve_convergence(n, DEG, ecls, pin_psi=pin_psi, layer_width=lw)
        except Exception as exc:
            print(f"{n - DEG:>6} {n:>4}   skipped ({type(exc).__name__})", flush=True)
            continue
        print(f"{n - DEG:>6} {n:>4} {ndof:>8} {w:>12.6f} {w / wref:>10.4f}", flush=True)
        rows.append({"element": ename, "ratio": RATIO, "n": n, "deg": DEG, "nel": n - DEG,
                     "ndof": ndof, "w_abs": w, "w_ref": wref, "w_over_ref": w / wref})

print()
save_rows(rows, os.path.join(OUT_DIR, "scordelis_convergence_results.csv"))

# %% [markdown]
#
# <img src="scordelis/scordelis_convergence.pdf" width="620" align="center" alt="Scordelis-Lo roof convergence at R/t=100: normalized free-edge deflection |u_ref|/u* vs DOFs, five formulations with Lagrange BCs: RM-5p, RM-Hier-5p, RM-HierDisp-5p, RM-Hier-4p (target 0.3020), and RM-Hier-4p with psi globally pinned to zero (KL behaviour, target 0.3006).">

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
#
# The **same field set** is exported for three elements for side-by-side comparison of the membrane
# fields $n^{ab}$: the standard rotation-free **RM-Hier-4p**; its membrane-locking-fixed
# **mixed-strain** element (`MixedMembraneStrainShell` wrapping RM-Hier-4p,
# `scordelis_lo_roof_as_*.vtu`); and the hierarchic five-parameter **RM-Hier-5p**
# (`scordelis_lo_roof_5p_*.vtu`). The 5p element has no twist potential, so its files omit the
# `psi`/`curl_psi` fields.
#
# **Membrane-force recovery.** For the mixed-strain element the displacement membrane block is
# suppressed, so $n^{ab}$ is **not** taken from the displacement gradient — it comes from the
# wrapper's own `TRACTION` evaluated on the *untruncated* solution `ck.solve(prob, full=True)` (which
# carries the assumed-strain field), giving a faithful, oscillation-free membrane force where the
# displacement recovery would lock.

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


def solve_full_roof(patch: ck.SurfacePatch, deg: int,
                    element_cls: type[ck.Element] = ck.ShellReissnerMindlinHier4p,
                    assumed_strain: bool = False):
    """Solve the whole roof (no symmetry). Rigid diaphragms on both curved ends fix U_X, U_Z; both
    longitudinal edges are free. The axial rigid-body translation (constant U_Y) is pinned for every
    element; the rotation-free 4p family additionally needs the constant-psi null mode pinned (slot
    3, which the 5p element lacks). With ``assumed_strain=True`` the base shell is wrapped in a
    ``MixedMembraneStrainShell`` (membrane-locking fix; the wrapper IS the element, its membrane
    block suppressed and supplied by the mixed strain field). Returns ``(element, u_full,
    num_physical)`` where ``u_full`` is the untruncated solution (membrane-force recovery needs it)."""
    base = element_cls(ck.PlaneStress2d(E, NU, T))
    quad = ck.GaussLegendre(deg + 1, dim=2)
    element = ck.MixedMembraneStrainShell(patch, base, quad) if assumed_strain else base
    prob = ck.LinearElasticProblem([patch], element, quad)
    prob.add_domain_load(Q)
    gauss1 = ck.GaussLegendre(deg + 1, dim=1)
    for end in (True, False):                  # both curved ends y = 0 and y = L
        c = ck.LagrangeBoundaryCondition(patch.boundary(1, end), gauss1)
        c.add(ck.Field.U_X)
        c.add(ck.Field.U_Z)
        prob.add_condition(c, patch="roof")
    prob.add_constraint(ck.DirectConstraint([1], value=0.0))       # axial U_Y rigid body (all)
    if isinstance(base, ck.ShellReissnerMindlinHier4p):            # 4p family: pin constant psi
        prob.add_constraint(ck.DirectConstraint([3], value=0.0))
    u_full = ck.solve(prob, full=True)         # untruncated (incl. aux strain / multiplier DOFs)
    return element, u_full, prob.num_physical_dofs


# %%
def export_full_roof(patch: ck.SurfacePatch, element, u_full, num_physical, path: str) -> None:
    """Write the standard field set (displacement, n^{ab}, m^{ab}, rotation) for one solved roof to
    an exact rational-Bezier .vtu. The physical (truncated) part of ``u_full`` drives the
    displacement / bending / psi Functions. The membrane force n^{ab} needs care: for the mixed
    shell the displacement membrane block is suppressed, so it is taken from ``TRACTION`` evaluated
    on the wrapper with the **untruncated** ``u_full`` (which carries the assumed-strain field) —
    faithful and oscillation-free; the plain/5p elements use the physical solution. For the
    rotation-free 4p family (psi at slot 3) ``psi`` and ``curl_psi`` are added; the 5p has none."""
    u = np.asarray(u_full)[:num_physical]      # physical DOFs for the displacement-basis Functions
    disp     = ck.Function(u, element, patch, ck.FieldType.DISPLACEMENT)
    moment   = ck.Function(u, element, patch, ck.FieldType.MOMENT)      # [m11, m22, m12] (bending)
    rotation = ck.Function(u, element, patch, ck.FieldType.ROTATION)    # [rot1, rot2]

    # Membrane force: the mixed shell needs the full solution (assumed-strain field) and recovers a
    # faithful n^{ab} through its own TRACTION; the plain element uses the physical solution.
    is_mixed = isinstance(element, ck.MixedMembraneStrainShell)
    traction = ck.Function(u_full if is_mixed else u, element, patch, ck.FieldType.TRACTION)
    fields = {
        "displacement": disp,
        "n11": traction[0], "n22": traction[1], "n12": traction[2],
        "m11": moment[0],   "m22": moment[1],   "m12": moment[2],
        "rot1":     rotation[0], "rot2":   rotation[1],
    }
    base = element.base if is_mixed else element
    if isinstance(base, ck.ShellReissnerMindlinHier4p):   # psi is a 4p-only field (slot 3)
        primal = ck.Function(u, element, patch, ck.FieldType.PRIMAL)    # [v_x, v_y, v_z, psi, ...]
        fields["psi"] = primal[3]
        # curl(psi) alone: zero every DOF except the psi slot (3), keeping the per-node DOF layout.
        arr = np.asarray(u).reshape(-1, element.num_node_dofs)
        psi_only = np.zeros_like(arr)
        psi_only[:, 3] = arr[:, 3]
        fields["curl_psi"] = ck.Function(psi_only.ravel(), element, patch, ck.FieldType.ROTATION)

    with ck.BezierVtuWriter(path) as writer:
        writer.add(patch, functions=fields)


N, DEG = 12, 4
# Standard 4p, its membrane-locking-fixed mixed-strain element (MixedMembraneStrainShell over 4p),
# and the hierarchic 5p element; non-standard ones get an "as_"/"5p_" filename prefix. Entry:
# (prefix, base element class, wrap in the mixed-strain shell?).
EXPORT_ELEMENTS = (
    ("",    ck.ShellReissnerMindlinHier4p, False),
    ("as_", ck.ShellReissnerMindlinHier4p, True),
    ("5p_", ck.ShellReissnerMindlinHierDisp5p, False),
)

for prefix, ecls, use_as in EXPORT_ELEMENTS:
    for tag, lw in (("uniform", 0.0), ("graded", np.sqrt(R * T))):
        patch = scordelis_full_roof(N, DEG, lw)
        element, u_full, num_physical = solve_full_roof(patch, DEG, ecls, assumed_strain=use_as)
        export_full_roof(patch, element, u_full, num_physical,
                         os.path.join(OUT_DIR, f"scordelis_lo_roof_{prefix}{tag}.vtu"))

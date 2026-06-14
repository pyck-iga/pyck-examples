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
import numpy as np
import pyck as ck

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
# <img src="../figures/scordelis/scordelis_lo_geom.svg" width="480" align="center" alt="Scordelis-Lo roof: quarter model in the textbook frame.">


# %%
R = 25.0
L = 50.0
ARC_DEG = 80.0
T = 0.25
E = 4.32e8
NU = 0.0
Q = np.array([0.0, 0.0, -90.0])


def scordelis_quarter_roof(n: int, deg: int) -> ck.SurfacePatch:
    """Build the quarter-roof NURBS patch: a quarter-cylinder reoriented into the
    textbook frame (Z up, axis along Y).

    Parameters:
        n: Number of control points per parametric direction.
        deg: Polynomial degree of the patch in both directions.

    Returns:
        The quarter-roof patch in the textbook frame, named ``"roof"``.
    """
    patch = ck.SurfacePatch.quarter_cylinder(
        nu=n, nv=n, deg=deg, radius=R, height=L / 2.0, angle=ARC_DEG / 2.0,
        name="roof",
    )
    bu, bv = patch.basis
    cps = np.asarray(patch.control_points)
    # reorient (X,Y,Z) -> (Y,Z,X); weights live in the basis, so the arc is exact
    new = np.column_stack([cps[:, 1], cps[:, 2], cps[:, 0]])
    return ck.SurfacePatch(bu, bv, new, name=patch.name)

# %% [markdown]
# ## Reference values
#
# The traditional $0.3024$ target is **spurious** (Krysl 2023): never actually
# computed, only historically *proposed* and propagated by citation error. That work
# re-establishes the converged value for a shear-flexible (Reissner-Mindlin) shell,
# $$ |u_\text{ref}| = 0.3020. $$
# RM-Hier-4p is shear-flexible, so this is the appropriate reference.

# %%
REFERENCE = 0.3020       # Reissner-Mindlin (shear-flexible)
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
# Imposed weakly with **Lagrange multipliers** (no penalty stiffness to tune). The
# same three functions work for every formulation: the `ROT_N` trace is the
# recovered bending rotation, so it applies even to the rotation-free KL-3p.

# %%
def crown_symmetry(
    prob: ck.LinearElasticProblem, patch: ck.SurfacePatch, gauss1: ck.GaussLegendre
) -> None:
    """Crown symmetry plane."""
    c = ck.LagrangeBoundaryCondition(patch.boundary(0, True), gauss1)
    c.add(ck.Field.U_X)
    c.add(ck.Field.ROT_N)
    prob.add_condition(c, patch="roof")


def midspan_symmetry(
    prob: ck.LinearElasticProblem, patch: ck.SurfacePatch, gauss1: ck.GaussLegendre
) -> None:
    """Midspan symmetry plane."""
    c = ck.LagrangeBoundaryCondition(patch.boundary(1, True), gauss1)
    c.add(ck.Field.U_Y)
    c.add(ck.Field.ROT_N)
    prob.add_condition(c, patch="roof")


def rigid_diaphragm(
    prob: ck.LinearElasticProblem, patch: ck.SurfacePatch, gauss1: ck.GaussLegendre
) -> None:
    """Rigid diaphragm at the curved end."""
    c = ck.LagrangeBoundaryCondition(patch.boundary(1, False), gauss1)
    c.add(ck.Field.U_X)
    c.add(ck.Field.U_Z)
    prob.add_condition(c, patch="roof")

# %% [markdown]
# ## Solver

# %%
def solve_roof(n: int, deg: int, element_cls: type[ck.Element]):
    """Solve the quarter roof on an (n,n) uniform mesh."""
    patch = scordelis_quarter_roof(n, deg)
    material = ck.PlaneStress2d(E, NU, T)
    element = element_cls(material)
    gauss2 = ck.GaussLegendre(deg + 1, dim=2)
    prob = ck.LinearElasticProblem([patch], element, gauss2)

    # Self-weight body traction (force per unit mid-surface area)
    prob.add_domain_load(Q)

    # Weak boundary conditions (Lagrange multipliers)
    gauss1 = ck.GaussLegendre(deg + 1, dim=1)
    crown_symmetry(prob, patch, gauss1)
    midspan_symmetry(prob, patch, gauss1)
    rigid_diaphragm(prob, patch, gauss1)

    # Solve linear problem
    u = ck.solve(prob)

    fn = ck.Function(u, element, patch, ck.FieldType.DISPLACEMENT)
    disp_A = np.asarray(fn(U_REF_POINT)).reshape(3)
    ndof = patch.num_control_pts * element.num_node_dofs
    return disp_A @ np.array([0.0, 0.0, 1.0]), ndof

# %% [markdown]
# ## Studies

# %% [markdown]
# ### Polynomial refinement
#
# RM-Hier-4p at $p=3,4,5,6$ on the **uniform** mesh, against control points per
# direction $N$. Higher $p$ approaches the reference faster. Each degree starts 
# at its coarsest single-element mesh ($N=\deg+1$).
# %%
n_sweep = (4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 20, 24, 32)
print(f"Study 1: polynomial refinement   (RM-Hier-4p, uniform, ref u* = {REFERENCE:.4f})")
for deg in (3, 4, 5, 6):
    print(f"\n######## p={deg} ########")
    print(f"{'elems':>6} {'N':>4} {'ndof':>8} {'|w_A|':>12} {'err %':>9}")
    for n in n_sweep:
        if n - deg < 1:                  # >= 1 element/direction (min N = deg+1)
            continue
        try:
            w, ndof = solve_roof(n, deg, ck.ShellReissnerMindlinHier4p)
        except Exception as exc:         # rank-deficient on coarsest meshes
            print(f"{n - deg:>6} {n:>4}   skipped ({type(exc).__name__})", flush=True)
            continue
        err = 100.0 * (abs(w) - REFERENCE) / REFERENCE
        print(f"{n - deg:>6} {n:>4} {ndof:>8} {abs(w):>12.6f} {err:>8.3f}%", flush=True)
print()

# %% [markdown]
#
# <img src="../figures/scordelis/convergence_polynomial.svg" width="620" align="center" alt="RM-Hier-4p polynomial refinement: normalized deflection vs control points per direction.">


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
# | KL-3p (Kiendl) | 0.30045 | 0.30059 | 0.30059 | 0.30061 | 0.30059 |
# | RM-Hier-4p | 0.30066 | 0.30108 | 0.30131 | 0.30150 | 0.30174 |
# | RM-Hier-5p (Echter) | 0.30076 | 0.30131 | 0.30159 | 0.30177 | 0.30194 |
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
print(f"Study 2: element comparison   (p={deg}, uniform, ref u* = {REFERENCE:.4f})")
for ename, ecls in elements:
    print(f"\n######## {ename} | p={deg} ########")
    print(f"{'elems':>6} {'N':>4} {'ndof':>8} {'|w_A|':>12} {'err %':>9}")
    for n in n_sweep:
        if n - deg < 2:                  # >= 2 elements/direction
            continue
        try:
            w, ndof = solve_roof(n, deg, ecls)
        except Exception as exc:
            print(f"{n - deg:>6} {n:>4}   skipped ({type(exc).__name__})", flush=True)
            continue
        err = 100.0 * (abs(w) - REFERENCE) / REFERENCE
        print(f"{n - deg:>6} {n:>4} {ndof:>8} {abs(w):>12.6f} {err:>8.3f}%", flush=True)
print()

# %% [markdown]
#
# <img src="../figures/scordelis/convergence_element.svg" width="620" align="center" alt="Element comparison (KL-3p, RM-Hier-4p, RM-Hier-5p, p=3): normalized deflection vs DOFs.">

# %% [markdown]
# ## ParaView export
#
# Solve a single mesh and write the **displacement** plus the membrane forces
# $n^{ab}$ and bending moments $m^{ab}$ to an exact rational-Bézier `.vtu`
# (ParaView 5.9+). The resultants come straight from `FieldType.TRACTION`
# (`n11,n22,n12`) and `FieldType.MOMENT` (`m11,m22,m12`) as contravariant curvilinear
# components. Open it in ParaView, **warp by `displacement`**, and colour by e.g. `m11`.
# (Transverse shear is omitted: the thin-shell constitutive shear is degenerate.)

# %%
n, deg = 12, 4
patch = scordelis_quarter_roof(n, deg)
element = ck.ShellReissnerMindlinHier4p(ck.PlaneStress2d(E, NU, T))
prob = ck.LinearElasticProblem([patch], element, ck.GaussLegendre(deg + 1, dim=2))
prob.add_domain_load(Q)
gauss1 = ck.GaussLegendre(deg + 1, dim=1)
crown_symmetry(prob, patch, gauss1)
midspan_symmetry(prob, patch, gauss1)
rigid_diaphragm(prob, patch, gauss1)
u = ck.solve(prob)

disp = ck.Function(u, element, patch, ck.FieldType.DISPLACEMENT)
traction = ck.Function(u, element, patch, ck.FieldType.TRACTION)   # [n11, n22, n12, q1, q2]
moment = ck.Function(u, element, patch, ck.FieldType.MOMENT)       # [m11, m22, m12]

fields = {
    "displacement": disp,
    "n11": traction[0], "n22": traction[1], "n12": traction[2],
    "m11": moment[0],   "m22": moment[1],   "m12": moment[2],
}
with ck.BezierVtuWriter("scordelis_lo_roof.vtu") as writer:
    writer.add(patch, functions=fields)

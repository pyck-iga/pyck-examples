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
# # Partly-Clamped Hyperbolic Paraboloid Benchmark
#
# A hyperbolic-paraboloid (saddle) shell, clamped along one edge and loaded by its own
# weight — the classic **bending-dominated** locking test of Bathe, Iosilevich & Chapelle.
# The doubly-curved saddle ($K<0$) with a single clamped edge makes it a more realistic and
# more demanding locking test than a flat plate or a free-edge elliptic shell, and the
# self-weight load excites a strongly **twist-dominated** response.
#
# The target quantities are the **strain energy** and the **vertical deflection** at the tip
# $X=L/2,\,Y=0$, reported against converged references for three slendernesses
# $t/L=1/100,\,1/1000,\,1/10000$.
#
# **References**
#
# 1. K.-J. Bathe, A. Iosilevich and D. Chapelle, *"An evaluation of the MITC shell
#    elements"*, Computers & Structures, 2000, **75**(1), pp. 1-30. — the original
#    partly-clamped hyperbolic-paraboloid benchmark (Section 3.2.2, Table 2).
# 2. P. Krysl, *"Benchmarking Computational Shell Models"*, Archives of Computational
#    Methods in Engineering, 2023, **30**(1), pp. 301-315. — re-derives the converged
#    (extrapolated) shear-flexible references used here as the convergence targets.

# %%
import csv
import os
import time

import numpy as np
import pyck as ck

HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
OUT_DIR = os.path.join(HERE, "hypar")            # all study outputs (CSVs + VTUs) go here
os.makedirs(OUT_DIR, exist_ok=True)

# %% [markdown]
# ## Problem setup
#
# The mid-surface is the hyperbolic paraboloid (saddle)
# $$ Z = X^2 - Y^2, \qquad X,\,Y \in [-L/2,\,L/2], $$
# clamped along the edge $X=-L/2$ and free on the other three edges, under its own weight
# (a uniform body load $\rho\,t$ per unit mid-surface area along $-Z$). The load is symmetric
# about $Y=0$ ($Z$ is even in $Y$), so **one half** is modelled: $X\in[-L/2,L/2]$, $Y\in[0,L/2]$,
# with a symmetry plane at $Y=0$. The self-weight scales with the thickness, so the strain energy
# scales as $1/t$ and the deflection as $1/t^2$. The reported strain energy is that of the **whole
# structure** ($2\times$ the half-model energy); the deflection target sits on the symmetry plane,
# so it is read directly.
#
# $$
# \begin{aligned}
# \text{geometry:}\quad & L = 1\ \text{m}, \quad Z = X^2 - Y^2 \\[4pt]
# \text{material \& load:}\quad & E = 2\times10^{11}\ \text{Pa}, \quad
#   \nu = 0.3, \quad \rho = 8000\ \text{N/m}^3
# \end{aligned}
# $$
#
# The saddle $Z=X^2-Y^2$ is separable and bi-quadratic, so it is reproduced **exactly** by a
# single bidegree-$(2,2)$ Bézier patch (control $z_{ij}=c^{x}_i+c^{y}_j$ with the parabolas'
# Bézier ordinates), then degree-elevated and knot-refined.
#
# <img src="hypar/hypar_geom.svg" width="520" align="center" alt="Partly-clamped hyperbolic paraboloid: saddle surface, clamped edge X=-L/2, half model with symmetry plane Y=0, self-weight load.">


# %%
L = 1.0
E = 2.0e11
NU = 0.3
RHO = 8000.0                             # self-weight force density [N/m^3]; per-area load = RHO*t

C_NIT = 300.0                            # Nitsche stabilisation constant
LAYER_C = 1.0                            # band width along the clamped edge = LAYER_C*sqrt(L*t)


def hypar_patch(deg: int, nel: int, layer_width: float = 0.0,
                nel_v: int | None = None) -> ck.SurfacePatch:
    """Exact bi-quadratic Bézier saddle ``Z = X^2 - Y^2`` on the half domain
    ``X in [-L/2,L/2], Y in [0,L/2]`` (u = X, v = Y; v=0 is the symmetry plane Y=0),
    degree-elevated to ``deg`` and knot-refined to ``nel`` (u) x ``nel_v`` (v) elements.

    ``nel_v`` defaults to ``nel``. Because ``v`` spans only the half domain ``[0,L/2]`` while ``u``
    spans the full width ``[-L/2,L/2]``, geometrically **square** cells need ``nel_v = nel//2``.

    With ``layer_width > 0`` the u-direction (across the clamped edge ``u=0``, X=-L/2) is a
    **two-region boundary-layer mesh**: ``nel//2`` uniform elements in the band ``X in [-L/2,
    -L/2+layer_width]`` and ``nel//2`` over the rest, resolving the clamped-edge layer. The
    v-direction stays uniform. ``layer_width=0`` is the plain uniform mesh.
    """
    nel_v = nel if nel_v is None else nel_v
    h = L / 2.0
    xs = np.array([-h, 0.0, h])
    cx = np.array([h * h, -h * h, h * h])                # x^2 deg-2 Bézier z-ordinates on [-L/2,L/2]
    ys = np.array([0.0, h / 2.0, h])                     # y on [0, L/2]
    cy = np.array([0.0, 0.0, -h * h])                    # -y^2 deg-2 Bézier z-ordinates on [0,L/2]
    cps = np.array([[xs[i], ys[j], cx[i] + cy[j]]        # u = X (i, fastest), v = Y (j)
                    for j in range(3) for i in range(3)], dtype=float)
    b = ck.BSpline.clamped_uniform(2, 3)                 # polynomial (saddle is exact, weights=1)
    patch = ck.SurfacePatch(b, b, cps, name="hypar")
    patch = patch.elevate_degree(0, deg - 2).elevate_degree(1, deg - 2)

    if layer_width > 0.0:                                # u: two-region layer mesh toward the clamp
        f = min(layer_width, 0.5 * L) / L                # parametric band width near u=0 (X linear in u)
        m = nel // 2
        knots = [j * f / m for j in range(1, m + 1)]                 # [0, f]   clamped-edge band
        knots += [f + j * (1.0 - f) / m for j in range(1, m)]        # (f, 1]   interior
        for u in knots:
            patch = patch.insert_knot(0, u)
        for k in range(1, nel_v):                         # v uniform
            patch = patch.insert_knot(1, k / nel_v)
    else:                                                # uniform: nel in u, nel_v in v
        for k in range(1, nel):
            patch = patch.insert_knot(0, k / nel)
        for k in range(1, nel_v):
            patch = patch.insert_knot(1, k / nel_v)
    return patch


def self_weight(t: float) -> np.ndarray:
    """Self-weight body load per unit mid-surface area, ``rho*t`` along ``-Z``."""
    return np.array([0.0, 0.0, -RHO * t])

# %% [markdown]
# ## Reference values
#
# Bathe-Iosilevich-Chapelle (Table 2, a $48\times24$ MITC16 mesh) report the strain energy and
# the vertical deflection $w$ at $X=L/2,\,Y=0$:
#
# | $t/L$ | strain energy $U$ [Nm] | $w(L/2,0)$ [m] |
# |--:|--:|--:|
# | $100$    | $1.6790\times10^{-3}$ | $-9.3355\times10^{-5}$ |
# | $1000$   | $1.1013\times10^{-2}$ | $-6.3941\times10^{-3}$ |
# | $10000$  | $8.9867\times10^{-2}$ | $-5.2988\times10^{-1}$ |
#
# As the convergence **targets** we use Krysl's extrapolated, shear-flexible references (the
# MITC16 single-mesh values above are slightly under-converged); RM-Hier elements are
# shear-flexible, so these are the appropriate targets.

# %%
# Krysl 2023 extrapolated, shear-flexible references: ratio -> (strain energy U [Nm], |w| [m])
REFERENCE = {
    100:   (1.68175e-3, 9.34956e-5),
    1000:  (1.10250e-2, 6.40044e-3),
    10000: (9.01458e-2, 5.31399e-1),
}
W_POINT = np.array([[1.0, 0.0]])                  # tip X=L/2, Y=0 on the symmetry plane (deflection target)

# %% [markdown]
# ## Boundary conditions
#
# - **Clamped edge** $X=-L/2$ ($u=0$) — fully fixed: $U_X=U_Y=U_Z=0$ and both bending
#   rotations $\text{ROT}_N=\text{ROT}_S=0$ (the hard clamp the shear-flexible reference assumes).
# - **Symmetry plane** $Y=0$ ($v=0$) — zero normal displacement $U_Y=0$ and zero bending
#   rotation $\text{ROT}_N=0$.
# - **Free edges** $X=L/2$ ($u=1$) and $Y=L/2$ ($v=1$).
#
# Both are imposed weakly by the symmetric **Nitsche** method (variationally consistent, no
# penalty to tune, no Lagrange inf-sup/rank-deficiency issue). Per-field weights scale the
# displacement traces ($\beta_u\sim E\,t\,n_{el}$) and the bending rotation
# ($\beta_\text{rot}\sim E\,t^3 n_{el}$), with $n_{el}\sim 1/h$ the elements per side. The clamp
# removes all rigid-body modes; the hierarchic constant-$\psi$ null mode is pinned with one
# DirectConstraint.

# %%
def clamp_edge(prob, patch, gauss1, t, nel) -> None:
    """Clamp the edge X=-L/2 (u=0) by Nitsche: all displacements + bending rotations."""
    w_u = C_NIT * E * t * nel
    w_rot = C_NIT * E * t**3 * nel
    c = ck.NitscheBoundaryCondition(patch.boundary(0, True), gauss1)
    c.add(ck.Field.U_X, w_u).add(ck.Field.U_Y, w_u).add(ck.Field.U_Z, w_u)
    c.add(ck.Field.ROT_N, w_rot).add(ck.Field.ROT_S, w_rot)
    prob.add_condition(c, patch="hypar")


def symmetry_plane(prob, patch, gauss1, t, nel) -> None:
    """Symmetry plane Y=0 (v=0) by Nitsche: zero normal displacement U_Y + bending rotation."""
    w_u = C_NIT * E * t * nel
    w_rot = C_NIT * E * t**3 * nel
    c = ck.NitscheBoundaryCondition(patch.boundary(1, True), gauss1)
    c.add(ck.Field.U_Y, w_u).add(ck.Field.ROT_N, w_rot)
    prob.add_condition(c, patch="hypar")

# %% [markdown]
# ## Solver

# %%
def solve_hypar(deg: int, nel: int, t: float,
                element_cls: type[ck.Element] = ck.ShellReissnerMindlinHier4p,
                assumed_strain: bool = False, layer_width: float = 0.0):
    """Solve the partly-clamped hyperbolic paraboloid; return ``(U, w_abs, ndof, t_asm, t_solve)``
    — the whole-structure strain energy, the tip deflection magnitude, the DOF count, and the
    assembly / linear-solve wall-clock times [s]. ``element_cls`` selects the base shell;
    ``assumed_strain=True`` wraps it in a ``MixedMembraneStrainShell`` (membrane-locking fix; the
    wrapper IS the element, its membrane block suppressed and supplied by the mixed strain field).
    ``layer_width > 0`` selects the clamped-edge boundary-layer mesh (see ``hypar_patch``)."""
    patch = hypar_patch(deg, nel, layer_width)
    base = element_cls(ck.PlaneStress2d(E, NU, t))
    gauss2 = ck.GaussLegendre(deg + 1, dim=2)
    element = ck.MixedMembraneStrainShell(patch, base, gauss2) if assumed_strain else base
    prob = ck.LinearElasticProblem([patch], element, gauss2)
    prob.add_domain_load(self_weight(t))

    gauss1 = ck.GaussLegendre(deg + 1, dim=1)
    clamp_edge(prob, patch, gauss1, t, nel)
    symmetry_plane(prob, patch, gauss1, t, nel)
    # RM-Hier-4p (4 DOFs/node) carries a constant-psi null mode (slot 3); pin it (the wrapper
    # forwards the 4 node DOFs, adding the strain field as global aux). The 5p needs no such pin.
    if element.num_node_dofs == 4:
        prob.add_constraint(ck.DirectConstraint([3], value=0.0))

    t0 = time.perf_counter()
    K, f = prob.assemble()
    t_asm = time.perf_counter() - t0

    t0 = time.perf_counter()
    u_full = ck.solve(K, f, full=True)           # untruncated (incl. mixed-strain aux DOFs)
    t_solve = time.perf_counter() - t0

    # Strain energy from a CLEAN (no-BC) stiffness so the Nitsche terms do not pollute it; the
    # wrapper's own stiffness includes the mixed coupling, so 0.5 u_full^T K u_full is the total
    # energy (bending+shear from u, membrane from the assumed field — the displacement membrane block
    # is suppressed). Whole structure = 2x the half-model (symmetry).
    K_energy, _ = ck.LinearElasticProblem([patch], element, gauss2).assemble()
    U = 2.0 * 0.5 * float(u_full @ (K_energy @ u_full))
    u = u_full[:prob.num_physical_dofs]
    w_abs = abs(float(ck.Function(u, element, patch, ck.FieldType.DISPLACEMENT)(W_POINT)[0, 2]))
    ndof = patch.num_control_pts * element.num_node_dofs
    return U, w_abs, ndof, t_asm, t_solve


# %% [markdown]
# ## Studies
#
# Two studies follow:
# - **Study 1** — consistency: strain energy $U/U^{\ast}$ and tip deflection $|w|/|w^{\ast}|$ vs
#   the references on the **uniform** mesh.
# - **Study 2** — convergence: relative energy error $|U/U^{\ast}-1|$ (log-log) on the **uniform**
#   mesh, $p=3,4,5$, plain vs mixed RM-Hier-4p, two slendernesses, against $O(h^{2(p-1)})$.

# %%
def save_rows(rows, path):
    """Write study result rows to `path` as CSV (columns from the first row's keys)."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {path}")

# %% [markdown]
# ## Study 1 — Consistency study on Uniform mesh
#
# Per thickness, sweep elements per direction on a **uniform** mesh ($p=3$), starting from the
# coarsest single-element mesh (`nel=1`). Two side-by-side panels show the two normalized targets
# approaching their references (converged value $\to 1$): the **tip deflection** $|w^h|/|w^{\ast}|$
# and the **strain energy** $U^h/U^{\ast}$, for the rotation-free **RM-Hier-4p** and its
# membrane-locking-fixed mixed variant **RM-Hier-4p-MMS** (base wrapped in a
# `MixedMembraneStrainShell`), at the three slendernesses $t/L=1/100,\,1/1000,\,1/10000$.
#
# Both targets approach from below (over-stiff). The partly-clamped saddle is **bending/twist-
# dominated**, so on a uniform mesh it suffers membrane locking that grows sharply as the shell
# thins: $t/L=1/100$ converges well, $1/10000$ stays far from the reference until much finer meshes.
# The mixed (MMS) element sits markedly closer to the reference, most visibly for the thin shells.

# %%
nel_sweep = (1, 2, 4, 6, 8, 12, 16, 20, 24, 32)
# (CSV name, base class, assumed-strain (MMS) fix?, degree). Rotation-free RM-Hier-4p, plain vs its
# membrane-locking-fixed mixed variant, at p=3.
ELEMENTS = (
    ("ShellReissnerMindlinHier4p",   ck.ShellReissnerMindlinHier4p, False, 3),
    ("ShellReissnerMindlinHier4pAS", ck.ShellReissnerMindlinHier4p, True,  3),
)
print("\nStudy 1 - Uniform-mesh consistency")

uniform_rows = []
for elem_name, elem_cls, use_as, deg in ELEMENTS:
    print(f"\n================ element: {elem_name}  (p={deg}) ================")
    for ratio in (100, 1000, 10000):
        t = 1.0 / ratio
        U_ref, w_ref = REFERENCE[ratio]
        print(f"\n######## t/L = 1/{ratio}   (ref U* = {U_ref:.6e} Nm, |w*| = {w_ref:.6e} m) ########")
        print(f"{'nel':>5} {'ndof':>8} {'U [Nm]':>15} {'errU %':>9} {'|w| [m]':>15} {'errw %':>9}")
        for nel in nel_sweep:
            try:
                U, w, ndof, _, _ = solve_hypar(deg, nel, t, elem_cls, assumed_strain=use_as)
            except Exception as exc:             # singular on the coarsest meshes
                print(f"{nel:>5}   skipped ({type(exc).__name__})", flush=True)
                continue
            errU = 100.0 * (U - U_ref) / U_ref
            errw = 100.0 * (w - w_ref) / w_ref
            print(f"{nel:>5} {ndof:>8} {U:>15.6e} {errU:>8.3f}% {w:>15.6e} {errw:>8.3f}%", flush=True)
            uniform_rows.append({"element": elem_name,
                                 "ratio": ratio, "t": t, "deg": deg, "nel": nel, "ndof": ndof,
                                 "energy": U, "energy_ref": U_ref, "energy_err_pct": errU,
                                 "w_abs": w, "w_ref": w_ref, "w_err_pct": errw})

save_rows(uniform_rows, os.path.join(OUT_DIR, "results_uniform.csv"))

# %% [markdown]
# <img src="hypar/hypar_consistency.pdf" width="860" align="center" alt="Partly-clamped hyperbolic paraboloid consistency on a uniform mesh: normalized tip deflection |w^h|/|w^ref| (left) and strain energy U^h/U^ref (right) vs elements per direction, three slendernesses (colour) for RM-Hier-4p and RM-Hier-4p-MMS (line style); membrane locking deepens as the shell thins, the mixed element sits closer to the reference.">

# %% [markdown]
# ## Study 2 — Convergence: uniform vs boundary-layer-graded mesh
#
# Energy-error convergence $|U/U^{\ast}-1|$ at degree $p=3$ for the **three slendernesses**
# $t/L=1/100,\,1/1000,\,1/10000$ and both formulations (plain rotation-free **RM-Hier-4p** and its
# mixed assumed-strain variant), measured against the **published Bathe-Iosilevich-Chapelle
# reference** $U^{\ast}$ (Table 2, MITC16). Two panels contrast the mesh:
#
# - **Uniform mesh** (left) — expected to **fail**: the clamped-edge boundary layer ($\sim\sqrt{Lt}$)
#   is unresolved, so the curves stall well below the optimal slope, worsening as the shell thins.
# - **Graded mesh** (right) — a band of width $\ell=\sqrt{L\,t}$ across the clamped edge: the layer
#   is resolved, so the energy recovers the **optimal $O(h^{2(p-1)})=O(h^4)$** rate (the $m=2$
#   signature). A crossover dip can appear at the finest meshes where the computed energy passes
#   through the slightly ($\sim$0.16%) under-converged MITC16 value.

# %%
DEG = 3                                            # p=3 only (the m=2 reference degree, rate 4)
NEL_SWEEP = (4, 6, 8, 12, 16, 24, 32)
RATE_RATIOS = (100, 1000, 10000)                   # three slendernesses, one colour each

# Two formulations: plain RM-Hier-4p and its mixed (assumed-membrane-strain) variant. Entry is
# (CSV/legend name, base element class, wrap in MixedMembraneStrainShell?).
RATE_FORMULATIONS = (
    ("ShellReissnerMindlinHier4p",   ck.ShellReissnerMindlinHier4p, False),
    ("ShellReissnerMindlinHier4pAS", ck.ShellReissnerMindlinHier4p, True),
)

# Published reference energies U* [Nm]: Bathe-Iosilevich-Chapelle Table 2 (MITC16, 48x24). The
# external reference removes the self-reference floor; the MITC16 value is ~0.16% under-converged
# at 1/100, so a curve can cross through it at the finest meshes.
U_BATHE = {100: 1.6790e-3, 1000: 1.1013e-2, 10000: 8.9867e-2}

rate_rows = []
print("\nStudy 2 - convergence vs published Bathe reference (uniform vs graded mesh; "
      "plain + mixed RM-Hier-4p, p=3)")
for mesh in ("uniform", "graded"):
    for ratio in RATE_RATIOS:
        t = 1.0 / ratio
        layer = 0.0 if mesh == "uniform" else np.sqrt(L * t)   # graded band ~ sqrt(L t)
        U_ref = U_BATHE[ratio]
        for ename, ecls, use_mms in RATE_FORMULATIONS:
            print(f"\n######## [{mesh}] {ename} | t/L = 1/{ratio}   (U_ref = {U_ref:.6e} Nm "
                  f"[Bathe], layer = {layer:.4g}) ########")
            print(f"{'nel':>5} {'ndof':>8} {'energy [Nm]':>16} {'err %':>9}")
            for nel in NEL_SWEEP:
                try:
                    U, w, ndof, _, _ = solve_hypar(DEG, nel, t, ecls, assumed_strain=use_mms,
                                                   layer_width=layer)
                except Exception as exc:             # singular on the coarsest meshes
                    print(f"{nel:>5}   skipped ({type(exc).__name__})", flush=True)
                    continue
                err = 100.0 * (U - U_ref) / U_ref
                print(f"{nel:>5} {ndof:>8} {U:>16.6e} {err:>8.3f}%", flush=True)
                rate_rows.append({"mesh": mesh, "ratio": ratio, "t": t, "deg": DEG,
                                  "element": ename, "nel": nel, "ndof": ndof, "energy": U,
                                  "energy_ref": U_ref, "energy_err_pct": err})

save_rows(rate_rows, os.path.join(OUT_DIR, "hypar_convergence_results.csv"))

# %% [markdown]
# <img src="hypar/hypar_rate.pdf" width="860" align="center" alt="Partly-clamped hyperbolic paraboloid energy convergence vs the Bathe reference at p=3: relative energy error vs elements per side on a uniform mesh (left, fails to reach the rate) and a boundary-layer graded mesh (right, recovers O(h^4)), three slendernesses t/L=1/100,1/1000,1/10000, plain vs mixed RM-Hier-4p, with the O(h^4) slope guide.">

# %% [markdown]
# ## ParaView export
#
# Solve the saddle on the **uniform** mesh at two slendernesses ($t/L=1/100$ and $1/1000$) and
# write each to an exact rational-Bézier `.vtu` (ParaView 5.9+). The hierarchic **RM-Hier-4p**
# element carries the twist potential $\psi$, so the export includes $\psi$ and
# $\operatorname{curl}\psi$ (the twist-dominated saddle makes these especially informative).
# Open in ParaView, **warp by `displacement`**, and colour by e.g. `psi` or `m12`.
#
# Each thickness is written for both the **plain** RM-Hier-4p (`hypar_t*.vtu`) and the
# **membrane-locking-fixed** mixed-strain element (`MixedMembraneStrainShell`, `hypar_as_t*.vtu`).
# For the latter the displacement membrane block is suppressed, so the membrane forces $n^{ab}$ come
# from the wrapper's own `TRACTION` evaluated on the **untruncated** solution
# `ck.solve(prob, full=True)` (which carries the assumed-strain field) — faithful and
# oscillation-free, whereas a plain displacement `TRACTION` would lock.

# %%
deg, nel = 4, 8                          # 8 along X (full width); v (Y half domain) -> nel//2 for square cells
gauss1 = ck.GaussLegendre(deg + 1, dim=1)


def export_hypar(t: float, assumed_strain: bool, path: str) -> None:
    """Solve and export one saddle. With ``assumed_strain`` the base shell is wrapped in a
    ``MixedMembraneStrainShell`` (membrane-locking fix); its membrane forces n^{ab} come from the
    wrapper's ``TRACTION`` on the **full** solution (the assumed-strain field — a plain TRACTION on
    the displacement would lock). All other fields come from the displacement as usual."""
    patch = hypar_patch(deg, nel, nel_v=nel // 2)        # square cells: half elements in the v (Y) half-domain
    base = ck.ShellReissnerMindlinHier4p(ck.PlaneStress2d(E, NU, t))
    gauss2 = ck.GaussLegendre(deg + 1, dim=2)
    element = ck.MixedMembraneStrainShell(patch, base, gauss2) if assumed_strain else base
    prob = ck.LinearElasticProblem([patch], element, gauss2)
    prob.add_domain_load(self_weight(t))
    clamp_edge(prob, patch, gauss1, t, nel)
    symmetry_plane(prob, patch, gauss1, t, nel)
    prob.add_constraint(ck.DirectConstraint([3], value=0.0))   # pin the constant-psi null mode
    u_full = ck.solve(prob, full=True)
    u = u_full[:prob.num_physical_dofs]

    # Fields whose membrane part is sourced from the mixed-strain aux DOFs (TRACTION, STRAIN) need
    # the full solution; the purely displacement-based fields use the physical part.
    is_mixed = isinstance(element, ck.MixedMembraneStrainShell)
    u_mix = u_full if is_mixed else u

    disp     = ck.Function(u, element, patch, ck.FieldType.DISPLACEMENT)
    moment   = ck.Function(u, element, patch, ck.FieldType.MOMENT)      # [m11, m22, m12]
    primal   = ck.Function(u, element, patch, ck.FieldType.PRIMAL)      # [v_x, v_y, v_z, psi]
    rotation = ck.Function(u, element, patch, ck.FieldType.ROTATION)    # −∇w_b + curl(psi)
    strain   = ck.Function(u_mix, element, patch, ck.FieldType.STRAIN)  # [..., gamma_1=6, gamma_2=7]
    traction = ck.Function(u_mix, element, patch, ck.FieldType.TRACTION)  # [n11,n22,n12,q1,q2]

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
        "gamma1":   strain[6],   "gamma2": strain[7],
    }
    with ck.BezierVtuWriter(path) as writer:
        writer.add(patch, functions=fields)


for tag, t in (("t100", 1.0 / 100), ("t1000", 1.0 / 1000)):
    export_hypar(t, False, os.path.join(OUT_DIR, f"hypar_{tag}.vtu"))
    export_hypar(t, True,  os.path.join(OUT_DIR, f"hypar_as_{tag}.vtu"))


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
# # Clamped Cylindrical Shell Strip Benchmark
#
# A cylindrical shell segment, clamped along one edge and loaded by a constant radial line
# load along the opposite free edge.  The strip behaves like a curved cantilever beam: 
# a quarter-circle of radius $R$ and width
# $L_y$, clamped at $\varphi=0$ and pushed radially at $\varphi=90^\circ$. The load is scaled with
# the thickness, $q_x = 0.1\,t^3$, which makes the reference tip deflection independent of $t$,
# so a single number exposes locking cleanly across every slenderness $R/t$.
#
# The target quantity is the radial tip deflection at the free edge, reported against an
# analytical Euler–Bernoulli (bending-only) reference for slendernesses
# $R/t = 10,\,100,\,1000,\,10^4$. It represents a bending dominated problem, therefore
# exposing the presence of shear-locking.

# %%
import csv
import os

import numpy as np
import pyck as ck

PATH = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
OUT_DIR = os.path.join(PATH, "cylindrical_strip")
os.makedirs(OUT_DIR, exist_ok=True)

# %% [markdown]
# ## Problem Setup
#
# The mid-surface is a quarter cylinder of radius $R$ and width $L_y$ (the cylinder axis is $Y$). The arc coordinate $\varphi$ is the parametric $u$-direction ($u=0$ the clamped edge, $u=1$ the free edge) and the axial coordinate $y$ is $v$. The clamped end sits at $(R,y,0)$ and the free end at $(0,y,R)$.
#
# $$
# \begin{aligned}
# \text{geometry:}\quad & R = 10,\quad L_y = 1,\quad \varphi\in[0,90^\circ] \\[4pt]
# \text{material and load:}\quad & E = 1000,\quad \nu = 0,\quad q_x = 0.1\,t^3\ \text{(radial line load)}
# \end{aligned}
# $$
#
# <img src="cylindrical_strip/cylindrical_strip_combined_preview.svg" width="560" align="center" alt="Clamped cylindrical strip: (a) undeformed quarter-arc strip with the clamped edge (hatched), the coordinate frame x/y/z, the radius R, and the radial line load q_x at the free tip; (b) the deformed strip coloured by displacement with the undeformed reference edges in gray.">
#
# A quarter circle is reproduced exactly by a single rational-quadratic (NURBS), whereass the strip is a linear extrusion of that arc in $Y$. With $\nu=0$ and a narrow strip the response is uniform along $Y$, so a single element spans the width and refinement is applied along the arc only.

# %%
# Geometric parameters
R, LY = 10.0, 1.0
# Material parameters
E, NU = 1000.0, 0.0


def strip_patch(deg: int, nel: int) -> ck.SurfacePatch:
    """Exact quarter-cylinder strip (radius `R`, width `LY`).
    
    Args:
        deg (int) : The polynomial degree of the patch.
        nel (int) : The number of elements in the arc direction.

    Returns:
        The strip patch `ck.SurfacePatch` instance.
    """
    s2 = np.sqrt(2.0)
    arc_w = [1.0, s2 / 2.0, 1.0]                          # quarter-circle quadratic NURBS weights
    arc_xz = [(R, 0.0), (R, R), (0.0, R)]                 # (X,Z) at phi = 0, 45, 90 deg

    cps = np.array([[x, y, z] for y in (0.0, LY)          # u = arc (i, fastest), v = axial Y (j)
                    for (x, z) in arc_xz], dtype=float)
    
    bu = ck.NURBS.clamped_uniform(2, 3, arc_w)            # arc (rational quadratic)
    bv = ck.NURBS.clamped_uniform(1, 2, [1.0, 1.0])       # axial (linear, weights 1)

    patch = ck.SurfacePatch(bu, bv, cps, name="strip")
    patch = patch.elevate_degree(0, deg - 2).elevate_degree(1, deg - 1)
    for k in range(1, nel):
        patch = patch.insert_knot(0, k / nel)             # refine arc only
        
    return patch

# %% [markdown]
# ## Reference Value
#
# Treat the strip as an **Euler–Bernoulli curved cantilever**, neglecting shear deformation and representing the shear and  membrane-locking-free limit the displacement element should reproduce. With a radial end load $P=q_x L_y$, the bending moment at the section at angle $\varphi$ is $M(\varphi)=-P R\cos\varphi$ (the radial-load moment about that section), and Castigliano's theorem gives the radial tip deflection
#
# $$ 
# \delta = \frac{1}{EI}\int_0^{\pi/2}\! M\,\frac{\partial M}{\partial P}\,R\,\mathrm{d}\varphi
#    = \frac{P R^3}{EI}\!\int_0^{\pi/2}\!\cos^2\varphi\,\mathrm{d}\varphi
#    = \frac{\pi\,P R^3}{4\,EI}. 
# $$
#
# With $P=0.1\,t^3$, $I=\tfrac{L_y t^3}{12}$, $E=1000$, $R=10$ the thickness cancels: 
#
# $$ 
# \delta = \frac{\pi\,(0.1\,t^3)(10^3)}{4\cdot 1000\cdot t^3/12} = 0.3\,\pi \approx 0.9425,
# $$ 
#
# independent of $t$ — a single reference for every slenderness.

# %%
WREF = 0.3 * np.pi                # analytical Euler-Bernoulli tip deflection
W_POINT = np.array([[1.0, 0.5]])  # free edge (u=1) mid-width (v=0.5): radial = Z

# %% [markdown]
# ## Boundary and Loading Conditions
#
# - **Clamped edge** $\varphi=0$ ($u=0$) — fully fixed: $U_X=U_Y=U_Z=0$ and both bending rotations
#   $\text{ROT}_N=\text{ROT}_S=0$, imposed exactly with **Lagrange multipliers**. This gives an
#   indefinite saddle-point system, solved by the equilibrated sparse solver.
# - **Radial line load** at the free edge $\varphi=90^\circ$ ($u=1$): a constant $Z$-traction
#   $q_x=0.1\,t^3$ (force per unit length; total $P=q_x L_y$).

# %%
def clamp_edge(prob, patch, gauss1) -> None:
    """Clamp the edge phi=0 (u=0) using Lagrange multipliers."""
    c = ck.LagrangeBoundaryCondition(patch.boundary(0, True), gauss1)
    c.add(ck.Field.U_X).add(ck.Field.U_Y).add(ck.Field.U_Z)
    c.add(ck.Field.ROT_N).add(ck.Field.ROT_S)
    prob.add_condition(c, patch="strip")


def radial_load(prob, patch, gauss1, t) -> None:
    """Constant radial (Z) line load q_x = 0.1 t^3 along the free edge u=1 (phi=90 deg)."""
    ld = ck.LoadBoundaryCondition(patch.boundary(0, False), gauss1)
    ld.add(ck.Field.U_Z, -0.1 * t**3)
    prob.add_condition(ld, patch="strip")

# %% [markdown]
# ## Solver
#
# The clamp removes all rigid-body modes, whereas the hierarchic constant-$\psi$ null mode is pinned with one
# DirectConstraint.

# %%
def solve_strip(
    deg: int, nel: int, t: float,
    element_cls: type[ck.Element], 
    assumed_strain: bool = False
) -> tuple[float, int]:
    """Solve the clamped cylindrical strip.

    Args:
        deg) : The polynomial degree for the cylinder strip patch. 
        nel : The number of elements in the arc direction.
        t : The thickness of the cylinder strip shell.
        element_cls : The element type.
        assumed_strain : Whether a mixed formulation is used to avoid membrane-locking.

    Returns:
        The radial tip reference deflection magnitude and the primal DOF count.
    """
    patch = strip_patch(deg, nel)
    base = element_cls(ck.PlaneStress2d(E, NU, t))
    gauss2 = ck.GaussLegendre(deg + 1, dim=2)
    element = ck.MixedMembraneStrainShell(patch, base, gauss2) if assumed_strain else base
    prob = ck.LinearElasticProblem([patch], element, gauss2)

    gauss1 = ck.GaussLegendre(deg + 1, dim=1)
    radial_load(prob, patch, gauss1, t)
    clamp_edge(prob, patch, gauss1)
    
    if isinstance(base, ck.ShellReissnerMindlinHier4p):  # pin the psi zero-energy mode
        prob.add_constraint(ck.DirectConstraint([3], value=0.0))
    
    elif isinstance(base, ck.ShellReissnerMindlinHier5p):  # anchor the constant-v_s shear kernel
        edge = {int(c) for d in (0, 1) for a in (True, False)
                for c in patch.boundary(d, a).displacement_dofs}
        prob.add_constraint(ck.DirectConstraint(
            [cp * 5 + 3 for cp in edge] + [cp * 5 + 4 for cp in edge], value=0.0))

    u_full = ck.solve(prob, full=True)
    u = u_full[:prob.num_physical_dofs]
    w_abs = abs(float(ck.Function(u, element, patch, ck.FieldType.DISPLACEMENT)(W_POINT)[0, 2]))
    ndof = patch.num_control_pts * element.num_node_dofs
    return w_abs, ndof

# %% [markdown]
# ## Study 1 — Tip Deflection vs Slenderness
#
# At a fixed **coarse** mesh ($n_{el}=4$, $p=3$), sweep the slenderness $R/t=10,\dots,10^4$ and plot
# the radial tip **deflection** $|w|$ for six formulations: the hierarchic **RM-Hier-4p** and
# **RM-Hier-5p** (shear-locking-free) and the standard **RM-5p**, each as the pure-displacement
# element and as its mixed membrane-strain variant (**MMS**, the base wrapped in a
# `MixedMembraneStrainShell`). Because $q_x=0.1\,t^3$ makes the reference thickness-independent
# ($w^{\mathrm{ref}}=0.3\pi$), a locking-free element traces a flat line at the reference.
#
# The MMS wrapper removes **membrane** locking; the hierarchic elements are additionally **shear**-
# locking-free — so RM-Hier-4p/5p-MMS sit on the reference at every slenderness, while the plain
# hierarchic elements membrane-lock (collapsing as the shell thins). The standard RM-5p
# **shear-locks**: even its MMS variant (membrane fixed) keeps falling away, because the shear
# locking remains.

# %%
def save_rows(rows, path):
    """Write study result rows to ``path`` as CSV (columns from the first row's keys)."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {path}")


DEG = 3
NEL_FIXED = 4
SLENDERNESS = (10, 100, 1000, 10000)

FORMULATIONS = (
    ("ShellReissnerMindlinHier5p",    ck.ShellReissnerMindlinHier5p, False),
    ("ShellReissnerMindlinHier5pMMS", ck.ShellReissnerMindlinHier5p, True),
    ("ShellReissnerMindlinHier4p",    ck.ShellReissnerMindlinHier4p, False),
    ("ShellReissnerMindlinHier4pMMS", ck.ShellReissnerMindlinHier4p, True),
    ("ShellReissnerMindlin5p",        ck.ShellReissnerMindlin5p,     False),
    ("ShellReissnerMindlin5pMMS",     ck.ShellReissnerMindlin5p,     True),
)
print(f"\nStudy 1 - Tip deflection vs slenderness (nel={NEL_FIXED}, p={DEG}, w_ref = {WREF:.6f})")

slender_rows = []
for elem_name, elem_cls, use_mms in FORMULATIONS:
    print(f"\n================ {elem_name} ================")
    print(f"{'R/t':>8} {'ndof':>8} {'|w|':>14} {'err %':>10}")
    for ratio in SLENDERNESS:
        t = R / ratio
        w, ndof = solve_strip(DEG, NEL_FIXED, t, elem_cls, assumed_strain=use_mms)
        err = 100.0 * (w - WREF) / WREF
        print(f"{ratio:>8} {ndof:>8} {w:>14.6e} {err:>9.3f}%", flush=True)
        slender_rows.append({"element": elem_name, "ratio": ratio, "t": t, "deg": DEG,
                             "nel": NEL_FIXED, "ndof": ndof, "w_abs": w,
                             "w_ref": WREF, "w_err_pct": err})

save_rows(slender_rows, os.path.join(OUT_DIR, "results_slenderness.csv"))

# %% [markdown]
# <img src="cylindrical_strip/strip_locking.pdf" width="560" align="center" alt="Clamped cylindrical strip: radial tip deflection |w| vs slenderness R/t at a fixed coarse mesh for six formulations (RM-Hier-4p, RM-Hier-5p, RM-5p, each plain and MMS); the hierarchic MMS variants track the analytical reference, the plain hierarchic elements membrane-lock, and the standard RM-5p shear-locks even with MMS.">

# %% [markdown]
# ## Study 2 — Convergence under refinement
#
# At two slendernesses — $R/t=100$ (mild locking) and $R/t=10000$ (severe) — refine the arc mesh from
# a coarse $n_{el}=2$ and plot $|w|/w^{\mathrm{ref}}$ against the number of degrees of freedom, shown
# as two subplots. The DOF axis makes it a cost comparison across the 4- and 5-DOF elements: the
# 4-DOF RM-Hier-4p reaches a given accuracy at fewer DOFs (its curve sits to the left of the 5-DOF
# elements). The MMS forms converge fastest (membrane-locking-free); the plain hierarchic elements
# lag, and the standard RM-5p is slowest (shear locking). The thin shell ($R/t=10000$) needs many
# more DOFs than the mild one.
#
# $N_{\mathrm{dof}}$ counts the **physical** control-point DOFs only — the MMS variants' extra
# assumed-membrane-strain auxiliary DOFs are not included. (Exact DOF-matching across the 4-/5-DOF
# elements is dropped here: its coarsest match needs a single 5-DOF element, which is anomalous.)

# %%
CONV_RATIOS = (100, 10000)                                       # mild and severe locking
NEL_BY_RATIO = {100: (2, 3, 4, 6, 8, 12), 10000: (2, 4, 8, 16, 32, 64)}   # coarse -> fine, per ratio
print(f"\nStudy 2 - Convergence under refinement (R/t={CONV_RATIOS}, p={DEG}, w_ref = {WREF:.6f})")

conv_rows = []
for ratio in CONV_RATIOS:
    t = R / ratio
    for elem_name, elem_cls, use_mms in FORMULATIONS:
        print(f"\n======== R/t={ratio}   {elem_name} ========")
        print(f"{'nel':>5} {'ndof':>8} {'|w|':>14} {'|w|/wref':>10}")
        for nel in NEL_BY_RATIO[ratio]:
            try:
                w, ndof = solve_strip(DEG, nel, t, elem_cls, assumed_strain=use_mms)
            except Exception as exc:                            # singular on the coarsest meshes
                print(f"{nel:>5}   skipped ({type(exc).__name__})", flush=True)
                continue
            print(f"{nel:>5} {ndof:>8} {w:>14.6e} {w / WREF:>10.4f}", flush=True)
            conv_rows.append({"element": elem_name, "ratio": ratio, "deg": DEG, "nel": nel,
                              "ndof": ndof, "w_abs": w, "w_ref": WREF, "w_over_ref": w / WREF})

save_rows(conv_rows, os.path.join(OUT_DIR, "results_convergence.csv"))

# %% [markdown]
# <img src="cylindrical_strip/strip_convergence.pdf" width="760" align="center" alt="Clamped cylindrical strip: normalized tip deflection |w|/w_ref vs number of DOFs under mesh refinement, two subplots for R/t=100 and R/t=10000, six formulations; RM-Hier-4p reaches accuracy at fewer DOFs than RM-Hier-5p, the standard RM-5p is slowest (shear locking), and the MMS variants converge fastest; the thin shell needs many more DOFs.">

# %% [markdown]
# ## ParaView export
#
# Solve the strip on a coarse mesh ($n_{el}=10$, $p=3$) at three slendernesses $R/t=100,1000,10000$
# and write each to an exact rational-Bézier `.vtu` (ParaView 5.9+), both **with**
# ($u\tilde\varepsilon$) and **without** ($u$) the mixed-strain fix — six files in all. Membrane
# locking shows up directly in the **membrane force** $n^{11}$: the pure-displacement element develops
# large spurious membrane-force oscillations (the parasitic membrane energy that stiffens it, growing
# as the shell thins), while the mixed-strain element gives a clean, near-zero membrane force — this
# is a pure bending problem. Open in ParaView, **warp by `displacement`**, and colour by `n11`.
#
# For the mixed element the membrane block is suppressed, so $n^{ab}$ comes from the wrapper's own
# `TRACTION` evaluated on the **untruncated** solution `ck.solve(prob, full=True)` (which carries the
# assumed-strain field) — a plain displacement `TRACTION` would lock.

# %%
deg, nel = 3, 10
gauss1 = ck.GaussLegendre(deg + 1, dim=1)


def export_strip(t: float, assumed_strain: bool, path: str) -> None:
    """Solve and export one strip. With ``assumed_strain`` the base shell is wrapped in a
    ``MixedMembraneStrainShell``; its membrane forces n^{ab} come from the wrapper's ``TRACTION`` on
    the **full** solution (the assumed-strain field — a plain displacement TRACTION would lock)."""
    patch = strip_patch(deg, nel)
    base = ck.ShellReissnerMindlinHier4p(ck.PlaneStress2d(E, NU, t))
    gauss2 = ck.GaussLegendre(deg + 1, dim=2)
    element = ck.MixedMembraneStrainShell(patch, base, gauss2) if assumed_strain else base
    prob = ck.LinearElasticProblem([patch], element, gauss2)
    radial_load(prob, patch, gauss1, t)
    clamp_edge(prob, patch, gauss1)
    if isinstance(base, ck.ShellReissnerMindlinHier4p):           # pin its constant-psi null mode (slot 3)
        prob.add_constraint(ck.DirectConstraint([3], value=0.0))
    u_full = ck.solve(prob, full=True)
    u = u_full[:prob.num_physical_dofs]

    is_mixed = isinstance(element, ck.MixedMembraneStrainShell)
    u_mix = u_full if is_mixed else u            # TRACTION's membrane part is sourced from aux DOFs

    disp     = ck.Function(u, element, patch, ck.FieldType.DISPLACEMENT)
    moment   = ck.Function(u, element, patch, ck.FieldType.MOMENT)        # [m11, m22, m12]
    traction = ck.Function(u_mix, element, patch, ck.FieldType.TRACTION)  # [n11, n22, n12, q1, q2]

    fields = {
        "displacement": disp,
        "n11": traction[0], "n22": traction[1], "n12": traction[2],
        "m11": moment[0],   "m22": moment[1],   "m12": moment[2],
    }
    with ck.BezierVtuWriter(path) as writer:
        writer.add(patch, functions=fields)


for ratio in (100, 1000, 10000):
    t = R / ratio
    export_strip(t, False, os.path.join(OUT_DIR, f"strip_u_t{ratio}.vtu"))
    export_strip(t, True,  os.path.join(OUT_DIR, f"strip_ue_t{ratio}.vtu"))

# %% [markdown]
# <img src="cylindrical_strip/cylindrical_strip_uue_grid_preview.svg" width="640" align="center" alt="Clamped cylindrical strip, membrane force n11 coloured on the deformed shell: top row RM-Hier-4p (plain) at R/t=100, 1000, 10000 showing spurious membrane-force oscillations that grow as the shell thins; bottom row RM-Hier-4p-MMS, clean and oscillation-free.">

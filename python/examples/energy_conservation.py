"""
SimCraft Example: Energy Conservation Check
===========================================

Zero-gravity scene with a pre-deformed elastic cube given an initial velocity.
With no external forces and no collisions, total mechanical energy (KE + PE)
should be conserved throughout the simulation.

Scene setup:
  - Single NeoHookean cube, pre-squashed along y (initial elastic PE)
  - Uniform initial velocity in x-direction (initial KE)
  - Zero gravity, no kinematic bodies
  - Energy logged every step; drift ΔE/E₀ printed to console

This test exposes:
  - Numerical dissipation of the time integrator (implicit Euler always dissipates)
  - Whether the elastic model returns to rest correctly
  - Momentum conservation: centre-of-mass velocity stays constant

Usage:
    python examples/energy_conservation.py           # with rendering
    python examples/energy_conservation.py --no-render
"""
import sys
import numpy as np

try:
    import simcraft
except ImportError:
    raise ImportError(
        "Cannot import simcraft. Install first:\n"
        "  python dev_setup.py                     (developer: after CLion build)\n"
        "  pip install .                            (user: from VS Dev Prompt)\n"
        "See python/README.md for details."
    )

from pathlib import Path

NO_RENDER = "--no-render" in sys.argv

print(f"simcraft {simcraft.__version__}")
print("=" * 60)
print("Energy Conservation Check")
print("  gravity    : [0, 0, 0]  (zero)")
print("  deformation: cube squashed to 30% height along y")
print("  velocity   : [5.0, 1.5, 0.0] m/s (uniform on all vertices)")
print("=" * 60)

# ─── 1. Mesh ────────────────────────────────────────────────────────────────
mesh_path = (Path(__file__).resolve().parents[2]
             / "FEM" / "assets" / "tets" / "cube10x10.tobj")
mesh = simcraft.TetMesh.from_file(str(mesh_path))
print(f"Mesh: {mesh.num_vertices} vertices, {mesh.num_elements} tets")

# ─── 2. Pre-deform the mesh ─────────────────────────────────────────────────
# Squash along y by 70 %, stretch along x by 40 %.
# This gives the body non-zero elastic PE at t = 0.
verts = mesh.vertices.copy()          # (N, 3) – rest-shape vertices
centre = verts.mean(axis=0)

scale = np.array([1.4, 0.3, 1.0])    # 强压缩 y（70%），明显拉伸 x → 大弹性势能
verts_deformed = centre + (verts - centre) * scale

# Uniform initial velocity on every vertex — 够大才能在渲染窗口里明显看到平移
v0 = np.array([5.0, 1.5, 0.0])       # m/s
init_vel = np.tile(v0, (verts.shape[0], 1))

# Build TetMesh: rest vertices define X, initial_positions define x at t=0.
# This cleanly separates rest shape from pre-deformation — no manual hacks needed.
mesh_predeformed = simcraft.TetMesh(verts, mesh.elements,
                                    velocities=init_vel,
                                    initial_positions=verts_deformed)

# ─── 3. Material ────────────────────────────────────────────────────────────
material = simcraft.NeoHookean(young=1e5, poisson=0.4)

# ─── 4. System ──────────────────────────────────────────────────────────────
system = simcraft.System()
system.add_elastic_body(mesh_predeformed, material, density=1000.0,
                        color=(0.40, 0.85, 0.55))
system.gravity = np.array([0.0, 0.0, 0.0])   # zero gravity, no external work
# No kinematic bodies / colliders → pure free motion

# ─── 5. Integrator ──────────────────────────────────────────────────────────
integrator = simcraft.IpcIntegrator(dHat=1e-3, kappa=1e8)

# ─── 6. Simulation parameters ───────────────────────────────────────────────
DT    = 0.005   # 较小 dt：隐式欧拉耗散更小，能量守恒曲线更平
STEPS = 400     # 总时长 2 s，足够看到多次弹性振荡

# ─── 7. Run ─────────────────────────────────────────────────────────────────
sim = simcraft.Simulation(system, integrator)

HDR = f"{'Step':>6}  {'t':>8}  {'KE':>14}  {'PE':>14}  {'E_total':>14}  {'ΔE/E₀':>10}"
SEP = "-" * 74

if NO_RENDER:
    # ── Headless: manual step loop with per-step energy logging ──────────────
    print(f"\n{HDR}\n{SEP}")

    E0 = None
    drift = 0.0
    for step in range(STEPS):
        sim.step(DT)

        ke = sim.kinetic_energy
        pe = sim.potential_energy
        E  = ke + pe

        if E0 is None:
            E0 = E if abs(E) > 1e-12 else 1.0

        drift = (E - E0) / abs(E0)

        if step % 20 == 0 or step < 5:
            t = sim.time
            print(f"{step:>6d}  {t:>8.4f}  {ke:>14.6f}  {pe:>14.6f}"
                  f"  {E:>14.6f}  {drift:>+10.4%}")

    print(SEP)
    print(f"Final energy drift: {drift:+.4%}")
    print("Note: Implicit Euler is dissipative — some negative drift is expected.")
    print(f"Done. {sim.steps_completed} steps completed.")

else:
    # ── Render mode with per-step energy callback ────────────────────────────
    # The new on_step callback lets us log energy even during rendered runs.
    print(f"\nStarting... close window to stop.")
    print(HDR)

    E0_container = [None]  # mutable container for closure

    def on_step(step, t):
        """Per-step callback: log energy drift while rendering."""
        ke = sim.kinetic_energy
        pe = sim.potential_energy
        E  = ke + pe

        if E0_container[0] is None:
            E0_container[0] = E if abs(E) > 1e-12 else 1.0

        drift = (E - E0_container[0]) / abs(E0_container[0])

        if step % 20 == 0 or step <= 5:
            print(f"{step:>6d}  {t:>8.4f}  {ke:>14.6f}  {pe:>14.6f}"
                  f"  {E:>14.6f}  {drift:>+10.4%}")

    sim.display(dt=DT, steps=STEPS,
                title="Energy Conservation Check",
                on_step=on_step)

    # Final report
    ke = sim.kinetic_energy
    pe = sim.potential_energy
    E  = ke + pe
    E0 = E0_container[0] or 1.0
    drift = (E - E0) / abs(E0)
    print(SEP)
    print(f"Final energy drift: {drift:+.4%}")
    print("Note: Implicit Euler is dissipative — some negative drift is expected.")
    print(f"Done. {sim.steps_completed} steps.")

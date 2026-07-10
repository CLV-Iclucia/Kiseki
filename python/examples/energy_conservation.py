"""
Kiseki Example: Energy Conservation Check
===========================================

Zero-gravity scene with a pre-deformed elastic cube given an initial velocity.
With no external forces and no collisions, total mechanical energy (KE + PE)
should be conserved throughout the simulation.

Scene setup:
  - Single NeoHookean cube, pre-squashed along y (initial elastic PE)
  - Uniform initial velocity in x-direction (initial KE)
  - Zero gravity, no kinematic bodies
  - Energy logged every step; drift 螖E/E鈧€ printed to console

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
    import kiseki
except ImportError:
    raise ImportError(
        "Cannot import kiseki. Install first:\n"
        "  python dev_setup.py                     (developer: after CLion build)\n"
        "  pip install .                            (user: from VS Dev Prompt)\n"
        "See python/README.md for details."
    )

from pathlib import Path

NO_RENDER = "--no-render" in sys.argv

print(f"kiseki {kiseki.__version__}")
print("=" * 60)
print("Energy Conservation Check")
print("  gravity    : [0, 0, 0]  (zero)")
print("  deformation: cube squashed to 30% height along y")
print("  velocity   : [5.0, 1.5, 0.0] m/s (uniform on all vertices)")
print("=" * 60)

# 鈹€鈹€鈹€ 1. Mesh 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
mesh_path = (Path(__file__).resolve().parents[2]
             / "FEM" / "assets" / "tets" / "cube10x10.tobj")
mesh = kiseki.TetMesh.from_file(str(mesh_path))
print(f"Mesh: {mesh.num_vertices} vertices, {mesh.num_elements} tets")

# 鈹€鈹€鈹€ 2. Pre-deform the mesh 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
# Squash along y by 70 %, stretch along x by 40 %.
# This gives the body non-zero elastic PE at t = 0.
verts = mesh.vertices.copy()          # (N, 3) 鈥?rest-shape vertices
centre = verts.mean(axis=0)

scale = np.array([1.4, 0.3, 1.0])    # 寮哄帇缂?y锛?0%锛夛紝鏄庢樉鎷変几 x 鈫?澶у脊鎬у娍鑳?
verts_deformed = centre + (verts - centre) * scale

# Uniform initial velocity on every vertex 鈥?澶熷ぇ鎵嶈兘鍦ㄦ覆鏌撶獥鍙ｉ噷鏄庢樉鐪嬪埌骞崇Щ
v0 = np.array([5.0, 1.5, 0.0])       # m/s
init_vel = np.tile(v0, (verts.shape[0], 1))

# Build TetMesh: rest vertices define X, initial_positions define x at t=0.
# This cleanly separates rest shape from pre-deformation 鈥?no manual hacks needed.
mesh_predeformed = kiseki.TetMesh(verts, mesh.elements,
                                    velocities=init_vel,
                                    initial_positions=verts_deformed)

# 鈹€鈹€鈹€ 3. Material 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
material = kiseki.NeoHookean(young=1e5, poisson=0.4)

# 鈹€鈹€鈹€ 4. System 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
system = kiseki.System()
system.add_elastic_body(mesh_predeformed, material, density=1000.0,
                        color=(0.40, 0.85, 0.55))
system.gravity = np.array([0.0, 0.0, 0.0])   # zero gravity, no external work
# No kinematic bodies / colliders 鈫?pure free motion

# 鈹€鈹€鈹€ 5. Integrator 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
integrator = kiseki.IpcIntegrator(dHat=1e-3, kappa=1e8)

# 鈹€鈹€鈹€ 6. Simulation parameters 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
DT    = 0.005   # 杈冨皬 dt锛氶殣寮忔鎷夎€楁暎鏇村皬锛岃兘閲忓畧鎭掓洸绾挎洿骞?
STEPS = 400     # 鎬绘椂闀?2 s锛岃冻澶熺湅鍒板娆″脊鎬ф尟鑽?

# 鈹€鈹€鈹€ 7. Run 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
sim = kiseki.Simulation(system, integrator)

HDR = f"{'Step':>6}  {'t':>8}  {'KE':>14}  {'PE':>14}  {'E_total':>14}  {'螖E/E鈧€':>10}"
SEP = "-" * 74

if NO_RENDER:
    # 鈹€鈹€ Headless: manual step loop with per-step energy logging 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
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
    print("Note: Implicit Euler is dissipative 鈥?some negative drift is expected.")
    print(f"Done. {sim.steps_completed} steps completed.")

else:
    # 鈹€鈹€ Render mode with per-step energy callback 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
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
    print("Note: Implicit Euler is dissipative 鈥?some negative drift is expected.")
    print(f"Done. {sim.steps_completed} steps.")

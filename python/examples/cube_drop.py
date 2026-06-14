"""
SimCraft Example: Cube Free Fall
================================

A cube mesh falls under gravity and collides with a ground plane.
Demonstrates: mesh → material → system → kinematic body → integrator → display.

Usage:
    python examples/cube_drop.py

Note:
    Ensure simcraft.pyd is on PYTHONPATH before running.
    E.g.:  set PYTHONPATH=G:\\SimCraft\\cmake-build-release
"""
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

print(f"simcraft {simcraft.__version__}")

# ─── 1. Mesh ────────────────────────────────────────────────────────────────
mesh_path = Path(__file__).resolve().parents[2] / "FEM" / "assets" / "tets" / "cube10x10.tobj"
mesh = simcraft.TetMesh.from_file(str(mesh_path))
print(f"Mesh: {mesh.num_vertices} vertices, {mesh.num_elements} tets")

# ─── 2. Material ────────────────────────────────────────────────────────────
material = simcraft.NeoHookean(young=1e5, poisson=0.4)

# ─── 3. System ──────────────────────────────────────────────────────────────
system = simcraft.System()
system.add_elastic_body(mesh, material, density=1000.0, color=(0.40, 0.70, 0.95))
system.gravity = np.array([0.0, -9.81, 0.0])

# Ground plane at y = -1
ground = simcraft.KinematicBody.plane(
    normal=np.array([0.0, 1.0, 0.0]),
    offset=-1.0
)
system.add_kinematic_body(ground)

# ─── 4. Integrator ──────────────────────────────────────────────────────────
integrator = simcraft.IpcIntegrator(dHat=1e-3, kappa=1e8)

# ─── 5. Run ─────────────────────────────────────────────────────────────────
sim = simcraft.Simulation(system, integrator)

print("Starting... close window to stop.")
sim.display(dt=0.01, steps=500, title="Cube Drop")
print(f"Done. {sim.steps_completed} steps.")

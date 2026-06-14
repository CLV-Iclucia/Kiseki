"""
SimCraft Example: Double Plane Squeeze
=======================================

An elastic cube is squeezed between two planes approaching from above and below.
Tests IPC barrier under compression from both sides.

Demonstrates:
  - Two kinematic bodies with opposing constant velocities
  - IPC barrier preventing penetration from both directions
  - Zero gravity (pure mechanical squeeze)
  - Real-time rendering

Usage:
    python examples/squeeze.py
"""
import numpy as np

try:
    import simcraft
except ImportError:
    raise ImportError(
        "Cannot import simcraft. Install first:\n"
        "  python dev_setup.py\n"
        "  pip install .\n"
    )

from pathlib import Path

print(f"simcraft {simcraft.__version__}")

# ─── 1. Mesh ────────────────────────────────────────────────────────────────
mesh_path = Path(__file__).resolve().parents[2] / "FEM" / "assets" / "tets" / "cube10x10.tobj"
mesh = simcraft.TetMesh.from_file(str(mesh_path))
print(f"Cube: {mesh.num_vertices} vertices, {mesh.num_elements} tets")

# ─── 2. Material (soft, near incompressible) ────────────────────────────────
material = simcraft.NeoHookean(young=1e4, poisson=0.48)

# ─── 3. System (no gravity) ─────────────────────────────────────────────────
system = simcraft.System()
system.add_elastic_body(mesh, material, density=1000.0, color=(0.30, 0.85, 0.55))
system.gravity = np.array([0.0, 0.0, 0.0])

# Top plane moving down
top_plane = simcraft.KinematicBody.plane(
    normal=np.array([0.0, -1.0, 0.0]),
    offset=-2.0
)
top_plane.set_constant_velocity(np.array([0.0, -0.3, 0.0]))
system.add_kinematic_body(top_plane)

# Bottom plane moving up
bottom_plane = simcraft.KinematicBody.plane(
    normal=np.array([0.0, 1.0, 0.0]),
    offset=-2.0
)
bottom_plane.set_constant_velocity(np.array([0.0, 0.3, 0.0]))
system.add_kinematic_body(bottom_plane)

# ─── 4. Integrator (high barrier stiffness) ─────────────────────────────────
integrator = simcraft.IpcIntegrator(dHat=5e-3, kappa=1e10)

# ─── 5. Run with rendering ──────────────────────────────────────────────────
sim = simcraft.Simulation(system, integrator)

print("Starting... close window to stop.")
print("Two planes approach from y=±2 at 0.3 m/s each")
sim.display(dt=0.005, steps=400, title="Double Plane Squeeze")
print(f"Done. {sim.steps_completed} steps.")

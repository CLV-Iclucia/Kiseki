"""
SimCraft Example: Rotating Obstacle
=====================================

A soft cube falls under gravity near a rotating kinematic plane.
Tests contact with moving obstacle and asymmetric force response.

Demonstrates:
  - KinematicBody with set_rotation() (spinning wall)
  - Contact forces between elastic body and rotating obstacle
  - Combined gravity + collision + rotation dynamics

Usage:
    python examples/rotating_obstacle.py
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

# ─── 2. Material ────────────────────────────────────────────────────────────
material = simcraft.NeoHookean(young=1e5, poisson=0.4)

# ─── 3. System ──────────────────────────────────────────────────────────────
system = simcraft.System()
system.add_elastic_body(mesh, material, density=1000.0, color=(0.95, 0.75, 0.25))
system.gravity = np.array([0.0, -9.81, 0.0])

# Static ground
ground = simcraft.KinematicBody.plane(
    normal=np.array([0.0, 1.0, 0.0]),
    offset=-2.0
)
system.add_kinematic_body(ground)

# Rotating wall (spins around z-axis)
rotating_wall = simcraft.KinematicBody.plane(
    normal=np.array([1.0, 0.0, 0.0]),
    offset=-1.5
)
rotating_wall.set_rotation(
    axis=np.array([0.0, 0.0, 1.0]),
    center=np.array([0.0, 0.0, 0.0]),
    omega=2.0
)
system.add_kinematic_body(rotating_wall)

# ─── 4. Run with rendering ──────────────────────────────────────────────────
integrator = simcraft.IpcIntegrator(dHat=2e-3, kappa=1e9)
sim = simcraft.Simulation(system, integrator)

print("Starting... close window to stop.")
sim.display(dt=0.005, steps=500, title="Rotating Obstacle")
print(f"Done. {sim.steps_completed} steps.")

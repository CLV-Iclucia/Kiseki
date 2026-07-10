"""
Kiseki Example: Rotating Obstacle
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
    import kiseki
except ImportError:
    raise ImportError(
        "Cannot import kiseki. Install first:\n"
        "  python dev_setup.py\n"
        "  pip install .\n"
    )

from pathlib import Path

print(f"kiseki {kiseki.__version__}")

# 鈹€鈹€鈹€ 1. Mesh 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
mesh_path = Path(__file__).resolve().parents[2] / "FEM" / "assets" / "tets" / "cube10x10.tobj"
mesh = kiseki.TetMesh.from_file(str(mesh_path))
print(f"Cube: {mesh.num_vertices} vertices, {mesh.num_elements} tets")

# 鈹€鈹€鈹€ 2. Material 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
material = kiseki.NeoHookean(young=1e5, poisson=0.4)

# 鈹€鈹€鈹€ 3. System 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
system = kiseki.System()
system.add_elastic_body(mesh, material, density=1000.0, color=(0.95, 0.75, 0.25))
system.gravity = np.array([0.0, -9.81, 0.0])

# Static ground
ground = kiseki.KinematicBody.plane(
    normal=np.array([0.0, 1.0, 0.0]),
    offset=-2.0
)
system.add_kinematic_body(ground)

# Rotating wall (spins around z-axis)
rotating_wall = kiseki.KinematicBody.plane(
    normal=np.array([1.0, 0.0, 0.0]),
    offset=-1.5
)
rotating_wall.set_rotation(
    axis=np.array([0.0, 0.0, 1.0]),
    center=np.array([0.0, 0.0, 0.0]),
    omega=2.0
)
system.add_kinematic_body(rotating_wall)

# 鈹€鈹€鈹€ 4. Run with rendering 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
integrator = kiseki.IpcIntegrator(dHat=2e-3, kappa=1e9)
sim = kiseki.Simulation(system, integrator)

print("Starting... close window to stop.")
sim.display(dt=0.005, steps=500, title="Rotating Obstacle")
print(f"Done. {sim.steps_completed} steps.")

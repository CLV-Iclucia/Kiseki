"""
Kiseki Example: Double Plane Squeeze
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

# 鈹€鈹€鈹€ 2. Material (soft, near incompressible) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
material = kiseki.NeoHookean(young=1e4, poisson=0.48)

# 鈹€鈹€鈹€ 3. System (no gravity) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
system = kiseki.System()
system.add_elastic_body(mesh, material, density=1000.0, color=(0.30, 0.85, 0.55))
system.gravity = np.array([0.0, 0.0, 0.0])

# Top plane moving down
top_plane = kiseki.KinematicBody.plane(
    normal=np.array([0.0, -1.0, 0.0]),
    offset=-2.0
)
top_plane.set_constant_velocity(np.array([0.0, -0.3, 0.0]))
system.add_kinematic_body(top_plane)

# Bottom plane moving up
bottom_plane = kiseki.KinematicBody.plane(
    normal=np.array([0.0, 1.0, 0.0]),
    offset=-2.0
)
bottom_plane.set_constant_velocity(np.array([0.0, 0.3, 0.0]))
system.add_kinematic_body(bottom_plane)

# 鈹€鈹€鈹€ 4. Integrator (high barrier stiffness) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
integrator = kiseki.IpcIntegrator(dHat=5e-3, kappa=1e10)

# 鈹€鈹€鈹€ 5. Run with rendering 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
sim = kiseki.Simulation(system, integrator)

print("Starting... close window to stop.")
print("Two planes approach from y=卤2 at 0.3 m/s each")
sim.display(dt=0.005, steps=400, title="Double Plane Squeeze")
print(f"Done. {sim.steps_completed} steps.")

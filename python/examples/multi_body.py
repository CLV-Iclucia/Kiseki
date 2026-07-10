"""
Kiseki Example: Multi-Body Collision (Head-On Impact)
=======================================================

Two cubes fly toward each other in zero gravity and collide.
Tests elastic-elastic IPC collision with initial velocities.

Demonstrates:
  - Multiple elastic bodies with initial velocities
  - Zero gravity (pure momentum exchange)
  - Elastic-elastic collision via IPC
  - Per-body colors for visual distinction

Usage:
    python examples/multi_body.py
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

# 鈹€鈹€鈹€ Setup 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
mesh_path = Path(__file__).resolve().parents[2] / "FEM" / "assets" / "tets" / "cube10x10.tobj"

material = kiseki.NeoHookean(young=2e5, poisson=0.4)
system = kiseki.System()
system.gravity = np.array([0.0, 0.0, 0.0])  # zero gravity

# Load template mesh
template = kiseki.TetMesh.from_file(str(mesh_path))
base_verts = template.vertices  # (N, 3)
base_tets = template.elements   # (M, 4)

# Compute mesh width for spacing
x_min, x_max = base_verts[:, 0].min(), base_verts[:, 0].max()
width = x_max - x_min
separation = 0.05  # initial gap between the two cubes

# Two cubes facing each other along x-axis, flying inward
speed = 2.0  # m/s

body_colors = [
    (0.40, 0.70, 0.95),  # sky blue  (left cube, moves right)
    (0.95, 0.55, 0.35),  # coral     (right cube, moves left)
]

n_verts = base_verts.shape[0]

# Left cube: offset to the left, velocity 鈫?right (+x)
left_offset = np.array([-(width / 2 + separation / 2), 0.0, 0.0])
left_vel = np.full((n_verts, 3), [speed, 0.0, 0.0])
mesh_left = kiseki.TetMesh(base_verts + left_offset, base_tets, velocities=left_vel)
system.add_elastic_body(mesh_left, material, density=800.0, color=body_colors[0])
print(f"  Left cube:  {n_verts} verts, vel = [+{speed}, 0, 0]")

# Right cube: offset to the right, velocity 鈫?left (-x)
right_offset = np.array([+(width / 2 + separation / 2), 0.0, 0.0])
right_vel = np.full((n_verts, 3), [-speed, 0.0, 0.0])
mesh_right = kiseki.TetMesh(base_verts + right_offset, base_tets, velocities=right_vel)
system.add_elastic_body(mesh_right, material, density=800.0, color=body_colors[1])
print(f"  Right cube: {n_verts} verts, vel = [-{speed}, 0, 0]")

print(f"Total bodies: {system.num_bodies}")

# 鈹€鈹€鈹€ Run with rendering 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
integrator = kiseki.IpcIntegrator(dHat=2e-3, kappa=1e9)
sim = kiseki.Simulation(system, integrator)

print("Starting... close window to stop.")
sim.display(dt=0.005, steps=500, title="Head-On Collision")
print(f"Done. {sim.steps_completed} steps.")

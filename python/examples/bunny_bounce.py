"""
Kiseki Example: Hanging Bunny
================================

Stanford bunny hangs from its ears (top vertices pinned) and
deforms under gravity like a soft jelly.

Demonstrates:
  - Automatic selection of top-face vertices via coordinate query
  - Pin constraints on a subset of vertices
  - Large soft-body deformation under gravity
  - Real-time rendering

Usage:
    python examples/bunny_bounce.py
"""
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

print(f"kiseki {kiseki.__version__}")

# 鈹€鈹€鈹€ Setup 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
mesh_path = Path(__file__).resolve().parents[2] / "FEM" / "assets" / "tets" / "bunny.tobj"
mesh = kiseki.TetMesh.from_file(str(mesh_path))
print(f"Bunny: {mesh.num_vertices} vertices, {mesh.num_elements} tets")

# Get vertex positions before add_elastic_body moves the mesh
verts = mesh.vertices

material = kiseki.NeoHookean(young=1e4, poisson=0.45)

system = kiseki.System()
system.add_elastic_body(mesh, material, density=1000.0, color=(0.95, 0.55, 0.35))
system.gravity = np.array([0.0, -9.81, 0.0])

# Pin the topmost vertices (top 5% by y-coordinate)
y_max = verts[:, 1].max()
y_min = verts[:, 1].min()
y_threshold = y_max - 0.05 * (y_max - y_min)  # top 5%

top_verts = np.where(verts[:, 1] >= y_threshold)[0].astype(np.int32)
system.constraints.pin_vertices(top_verts)
print(f"Pinned {len(top_verts)} vertices at the top (y >= {y_threshold:.3f})")

# 鈹€鈹€鈹€ Run 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
integrator = kiseki.IpcIntegrator(dHat=1e-3, kappa=1e8)
sim = kiseki.Simulation(system, integrator)

print("Running... close window to stop.")
sim.display(dt=0.01, steps=500, title="Hanging Bunny")
print(f"Done. {sim.steps_completed} steps.")

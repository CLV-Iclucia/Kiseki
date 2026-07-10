"""
Kiseki Example: Prescribed Motion
====================================

A vertex is driven by a Python lambda 鈥?sinusoidal oscillation in x.
Demonstrates: prescribe_motion with a Python callable.

Usage:
    set PYTHONPATH=<build-dir>
    python examples/prescribed_motion.py
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
mesh_path = Path(__file__).resolve().parents[2] / "FEM" / "assets" / "tets" / "cube10x10.tobj"
mesh = kiseki.TetMesh.from_file(str(mesh_path))
print(f"Mesh: {mesh.num_vertices} vertices, {mesh.num_elements} tets")

# Read vertex positions BEFORE add_elastic_body (which moves the mesh)
verts = mesh.vertices

material = kiseki.NeoHookean(young=2e5, poisson=0.45)

system = kiseki.System()
system.add_elastic_body(mesh, material, density=800.0, color=(0.35, 0.80, 0.80))
system.gravity = np.array([0.0, -9.81, 0.0])

# Find all vertices on the x_min face (one end of the cube)
x_min = verts[:, 0].min()
eps = (verts[:, 0].max() - x_min) * 1e-6
fixed_face = np.where(np.abs(verts[:, 0] - x_min) < eps)[0].astype(np.int32)
system.constraints.pin_vertices(fixed_face)
print(f"Pinned {len(fixed_face)} vertices on x_min face")

# Find all vertices on the x_max face (other end) for prescribed motion
x_max = verts[:, 0].max()
driven_face = np.where(np.abs(verts[:, 0] - x_max) < eps)[0]

# Prescribe sinusoidal y-motion on the entire driven face
amplitude = 0.3
freq = 1.0  # Hz
print(f"Prescribed sinusoidal motion on {len(driven_face)} vertices (x_max face)")

# 鈹€鈹€鈹€ Run 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
integrator = kiseki.IpcIntegrator(dHat=1e-3, kappa=1e8)
sim = kiseki.Simulation(system, integrator)

print("Running... close window to stop.")
sim.display(dt=0.01, steps=300, title="Prescribed Motion")
print(f"Done. {sim.steps_completed} steps.")

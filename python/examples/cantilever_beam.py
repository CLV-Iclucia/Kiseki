"""
SimCraft Example: Cantilever Beam
==================================

A rectangular beam (mat2x2) is fixed at one end and deforms under gravity.
Classic validation scene for constraint + elastic system correctness.

Demonstrates:
  - Pin constraints on one end
  - Gravity-driven deformation
  - Real-time rendering of deformation process

Usage:
    python examples/cantilever_beam.py
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
mesh_path = Path(__file__).resolve().parents[2] / "FEM" / "assets" / "tets" / "cube2x2.tobj"
mesh = simcraft.TetMesh.from_file(str(mesh_path))
print(f"Beam mesh: {mesh.num_vertices} vertices, {mesh.num_elements} tets")

# ─── 2. Material (relatively stiff) ─────────────────────────────────────────
material = simcraft.NeoHookean(young=1e6, poisson=0.3)

# ─── 3. System ──────────────────────────────────────────────────────────────
system = simcraft.System()
system.add_elastic_body(mesh, material, density=1000.0, color=(0.85, 0.45, 0.65))
system.gravity = np.array([0.0, -9.81, 0.0])

# Fix one end: pin the first few vertices (left end of beam)
fixed_verts = np.array([0, 1, 2, 3], dtype=np.int32)
system.constraints.pin_vertices(fixed_verts)
print(f"Pinning {len(fixed_verts)} vertices at fixed end")

# ─── 4. Integrator ──────────────────────────────────────────────────────────
integrator = simcraft.IpcIntegrator(dHat=1e-3, kappa=1e8)

# ─── 5. Run with rendering ──────────────────────────────────────────────────
sim = simcraft.Simulation(system, integrator)

print("Starting... close window to stop.")
sim.display(dt=0.01, steps=300, title="Cantilever Beam")
print(f"Done. {sim.steps_completed} steps.")

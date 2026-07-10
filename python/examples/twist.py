"""
Kiseki Example: Twist Test
=============================

A beam is fixed at one end; the other end is driven by prescribed sinusoidal
motion. Tests large-deformation + prescribed motion with real-time rendering.

Demonstrates:
  - prescribe_motion with Python lambda (sinusoidal z-displacement)
  - Mixed constraints: pin (fixed end) + prescribed motion (driven end)
  - Large deformation under torsion
  - Real-time visualization

Usage:
    python examples/twist.py
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
mesh_path = Path(__file__).resolve().parents[2] / "FEM" / "assets" / "tets" / "mat2x2.tobj"
mesh = kiseki.TetMesh.from_file(str(mesh_path))
print(f"Beam: {mesh.num_vertices} vertices, {mesh.num_elements} tets")

# 鈹€鈹€鈹€ 2. Material 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
material = kiseki.NeoHookean(young=5e5, poisson=0.35)

# 鈹€鈹€鈹€ 3. System (no gravity, pure twist) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
system = kiseki.System()
system.add_elastic_body(mesh, material, density=500.0, color=(0.55, 0.45, 0.90))
system.gravity = np.array([0.0, 0.0, 0.0])

# Fix left end (first 4 vertices)
fixed_verts = np.array([0, 1, 2, 3], dtype=np.int32)
system.constraints.pin_vertices(fixed_verts)
print(f"Fixed end: {len(fixed_verts)} vertices pinned")

# Prescribe sinusoidal z-displacement on right end vertices
n_verts = mesh.num_vertices
driven_verts = [n_verts - 4, n_verts - 3, n_verts - 2, n_verts - 1]
amplitude = 0.3
freq = 0.5  # Hz

for v_idx in driven_verts:
    def make_motion(amp, f):
        def motion_func(t):
            return np.array([0.0, 0.0, amp * np.sin(2.0 * np.pi * f * t)])
        return motion_func

    system.constraints.prescribe_motion(v_idx, make_motion(amplitude, freq))

print(f"Driven end: {len(driven_verts)} vertices, amplitude={amplitude}, freq={freq} Hz")

# 鈹€鈹€鈹€ 4. Integrator 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
integrator = kiseki.IpcIntegrator(dHat=1e-3, kappa=1e8)

# 鈹€鈹€鈹€ 5. Run with rendering 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
sim = kiseki.Simulation(system, integrator)

print("Starting... close window to stop.")
sim.display(dt=0.01, steps=400, title="Twist Test")
print(f"Done. {sim.steps_completed} steps.")

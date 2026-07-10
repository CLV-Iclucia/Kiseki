"""
Mesh Normalization Utility
==========================

Load a tetrahedral mesh, center it at the origin, scale to canonical size,
and save the normalized result.

Usage:
    python scripts/normalize_mesh.py <input.tobj> <output.tobj> [--scale 1.0]

The normalized mesh will be:
  - Centered at origin (mean vertex position = [0, 0, 0])
  - Scaled so max bounding box dimension = specified scale (default 1.0)
"""
import numpy as np
import sys
from pathlib import Path

try:
    import kiseki
except ImportError:
    raise ImportError(
        "Cannot import kiseki. Install first:\n"
        "  python dev_setup.py                     (developer: after CLion build)\n"
        "  pip install .                            (user: from VS Dev Prompt)\n"
        "See python/README.md for details."
    )


def normalize_mesh(input_path: str, output_path: str, target_scale: float = 1.0):
    """
    Load mesh, center it, scale to canonical size, and save.

    Args:
        input_path: Path to input .tobj mesh file
        output_path: Path to write normalized .tobj mesh
        target_scale: Target max bounding box dimension (default 1.0)
    """
    print(f"Loading mesh from {input_path}...")
    mesh = kiseki.TetMesh.from_file(str(input_path))

    # Get vertex positions BEFORE commit (mesh.vertices will be cleared after)
    verts = mesh.vertices.copy()  # (N, 3) numpy array
    tets = mesh.elements           # (M, 4) numpy array

    print(f"  Vertices: {len(verts)}")
    print(f"  Tetrahedra: {len(tets)}")

    # Compute bounding box
    bbox_min = verts.min(axis=0)
    bbox_max = verts.max(axis=0)
    bbox_size = bbox_max - bbox_min
    max_dim = bbox_size.max()
    center = (bbox_min + bbox_max) / 2.0

    print(f"\nOriginal bounding box:")
    print(f"  Min: {bbox_min}")
    print(f"  Max: {bbox_max}")
    print(f"  Size: {bbox_size}")
    print(f"  Center: {center}")
    print(f"  Max dimension: {max_dim:.6f}")

    # Normalize: center at origin, scale to target size
    verts_normalized = (verts - center) * (target_scale / max_dim)

    # Verify normalized bounding box
    bbox_min_norm = verts_normalized.min(axis=0)
    bbox_max_norm = verts_normalized.max(axis=0)
    bbox_size_norm = bbox_max_norm - bbox_min_norm
    max_dim_norm = bbox_size_norm.max()

    print(f"\nNormalized bounding box:")
    print(f"  Min: {bbox_min_norm}")
    print(f"  Max: {bbox_max_norm}")
    print(f"  Size: {bbox_size_norm}")
    print(f"  Max dimension: {max_dim_norm:.6f}")
    print(f"  Center: {verts_normalized.mean(axis=0)}")

    # Create normalized mesh and save
    # Note: TetMesh constructor will consume the vertices and tets
    normalized_mesh = kiseki.TetMesh(verts_normalized, tets)

    # Save to output file
    # For now, we'll use a workaround: create a temporary system just to demonstrate
    # the mesh is valid, then manually write it. Since kiseki doesn't expose
    # mesh writing directly, we'll write a simple .tobj format ourselves.
    print(f"\nWriting normalized mesh to {output_path}...")
    _write_tobj(output_path, verts_normalized, tets)
    print(f"Done!")


def _write_tobj(filepath: str, vertices: np.ndarray, tetrahedra: np.ndarray):
    """
    Write tetrahedral mesh to .tobj format (simple text format).

    Format:
      v x y z              (vertex)
      t i0 i1 i2 i3        (tetrahedron with 0-based vertex indices)
    """
    with open(filepath, 'w') as f:
        # Write vertices
        for i, v in enumerate(vertices):
            f.write(f"v {v[0]:.6e} {v[1]:.6e} {v[2]:.6e}\n")

        # Write tetrahedra (1-based indices in .tobj format)
        for tet in tetrahedra:
            i0, i1, i2, i3 = tet
            f.write(f"t {i0+1} {i1+1} {i2+1} {i3+1}\n")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python normalize_mesh.py <input.tobj> <output.tobj> [--scale 1.0]")
        print("\nExample:")
        print("  python normalize_mesh.py ../FEM/assets/tets/bunny.tobj bunny_normalized.tobj")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    scale = 1.0

    if len(sys.argv) > 3 and sys.argv[3] == "--scale":
        if len(sys.argv) > 4:
            scale = float(sys.argv[4])

    normalize_mesh(input_file, output_file, target_scale=scale)

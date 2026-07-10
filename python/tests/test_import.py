"""Verify kiseki module imports and exposes expected symbols."""
import kiseki
import numpy as np


# Phase 1: Basic import
def test_import_succeeds():
    assert kiseki is not None


def test_version_string():
    assert hasattr(kiseki, "__version__")
    assert kiseki.__version__ == "0.1.0"


def test_hello_placeholder():
    assert kiseki.hello() == "Kiseki is alive!"


# Phase 2: TetMesh
def test_tetmesh_from_numpy():
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    tets = np.array([[0, 1, 2, 3]], dtype=np.int32)
    mesh = kiseki.TetMesh(vertices, tets)
    assert mesh.num_vertices == 4
    assert mesh.num_elements == 1


def test_tetmesh_invalid_shape():
    try:
        kiseki.TetMesh(np.zeros((4, 2)), np.zeros((1, 4), dtype=np.int32))
        assert False, "Should have raised"
    except ValueError:
        pass


# Phase 3: NeoHookean material
def test_neohookean_creation():
    mat = kiseki.NeoHookean(young=1e6, poisson=0.45)
    assert mat.young == 1e6
    assert mat.poisson == 0.45


def test_neohookean_invalid_poisson():
    try:
        kiseki.NeoHookean(young=1e6, poisson=0.5)
        assert False, "Should have raised"
    except ValueError:
        pass


def test_neohookean_negative_young():
    try:
        kiseki.NeoHookean(young=-1.0, poisson=0.3)
        assert False, "Should have raised"
    except ValueError:
        pass


# Phase 4: System
def test_system_creation():
    system = kiseki.System()
    assert system.num_bodies == 0
    assert not system.locked


def test_system_gravity():
    system = kiseki.System()
    system.gravity = np.array([0, -9.81, 0])
    g = system.gravity
    assert abs(g[1] - (-9.81)) < 1e-10


def test_system_add_body():
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    tets = np.array([[0, 1, 2, 3]], dtype=np.int32)
    mesh = kiseki.TetMesh(vertices, tets)
    mat = kiseki.NeoHookean(young=1e6, poisson=0.45)
    system = kiseki.System()
    system.add_elastic_body(mesh, mat, density=1000.0)
    assert system.num_bodies == 1


# Phase 6: KinematicBody
def test_kinematic_plane():
    kb = kiseki.KinematicBody.plane(
        normal=np.array([0, 1, 0], dtype=np.float64),
        offset=-1.0
    )
    assert kb.motion == "static"


def test_kinematic_velocity():
    kb = kiseki.KinematicBody.plane(np.array([0, 1, 0], dtype=np.float64))
    kb.set_constant_velocity(np.array([0, 1, 0], dtype=np.float64))
    assert kb.motion == "constant_velocity"


# Phase 7: IpcIntegrator
def test_integrator_creation():
    intg = kiseki.IpcIntegrator(dHat=1e-3, eps=1e-2, kappa=1e10, stepSizeScale=0.9)
    assert intg.dHat == 1e-3
    assert intg.eps == 1e-2
    assert intg.kappa == 1e10
    assert intg.stepSizeScale == 0.9


def test_integrator_modify():
    intg = kiseki.IpcIntegrator()
    intg.dHat = 2e-3
    assert intg.dHat == 2e-3


def test_integrator_invalid():
    try:
        kiseki.IpcIntegrator(dHat=-1.0)
        assert False, "Should have raised"
    except ValueError:
        pass


# Phase 9: Lock mechanism (tested without actual simulation run)
def test_lock_initial_state():
    system = kiseki.System()
    intg = kiseki.IpcIntegrator()
    sim = kiseki.Simulation(system, intg)
    assert not sim.locked
    assert not system.locked
    assert not intg.locked


if __name__ == "__main__":
    test_import_succeeds()
    test_version_string()
    test_hello_placeholder()
    test_tetmesh_from_numpy()
    test_tetmesh_invalid_shape()
    test_neohookean_creation()
    test_neohookean_invalid_poisson()
    test_neohookean_negative_young()
    test_system_creation()
    test_system_gravity()
    test_system_add_body()
    test_kinematic_plane()
    test_kinematic_velocity()
    test_integrator_creation()
    test_integrator_modify()
    test_integrator_invalid()
    test_lock_initial_state()
    print("All Phase 1-9 checks passed.")

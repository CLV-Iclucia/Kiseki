#include "bindings.h"

#include <DER/der-scene.h>
#include <Runtime/global-solver.h>

#include <glm/glm.hpp>

#include <memory>
#include <stdexcept>
#include <utility>

namespace {

namespace der = ksk::der;
namespace runtime = ksk::runtime;

struct PyDERRod {
  der::DERRodDesc desc;
};

struct PyDERScene {
  runtime::RuntimeSceneDesc scene;
  int rod_count = 0;
  std::vector<int> rod_vertex_counts;
  bool locked = false;

  void checkUnlocked() const
  {
    if (locked) {
      throw std::runtime_error(
          "DERScene is locked after build(). Create a new scene to modify it.");
    }
  }
};

struct PyRuntimeSimulation {
  runtime::SimulationRunner simulation;
  std::vector<int> rod_vertex_counts;

  PyRuntimeSimulation(runtime::SimulationRunner&& runtimeSimulation,
                      std::vector<int> rodVertexCounts)
      : simulation(std::move(runtimeSimulation))
      , rod_vertex_counts(std::move(rodVertexCounts))
  {
  }

  runtime::RuntimeStepResult step()
  {
    return simulation.step();
  }

  runtime::RuntimeStepResult run(int steps)
  {
    if (steps <= 0) {
      throw py::value_error("steps must be positive");
    }
    return simulation.run(steps);
  }
};

glm::dvec3 parseVec3(py::array_t<double> values, const char* name)
{
  const auto buffer = values.unchecked<1>();
  if (buffer.shape(0) != 3) {
    throw py::value_error(std::string(name) + " must have shape (3,)");
  }
  return glm::dvec3(buffer(0), buffer(1), buffer(2));
}

std::vector<der::RodDof> parseRodBlocks(
    py::array_t<double> positions,
    std::optional<py::array_t<double>> theta)
{
  const auto pos = positions.unchecked<2>();
  if (pos.ndim() != 2 || pos.shape(1) != 3) {
    throw py::value_error("positions must have shape (N, 3)");
  }
  if (pos.shape(0) < 3) {
    throw py::value_error("a DER rod requires at least three vertices");
  }

  if (theta.has_value()) {
    const auto theta_values = theta->unchecked<1>();
    if (theta_values.shape(0) != pos.shape(0)) {
      throw py::value_error("theta must have shape (N,)");
    }
    std::vector<der::RodDof> blocks(static_cast<size_t>(pos.shape(0)));
    for (py::ssize_t i = 0; i < pos.shape(0); ++i) {
      blocks[static_cast<size_t>(i)] =
          der::RodDof(pos(i, 0), pos(i, 1), pos(i, 2), theta_values(i));
    }
    return blocks;
  }

  std::vector<der::RodDof> blocks(static_cast<size_t>(pos.shape(0)));
  for (py::ssize_t i = 0; i < pos.shape(0); ++i) {
    blocks[static_cast<size_t>(i)] =
        der::RodDof(pos(i, 0), pos(i, 1), pos(i, 2), 0.0);
  }
  return blocks;
}

py::array_t<double> geometryPositions(const runtime::RuntimeSimulation& sim)
{
  const auto& geometry = sim.scene().geometry;
  auto result = py::array_t<double>(
      {static_cast<py::ssize_t>(geometry.pointCount()), py::ssize_t{3}});
  auto out = result.mutable_unchecked<2>();
  for (py::ssize_t i = 0;
       i < static_cast<py::ssize_t>(geometry.pointCount());
       ++i) {
    const glm::dvec3 point =
        geometry.worldPosition(runtime::GeometryPointId{
            static_cast<int>(i),
        });
    out(i, 0) = point.x;
    out(i, 1) = point.y;
    out(i, 2) = point.z;
  }
  return result;
}

py::array_t<double> rodPositions(const PyRuntimeSimulation& sim, int rodIndex)
{
  if (rodIndex < 0 ||
      rodIndex >= static_cast<int>(sim.rod_vertex_counts.size())) {
    throw py::index_error("rod index out of range");
  }

  int offset = 0;
  for (int i = 0; i < rodIndex; ++i) {
    offset += sim.rod_vertex_counts[static_cast<size_t>(i)];
  }

  const int count = sim.rod_vertex_counts[static_cast<size_t>(rodIndex)];
  const auto& geometry = sim.simulation.simulation().scene().geometry;
  auto result =
      py::array_t<double>({static_cast<py::ssize_t>(count), py::ssize_t{3}});
  auto out = result.mutable_unchecked<2>();
  for (py::ssize_t i = 0; i < count; ++i) {
    const glm::dvec3 point =
        geometry.worldPosition(runtime::GeometryPointId{
            static_cast<int>(offset + i),
        });
    out(i, 0) = point.x;
    out(i, 1) = point.y;
    out(i, 2) = point.z;
  }
  return result;
}

}  // namespace

void bind_der(py::module_& m)
{
  py::class_<runtime::GlobalSolverConfig>(m, "GlobalSolverConfig")
      .def(py::init<>())
      .def_readwrite("max_newton_iterations",
                     &runtime::GlobalSolverConfig::maxNewtonIterations)
      .def_readwrite("max_pcg_iterations",
                     &runtime::GlobalSolverConfig::maxPcgIterations)
      .def_readwrite("max_line_search_iterations",
                     &runtime::GlobalSolverConfig::maxLineSearchIterations)
      .def_readwrite("newton_gradient_tolerance",
                     &runtime::GlobalSolverConfig::newtonGradientTolerance)
      .def_readwrite("newton_step_tolerance",
                     &runtime::GlobalSolverConfig::newtonStepTolerance)
      .def_readwrite("line_search_armijo",
                     &runtime::GlobalSolverConfig::lineSearchArmijo)
      .def_readwrite("line_search_shrink",
                     &runtime::GlobalSolverConfig::lineSearchShrink)
      .def_readwrite("pcg_tolerance",
                     &runtime::GlobalSolverConfig::pcgTolerance);

  py::class_<runtime::RuntimeStepResult>(m, "RuntimeStepResult")
      .def_readonly("iterations", &runtime::RuntimeStepResult::iterations)
      .def_readonly("final_gradient_norm",
                    &runtime::RuntimeStepResult::finalGradientNorm)
      .def_readonly("final_step_norm",
                    &runtime::RuntimeStepResult::finalStepNorm)
      .def_readonly("converged", &runtime::RuntimeStepResult::converged);

  py::class_<der::RodMaterial>(m, "DERMaterial")
      .def(py::init([](double density,
                       double radius,
                       double youngsModulus,
                       double shearModulus) {
             der::RodMaterial material;
             material.density = density;
             material.radius = radius;
             material.youngsModulus = youngsModulus;
             material.shearModulus = shearModulus;
             return material;
           }),
           py::arg("density") = 1300.0,
           py::arg("radius") = 4e-5,
           py::arg("youngs_modulus") = 4e9,
           py::arg("shear_modulus") = 1.5e9,
           py::arg("root_stiffness") = 2e6,
           py::arg("pin_root_twist") = false)
      .def_readwrite("density", &der::RodMaterial::density)
      .def_readwrite("radius", &der::RodMaterial::radius)
      .def_readwrite("youngs_modulus", &der::RodMaterial::youngsModulus)
      .def_readwrite("shear_modulus", &der::RodMaterial::shearModulus)
      .def_property_readonly("area", &der::RodMaterial::area)
      .def_property_readonly("area_moment", &der::RodMaterial::areaMoment)
      .def_property_readonly("polar_moment", &der::RodMaterial::polarMoment)
      .def_property_readonly("axial_stiffness",
                             &der::RodMaterial::axialStiffness)
      .def_property_readonly("bending_stiffness",
                             &der::RodMaterial::bendingStiffness)
      .def_property_readonly("twist_stiffness",
                             &der::RodMaterial::twistStiffness);

  py::class_<PyDERRod, std::shared_ptr<PyDERRod>>(m, "DERRod")
      .def(py::init([](py::array_t<double> positions,
                       der::RodMaterial material,
                       std::optional<py::array_t<double>> theta) {
        auto rod = std::make_shared<PyDERRod>();
        rod->desc.restBlocks = parseRodBlocks(positions, theta);
        rod->desc.material = material;
        return rod;
      }),
           py::arg("positions"),
           py::arg("material") = der::RodMaterial{},
           py::arg("theta") = py::none())
      .def_property_readonly("num_vertices", [](const PyDERRod& self) {
        return self.desc.restBlocks.size();
      })
      .def_property_readonly("positions", [](const PyDERRod& self) {
        auto result = py::array_t<double>(
            {static_cast<py::ssize_t>(self.desc.restBlocks.size()),
             py::ssize_t{3}});
        auto out = result.mutable_unchecked<2>();
        for (py::ssize_t i = 0;
             i < static_cast<py::ssize_t>(self.desc.restBlocks.size());
             ++i) {
          const der::RodDof& block = self.desc.restBlocks[i];
          out(i, 0) = block.x;
          out(i, 1) = block.y;
          out(i, 2) = block.z;
        }
        return result;
      })
      .def_property(
          "material",
          [](const PyDERRod& self) {
            return self.desc.material;
          },
          [](PyDERRod& self, der::RodMaterial material) {
            self.desc.material = material;
          });

  py::class_<PyDERScene, std::shared_ptr<PyDERScene>>(m, "DERScene")
      .def(py::init([] {
        return std::make_shared<PyDERScene>();
      }))
      .def("add_rod",
           [](PyDERScene& self, const std::shared_ptr<PyDERRod>& rod) {
             self.checkUnlocked();
             if (!rod) {
               throw py::type_error("rod must be a DERRod");
             }
             self.rod_vertex_counts.push_back(
                 static_cast<int>(rod->desc.restBlocks.size()));
             der::addRod(self.scene, rod->desc);
             ++self.rod_count;
           },
           py::arg("rod"))
      .def_property(
          "gravity",
          [](const PyDERScene& self) {
            auto result = py::array_t<double>(3);
            auto out = result.mutable_unchecked<1>();
            out(0) = self.scene.gravity.x;
            out(1) = self.scene.gravity.y;
            out(2) = self.scene.gravity.z;
            return result;
          },
          [](PyDERScene& self, py::array_t<double> gravity) {
            self.checkUnlocked();
            self.scene.gravity = parseVec3(gravity, "gravity");
          })
      .def_property(
          "time_step",
          [](const PyDERScene& self) {
            return self.scene.timeStep;
          },
          [](PyDERScene& self, double timeStep) {
            self.checkUnlocked();
            if (timeStep <= 0.0) {
              throw py::value_error("time_step must be positive");
            }
            self.scene.timeStep = timeStep;
          })
      .def_property(
          "solver",
          [](PyDERScene& self) -> runtime::GlobalSolverConfig& {
            return self.scene.solverConfig;
          },
          [](PyDERScene& self, runtime::GlobalSolverConfig solver) {
            self.checkUnlocked();
            self.scene.solverConfig = solver;
          },
          py::return_value_policy::reference_internal)
      .def("build",
           [](std::shared_ptr<PyDERScene> self) {
             if (!self) {
               throw py::type_error("scene must be a DERScene");
             }
             if (self->rod_count == 0) {
               throw py::value_error("DERScene requires at least one rod");
             }
             self->locked = true;
             return std::make_shared<PyRuntimeSimulation>(
                 runtime::buildSimulationRunner(self->scene),
                 self->rod_vertex_counts);
           });

  py::class_<PyRuntimeSimulation, std::shared_ptr<PyRuntimeSimulation>>(
      m, "RuntimeSimulation")
      .def("step", &PyRuntimeSimulation::step)
      .def("run", &PyRuntimeSimulation::run, py::arg("steps") = 100)
      .def_property_readonly("positions", [](const PyRuntimeSimulation& self) {
        return geometryPositions(self.simulation.simulation());
      })
      .def("rod_positions", &rodPositions, py::arg("rod_index"))
      .def_property_readonly("rod_vertex_counts",
                             [](const PyRuntimeSimulation& self) {
                               return self.rod_vertex_counts;
                             })
      .def_property_readonly("steps_completed",
                             [](const PyRuntimeSimulation& self) {
                               return self.simulation.stepsCompleted();
                             })
      .def_property_readonly("time", [](const PyRuntimeSimulation& self) {
        return self.simulation.time();
      })
      .def_property_readonly("time_step", [](const PyRuntimeSimulation& self) {
        return self.simulation.timeStep();
      })
      .def_property_readonly("last_step", [](const PyRuntimeSimulation& self) {
        return self.simulation.lastStep();
      });
}

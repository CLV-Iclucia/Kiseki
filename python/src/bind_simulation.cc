// Phase 8 & 9: Simulation Orchestration + Lock-After-Run Mechanism
// Phase 11: display() method — Pythonic shorthand for run_and_display
#include "bindings.h"
#include <Renderer/renderer.h>
#include <Renderer/scene-proxy.h>
#include <thread>

using namespace ksk::renderer;
using ksk::core::Vec3f;
using ksk::core::Vec3u;

// Forward-declare buildSceneProxy (defined in bind_renderer.cc as static)
// We re-use a local copy here since it's small and avoids cross-TU dependency.
namespace {

void computeSmoothNormals_sim(MeshProxy& mesh) {
  const auto& pos = mesh.positions;
  const auto& tris = mesh.triangles;
  mesh.normals.assign(pos.size(), Vec3f(0.0f));
  for (const auto& tri : tris) {
    Vec3f e1 = pos[tri.y] - pos[tri.x];
    Vec3f e2 = pos[tri.z] - pos[tri.x];
    Vec3f fn = glm::cross(e1, e2);
    mesh.normals[tri.x] += fn;
    mesh.normals[tri.y] += fn;
    mesh.normals[tri.z] += fn;
  }
  for (auto& n : mesh.normals) {
    float len = glm::length(n);
    n = (len > 1e-8f) ? n / len : Vec3f(0.0f, 1.0f, 0.0f);
  }
}

std::unique_ptr<SceneProxy> buildSceneProxy_sim(System& system,
                                                 const std::vector<glm::vec3>& colors,
                                                 int frameIdx, float simTime)
{
  auto scene = std::make_unique<SceneProxy>();
  scene->frameIndex = frameIdx;
  scene->simulationTime = simTime;

  for (int i = 0; i < static_cast<int>(system.primitives().size()); i++) {
    const auto& prim = system.primitive(i);
    auto surfaceView = prim.getSurfaceView();
    size_t vertCount = prim.getVertexCount();
    int primDofStart = prim.getDofStart();

    MeshProxy mesh;
    mesh.name = "body_" + std::to_string(i);
    if (i < static_cast<int>(colors.size()))
      mesh.objectColor = colors[i];

    mesh.positions.resize(vertCount);
    for (size_t v = 0; v < vertCount; v++) {
      int blockIdx = primDofStart / 3 + static_cast<int>(v);
      auto pos = system.x[blockIdx];
      mesh.positions[v] = Vec3f(float(pos.x), float(pos.y), float(pos.z));
    }

    mesh.triangles.resize(surfaceView.size());
    for (size_t t = 0; t < surfaceView.size(); t++) {
      auto tri = surfaceView[t];
      mesh.triangles[t] = Vec3u(unsigned(tri[0]), unsigned(tri[1]), unsigned(tri[2]));
    }

    computeSmoothNormals_sim(mesh);
    scene->meshes.push_back(std::move(mesh));
  }
  return scene;
}

} // anonymous namespace

void bind_simulation(py::module_& m)
{
  py::class_<PySimulation, std::shared_ptr<PySimulation>>(m, "Simulation")
    .def(py::init([](std::shared_ptr<PySystem> system,
                     std::shared_ptr<PyIntegratorConfig> integrator) {
      if (!system)
        throw py::type_error("system must be a valid System object");
      if (!integrator)
        throw py::type_error("integrator must be a valid IpcIntegrator object");

      auto sim = std::make_shared<PySimulation>();
      sim->py_system = system;
      sim->py_integrator = integrator;
      return sim;
    }), py::arg("system"), py::arg("integrator"),
       "Create a simulation from a system and integrator")

    .def("step", [](PySimulation& self, Real dt) {
      if (dt <= 0.0)
        throw py::value_error("dt must be positive");
      self.do_step(dt);
    }, py::arg("dt") = 0.01,
       "Advance simulation by one timestep. Locks system after first call.")

    .def("run", [](PySimulation& self, Real dt, int steps) {
      if (dt <= 0.0)
        throw py::value_error("dt must be positive");
      if (steps <= 0)
        throw py::value_error("steps must be positive");
      self.run(dt, steps);
    }, py::arg("dt") = 0.01, py::arg("steps") = 100,
       "Run simulation for multiple timesteps. Locks system after call.")

    // ─── display(): the Pythonic way to run with visualization ──────────────
    // Equivalent to kiseki.run_and_display(sim, renderer, dt, steps, ...)
    // but as a method call: sim.display(dt, steps)
    //
    // Key guarantee: frames are pushed automatically. User CANNOT forget.
    .def("display", [](std::shared_ptr<PySimulation> self,
                       Real dt, int steps,
                       int width, int height,
                       const std::string& title,
                       py::object on_step,
                       int log_interval) {
      if (dt <= 0.0)
        throw py::value_error("dt must be positive");
      if (steps <= 0)
        throw py::value_error("steps must be positive");

      bool has_on_step = !on_step.is_none();

      self->ensure_initialized();
      self->lock_all();

      // Create renderer
      RendererConfig cfg;
      cfg.windowWidth = width;
      cfg.windowHeight = height;
      cfg.windowTitle = title;
      cfg.vsync = true;
      auto renderer = createRenderer(cfg);

      // Push initial frame
      const auto& colors = self->py_system->primitive_colors;
      auto initScene = buildSceneProxy_sim(self->py_system->system, colors, 0,
                                           float(self->py_system->system.currentTime()));
      renderer->queue().push(std::move(initScene));

      // Simulation in background thread, auto-pushes frames
      std::exception_ptr sim_exception;
      std::thread simThread([&]() {
        try {
          for (int i = 0; i < steps; i++) {
            if (!renderer->isRunning()) break;

            self->integrator->step(dt);
            self->steps_completed++;

            // ─── Automatic frame push ─────────────────────────────────
            auto scene = buildSceneProxy_sim(self->py_system->system, colors, i + 1,
                                             float(self->py_system->system.currentTime()));
            renderer->queue().push(std::move(scene));

            // ─── Optional on_step callback ────────────────────────────
            if (has_on_step) {
              py::gil_scoped_acquire acquire;
              on_step(i + 1, self->py_system->system.currentTime());
            }

            // ─── Logging ──────────────────────────────────────────────
            if (log_interval > 0 && (i + 1) % log_interval == 0) {
              spdlog::info("[display] step {}/{}, t = {:.4f}",
                           i + 1, steps, self->py_system->system.currentTime());
            }
          }
        } catch (...) {
          sim_exception = std::current_exception();
        }
        renderer->shutdown();
      });

      {
        py::gil_scoped_release release;
        renderer->runOnCurrentThread();
      }
      simThread.join();

      if (sim_exception) std::rethrow_exception(sim_exception);
    },
    py::arg("dt") = 0.01, py::arg("steps") = 1000,
    py::arg("width") = 1280, py::arg("height") = 720,
    py::arg("title") = "Kiseki",
    py::arg("on_step") = py::none(), py::arg("log_interval") = 0,
    "Run simulation with real-time display (method style).\n\n"
    "Frames are pushed automatically after every timestep. You only need\n"
    "to set up the physics — visualization is handled for you.\n\n"
    "Args:\n"
    "    dt:           Timestep size (seconds). Default: 0.01\n"
    "    steps:        Number of timesteps. Default: 1000\n"
    "    width:        Window width. Default: 1280\n"
    "    height:       Window height. Default: 720\n"
    "    title:        Window title. Default: 'Kiseki'\n"
    "    on_step:      Optional callback(step: int, time: float) per step.\n"
    "    log_interval: Print progress every N steps (0 = silent).\n\n"
    "Example::\n\n"
    "    sim = kiseki.Simulation(system, integrator)\n"
    "    sim.display(dt=0.01, steps=500, title='Cube Drop')\n")

    .def_property_readonly("steps_completed", [](const PySimulation& self) {
      return self.steps_completed;
    }, "Number of timesteps completed")

    .def_property_readonly("locked", [](const PySimulation& self) {
      return self.has_run;
    }, "Whether simulation has been run (system locked)")

    .def_property_readonly("kinetic_energy", [](const PySimulation& self) -> Real {
      return self.py_system->system.kineticEnergy();
    }, "Current kinetic energy of the system (J)")

    .def_property_readonly("potential_energy", [](const PySimulation& self) -> Real {
      return self.py_system->system.potentialEnergy();
    }, "Current elastic potential energy of the system (J)")

    .def_property_readonly("total_energy", [](const PySimulation& self) -> Real {
      return self.py_system->system.totalEnergy();
    }, "Current total mechanical energy KE + PE (J)")

    .def_property_readonly("time", [](const PySimulation& self) -> Real {
      return self.py_system->system.currentTime();
    }, "Current simulation time (s)");
}

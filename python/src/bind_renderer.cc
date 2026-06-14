// Phase 10: Renderer binding + run_and_display orchestration
// Design: Renderer is fully decoupled from Simulation.
// run_and_display() is a free function that composes them via SceneProxy.
// The API automatically pushes frames after every step — users cannot forget.
#include "bindings.h"
#include <Renderer/renderer.h>
#include <Renderer/scene-proxy.h>
#include <thread>
#include <atomic>

using namespace sim::renderer;
using sim::core::Vec3f;
using sim::core::Vec3u;

/// Compute area-weighted smooth vertex normals
static void computeSmoothNormals(MeshProxy& mesh) {
  const auto& pos = mesh.positions;
  const auto& tris = mesh.triangles;

  mesh.normals.assign(pos.size(), Vec3f(0.0f));

  for (const auto& tri : tris) {
    const auto& p0 = pos[tri.x];
    const auto& p1 = pos[tri.y];
    const auto& p2 = pos[tri.z];

    Vec3f e1 = p1 - p0;
    Vec3f e2 = p2 - p0;
    Vec3f faceNormal = glm::cross(e1, e2); // unnormalized = area-weighted

    mesh.normals[tri.x] += faceNormal;
    mesh.normals[tri.y] += faceNormal;
    mesh.normals[tri.z] += faceNormal;
  }

  for (auto& n : mesh.normals) {
    float len = glm::length(n);
    if (len > 1e-8f)
      n /= len;
    else
      n = Vec3f(0.0f, 1.0f, 0.0f);
  }
}

/// Build a SceneProxy from the current System state
static std::unique_ptr<SceneProxy> buildSceneProxy(System& system,
                                                    const std::vector<glm::vec3>& colors,
                                                    int frameIdx, float simTime)
{
  auto scene = std::make_unique<SceneProxy>();
  scene->frameIndex = frameIdx;
  scene->simulationTime = simTime;

  // Diagnostic: log first vertex position periodically
  if (frameIdx % 10 == 0 && system.x.numBlocks() > 0) {
    auto v0 = system.x[0];
    spdlog::debug("[SceneProxy] frame={} body0_vert0=({:.6f},{:.6f},{:.6f})",
                  frameIdx, v0.x, v0.y, v0.z);
  }

  // Extract each primitive's surface mesh
  int dofStart = 0;
  for (int i = 0; i < static_cast<int>(system.primitives().size()); i++) {
    const auto& prim = system.primitive(i);
    auto surfaceView = prim.getSurfaceView();
    size_t vertCount = prim.getVertexCount();
    int primDofStart = prim.getDofStart();

    MeshProxy mesh;
    mesh.name = "body_" + std::to_string(i);

    // Per-object color
    if (i < static_cast<int>(colors.size()))
      mesh.objectColor = colors[i];

    // Extract vertex positions from system.x (BlockVector<3>)
    mesh.positions.resize(vertCount);
    for (size_t v = 0; v < vertCount; v++) {
      int blockIdx = primDofStart / 3 + static_cast<int>(v);
      auto pos = system.x[blockIdx]; // glm::dvec3
      mesh.positions[v] = Vec3f(
          static_cast<float>(pos.x),
          static_cast<float>(pos.y),
          static_cast<float>(pos.z));
    }

    // Extract surface triangles
    mesh.triangles.resize(surfaceView.size());
    for (size_t t = 0; t < surfaceView.size(); t++) {
      auto tri = surfaceView[t];
      mesh.triangles[t] = Vec3u(
          static_cast<unsigned>(tri[0]),
          static_cast<unsigned>(tri[1]),
          static_cast<unsigned>(tri[2]));
    }

    // Compute smooth vertex normals
    computeSmoothNormals(mesh);

    scene->meshes.push_back(std::move(mesh));
  }

  return scene;
}

/// Python-facing Renderer wrapper
struct PyRenderer {
  RendererConfig config;
  std::unique_ptr<Renderer> impl;

  void create() {
    if (!impl)
      impl = createRenderer(config);
  }
};

void bind_renderer(py::module_& m)
{
  // RendererConfig as Renderer constructor args
  py::class_<PyRenderer, std::shared_ptr<PyRenderer>>(m, "Renderer")
    .def(py::init([](int width, int height, const std::string& title, bool vsync) {
      auto r = std::make_shared<PyRenderer>();
      r->config.windowWidth = width;
      r->config.windowHeight = height;
      r->config.windowTitle = title;
      r->config.vsync = vsync;
      return r;
    }),
    py::arg("width") = 1280,
    py::arg("height") = 720,
    py::arg("title") = "SimCraft",
    py::arg("vsync") = true,
    "Create a renderer with window configuration")

    .def_property_readonly("width", [](const PyRenderer& self) {
      return self.config.windowWidth;
    })
    .def_property_readonly("height", [](const PyRenderer& self) {
      return self.config.windowHeight;
    });

  // run_and_display: top-level composition function
  // Automatically pushes frames after every step — user cannot forget.
  m.def("run_and_display", [](std::shared_ptr<PySimulation> sim,
                               std::shared_ptr<PyRenderer> renderer,
                               Real dt, int steps,
                               py::object on_step,
                               int log_interval) {
    if (!sim)
      throw py::type_error("simulation must be a valid Simulation object");
    if (!renderer)
      throw py::type_error("renderer must be a valid Renderer object");
    if (dt <= 0.0)
      throw py::value_error("dt must be positive");
    if (steps <= 0)
      throw py::value_error("steps must be positive");

    // Determine if we have a Python per-step callback
    bool has_on_step = !on_step.is_none();

    // Ensure simulation is initialized (system.init + integrator created)
    sim->ensure_initialized();
    sim->lock_all();

    // Create the renderer (OpenGL context)
    renderer->create();
    auto& rend = *renderer->impl;

    // Push initial frame (step 0 = initial state before simulation starts)
    const auto& colors = sim->py_system->primitive_colors;
    auto initScene = buildSceneProxy(sim->py_system->system, colors, 0,
                                     static_cast<float>(sim->py_system->system.currentTime()));
    rend.queue().push(std::move(initScene));

    // Simulation runs in background thread, pushes frames automatically
    std::exception_ptr sim_exception;
    std::thread simThread([&]() {
      try {
        for (int i = 0; i < steps; i++) {
          if (!rend.isRunning()) break;

          sim->integrator->step(dt);
          sim->steps_completed++;

          // ─── Automatic frame push (the key design guarantee) ─────────
          auto scene = buildSceneProxy(sim->py_system->system, colors, i + 1,
                                       static_cast<float>(sim->py_system->system.currentTime()));
          rend.queue().push(std::move(scene));

          // ─── Optional per-step callback (with GIL for Python safety) ──
          if (has_on_step) {
            py::gil_scoped_acquire acquire;
            on_step(i + 1, sim->py_system->system.currentTime());
          }

          // ─── Default logging ──────────────────────────────────────────
          if (log_interval > 0 && (i + 1) % log_interval == 0) {
            spdlog::info("[run_and_display] step {}/{}, t = {:.4f}",
                         i + 1, steps, sim->py_system->system.currentTime());
          }
        }
      } catch (...) {
        sim_exception = std::current_exception();
      }
      rend.shutdown();
    });

    // Render loop runs on main thread (GLFW requirement)
    {
      py::gil_scoped_release release;
      rend.runOnCurrentThread();
    }

    simThread.join();

    // Re-throw simulation exceptions on main thread (with GIL held)
    if (sim_exception) std::rethrow_exception(sim_exception);

  }, py::arg("simulation"), py::arg("renderer"),
     py::arg("dt") = 0.01, py::arg("steps") = 1000,
     py::arg("on_step") = py::none(), py::arg("log_interval") = 0,
     "Run simulation with real-time display.\n\n"
     "Rendering frames are pushed automatically after every step — you cannot\n"
     "forget to update the display. The user only provides physics setup.\n\n"
     "Args:\n"
     "    simulation: A Simulation object (system + integrator).\n"
     "    renderer:   A Renderer object (window config).\n"
     "    dt:         Timestep size (seconds).\n"
     "    steps:      Number of timesteps to run.\n"
     "    on_step:    Optional callback(step: int, time: float) called after each step.\n"
     "               Use for logging, energy tracking, etc.\n"
     "    log_interval: Print progress every N steps (0 = silent). Default: 0.\n\n"
     "Main thread renders (GLFW), background thread simulates.\n"
     "Window closes when simulation completes or user closes window.");

}

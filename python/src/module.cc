#include <pybind11/pybind11.h>
#include "bindings.h"

namespace py = pybind11;

PYBIND11_MODULE(kiseki, m)
{
  m.doc() = "Kiseki physics simulation framework";
  m.attr("__version__") = KISEKI_VERSION;
  m.def("hello", []() { return "Kiseki is alive!"; },
        "Placeholder function to verify the module loads correctly");

  // Register all bindings
  bind_mesh(m);
  bind_material(m);
  bind_system(m);
  bind_constraints(m);
  bind_kinematic(m);
  bind_integrator(m);
  bind_simulation(m);
  bind_renderer(m);
  bind_der(m);
}

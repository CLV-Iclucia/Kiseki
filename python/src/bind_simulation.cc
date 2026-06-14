// Phase 8 & 9: Simulation Orchestration + Lock-After-Run Mechanism
#include "bindings.h"

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

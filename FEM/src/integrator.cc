//
// Created by CreeperIclucia-Vader on 25-5-26.
//
#include <fem/integrator.h>
#include <fem/integrator-factory.h>
#include <fem/ipc/integrator.h>

namespace ksk::fem {

std::unique_ptr<Integrator>
createIntegrator(System& system, const core::JsonNode &node) {
  if (!node.is<core::JsonDict>())
    throw std::runtime_error("Expected a dictionary for Integrator");

  const auto &dict = node.as<core::JsonDict>();
  const auto &type = dict.at("type").as<std::string>();
  const auto &config = dict.at("config");
  return IntegratorFactory::instance().create(type, system, config);
}

std::unique_ptr<Integrator> Integrator::create(System &system,
                                               const core::JsonNode &json) {
  return createIntegrator(system, json);
}

} // namespace fem
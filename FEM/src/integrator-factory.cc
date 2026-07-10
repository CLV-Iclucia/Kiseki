//
// Created by CreeperIclucia-Vader on 25-5-29.
//
#include <fem/integrator-factory.h>
#include <spdlog/spdlog.h>

namespace ksk::fem {

IntegratorFactory& IntegratorFactory::instance() {
  static IntegratorFactory instance;
  return instance;
}

void IntegratorFactory::registerCreator(const std::string &name,
                                        Creator creator) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (creators_.find(name) != creators_.end())
    throw std::runtime_error("Duplicate registration: " + name);
  creators_[name] = std::move(creator);
  spdlog::info("Registered Integrator: {}", name);
}

std::unique_ptr<Integrator>
IntegratorFactory::create(const std::string &name, System& system,
                          const core::JsonNode &json) const {
  std::lock_guard<std::mutex> lock(mutex_);
  if (auto it = creators_.find(name); it != creators_.end()) {
    try {
      return it->second(system, json);
    } catch (const std::exception &e) {
      throw std::runtime_error("Creation failed for '" + name +
                               "': " + e.what());
    }
  }
  throw std::runtime_error("Unknown type: " + name);
}

} 
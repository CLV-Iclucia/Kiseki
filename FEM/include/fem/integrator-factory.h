//
// Created by CreeperIclucia-Vader on 25-5-29.
//

#pragma once

#include <fem/integrator.h>
#include <mutex>

namespace ksk::fem {

struct IntegratorFactory {
  using Creator = std::function<std::unique_ptr<Integrator>(System&, const core::JsonNode&)>;

  IntegratorFactory(const IntegratorFactory&) = delete;
  void operator=(const IntegratorFactory&) = delete;
  static IntegratorFactory& instance();
  void registerCreator(const std::string& name, Creator creator);
  std::unique_ptr<Integrator>
  create(const std::string &name, System& system, const core::JsonNode &json) const;

private:
  IntegratorFactory() = default;
  std::unordered_map<std::string, Creator> creators_;
  mutable std::mutex mutex_;
};

} 
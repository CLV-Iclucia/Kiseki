//
// Created by CreeperIclucia-Vader on 25-5-26.
//

#pragma once
#include <Core/animation.h>
#include <fem/integrator.h>
#include <fem/system.h>

namespace ksk::fem {
struct FEMSimulation final : public core::Animation {
  FEMSimulation() = default;
  void step(core::Frame &frame) override {
    if (integrator)
      integrator->step(frame.dt);
    frame.onAdvance();
  }
  [[nodiscard]] bool canSimulate() const { return integrator != nullptr; }

  /// 访问内部 System（只读 / 读写）
  [[nodiscard]] System& getSystem() { return system; }
  [[nodiscard]] const System& getSystem() const { return system; }

private:
  friend struct FEMSimulationBuilder;
  System system;
  std::unique_ptr<Integrator> integrator;
};

struct FEMSimulationBuilder {
private:
  SystemBuilder system{};

public:
  FEMSimulation build(const core::JsonNode &json) {
    if (!json.is<core::JsonDict>())
      throw std::runtime_error("FEMSimulationBuilder requires a JSON object");
    const auto &dict = json.as<core::JsonDict>();
    if (!dict.contains("system") || !dict.contains("integrator"))
      throw std::runtime_error(
          "FEMSimulationBuilder requires 'system' and 'integrator' keys");
    FEMSimulation sim;
    sim.system = std::move(system.build(dict.at("system")));
    sim.integrator = Integrator::create(sim.system, dict.at("integrator"));
    return sim;
  }
};

} // namespace fem
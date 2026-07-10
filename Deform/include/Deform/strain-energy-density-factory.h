//
// Created by CreeperIclucia-Vader on 25-5-29.
//

#pragma once
#include <Deform/strain-energy-density.h>
#include <mutex>
namespace ksk::deform {

template<typename T>
struct StrainEnergyDensityFactory {
public:
  using Creator = std::function<std::unique_ptr<StrainEnergyDensity<T>>(const core::JsonNode&)>;

  StrainEnergyDensityFactory(const StrainEnergyDensityFactory&) = delete;
  void operator=(const StrainEnergyDensityFactory&) = delete;
  static StrainEnergyDensityFactory& instance();
  void registerCreator(const std::string& name, Creator creator);
  std::unique_ptr<StrainEnergyDensity<T>>
  create(const std::string &name, const core::JsonNode &json) const;

private:
  StrainEnergyDensityFactory() = default;
  std::unordered_map<std::string, Creator> creators_;
  mutable std::mutex mutex_;
};

template struct StrainEnergyDensityFactory<float>;
template struct StrainEnergyDensityFactory<double>;

}
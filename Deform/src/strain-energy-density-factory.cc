//
// Created by CreeperIclucia-Vader on 25-5-29.
//
#include <Deform/strain-energy-density-factory.h>
#include <spdlog/spdlog.h>
namespace ksk::deform {
template <typename T>
StrainEnergyDensityFactory<T>& StrainEnergyDensityFactory<T>::instance() {
  static StrainEnergyDensityFactory<T> instance;
  return instance;
}

template <typename T>
void StrainEnergyDensityFactory<T>::registerCreator(const std::string &name,
                                                    Creator creator) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (creators_.find(name) != creators_.end())
    throw std::runtime_error("Duplicate registration: " + name);
  creators_[name] = std::move(creator);
  spdlog::info("Registered StrainEnergyDensity<{}>: {}", typeid(T).name(),
               name);
}

template <typename T>
std::unique_ptr<StrainEnergyDensity<T>>
StrainEnergyDensityFactory<T>::create(const std::string &name,
                                      const core::JsonNode &json) const {
  std::lock_guard<std::mutex> lock(mutex_);
  if (auto it = creators_.find(name); it != creators_.end()) {
    try {
      return it->second(json);
    } catch (const std::exception &e) {
      throw std::runtime_error("Creation failed for '" + name +
                               "': " + e.what());
    }
  }
  throw std::runtime_error("Unknown type: " + name);
}

template struct StrainEnergyDensityFactory<float>;
template struct StrainEnergyDensityFactory<double>;
}
//
// Created by CreeperIclucia-Vader on 25-5-28.
//
#include <Deform/strain-energy-density-factory.h>
namespace ksk::deform {

template <typename T>
  static int snh_auto_reg = ([]() {
    StrainEnergyDensityFactory<T>::instance().registerCreator(
        "StableNeoHookean",
        [](const core::JsonNode &json) {
          return StableNeoHookean<T>::createFromJson(json);
        });
  }(), 0);

template int snh_auto_reg<double>;

template <typename T>
static int arap_auto_reg = ([]() {
    StrainEnergyDensityFactory<T>::instance().registerCreator(
        "ARAP",
        [](const core::JsonNode &json) {
          return ARAP<T>::createFromJson(json);
        });
  }(), 0);

template int arap_auto_reg<double>;

template <typename T>
static int le_auto_reg = ([]() {
    StrainEnergyDensityFactory<T>::instance().registerCreator(
        "LinearElastic",
        [](const core::JsonNode &json) {
          return LinearElastic<T>::createFromJson(json);
        });
  }(), 0);

template int le_auto_reg<double>;

template <typename T>
std::unique_ptr<StrainEnergyDensity<T>>
createStrainEnergyDensity(const core::JsonNode &node) {
  if (!node.is<core::JsonDict>())
    throw std::runtime_error("Expected a dictionary for StrainEnergyDensity");

  const auto &dict = node.as<core::JsonDict>();
  const auto &type = dict.at("type").as<std::string>();
  return StrainEnergyDensityFactory<T>::instance().create(type, node);
}

template std::unique_ptr<StrainEnergyDensity<double>>
createStrainEnergyDensity<double>(const core::JsonNode &node);
} // namespace ksk::deform
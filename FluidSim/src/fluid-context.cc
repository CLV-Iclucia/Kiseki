// ============================================================================
// src/fluid-context.cc
// ============================================================================

#include <FluidSim/fluid-context.h>

namespace fluid {

bool FluidContext::has(std::string_view name) const {
    std::string key(name);
    return registry_.find(key) != registry_.end();
}

void FluidContext::remove(std::string_view name) {
    std::string key(name);
    registry_.erase(key);
}

std::vector<std::string> FluidContext::keys() const {
    std::vector<std::string> result;
    result.reserve(registry_.size());
    for (const auto& [k, _] : registry_)
        result.push_back(k);
    return result;
}

} // namespace fluid

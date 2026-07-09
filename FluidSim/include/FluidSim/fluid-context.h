// ============================================================================
// include/FluidSim/fluid-context.h
// ============================================================================
#pragma once

#include <FluidSim/fluid-types.h>
#include <Core/core.h>
#include <any>
#include <string>
#include <string_view>
#include <unordered_map>
#include <stdexcept>
#include <vector>

namespace fluid {

class FluidContext : NonCopyable {
public:
    virtual ~FluidContext() = default;

    FluidDomain domain{};
    Real        gridSpacing{};

    template<typename T>
    T& set(std::string_view name, T value);

    template<typename T, typename... Args>
    T& ensure(std::string_view name, Args&&... args);

    template<typename T>
    T& get(std::string_view name);

    template<typename T>
    const T& get(std::string_view name) const;

    template<typename T>
    T* tryGet(std::string_view name);

    template<typename T>
    const T* tryGet(std::string_view name) const;

    bool has(std::string_view name) const;

    void remove(std::string_view name);

    std::vector<std::string> keys() const;

    virtual void beginFrame() {}
    virtual void endFrame() {}

private:
    std::unordered_map<std::string, std::any> registry_;
};


template<typename T>
T& FluidContext::set(std::string_view name, T value) {
    std::string key(name);
    registry_[key] = std::move(value);
    return std::any_cast<T&>(registry_[key]);
}

template<typename T, typename... Args>
T& FluidContext::ensure(std::string_view name, Args&&... args) {
    std::string key(name);
    auto it = registry_.find(key);
    if (it != registry_.end()) {
        if (auto* ptr = std::any_cast<T>(&it->second))
            return *ptr;
        throw std::runtime_error(
            "Registry key '" + key + "' exists but with incompatible type");
    }
    registry_[key] = T(std::forward<Args>(args)...);
    return std::any_cast<T&>(registry_[key]);
}

template<typename T>
T& FluidContext::get(std::string_view name) {
    std::string key(name);
    auto it = registry_.find(key);
    if (it == registry_.end())
        throw std::runtime_error("Registry key '" + key + "' not found");
    auto* ptr = std::any_cast<T>(&it->second);
    if (!ptr)
        throw std::runtime_error("Registry key '" + key + "' type mismatch");
    return *ptr;
}

template<typename T>
const T& FluidContext::get(std::string_view name) const {
    return const_cast<FluidContext*>(this)->get<T>(name);
}

template<typename T>
T* FluidContext::tryGet(std::string_view name) {
    std::string key(name);
    auto it = registry_.find(key);
    if (it == registry_.end()) return nullptr;
    return std::any_cast<T>(&it->second);
}

template<typename T>
const T* FluidContext::tryGet(std::string_view name) const {
    return const_cast<FluidContext*>(this)->tryGet<T>(name);
}

} // namespace fluid

//
// Created by creeper on 23-9-1.
//

#pragma once

#include <any>
#include <string>
#include <string_view>
#include <unordered_map>
#include <optional>

namespace sim::core {

struct NonCopyable {
  NonCopyable() = default;
  NonCopyable(const NonCopyable &) = delete;
  NonCopyable &operator=(const NonCopyable &) = delete;
  NonCopyable(NonCopyable &&) = default;
};

struct Resource {
  Resource(Resource &&) = delete;
};

/// 类型擦除的属性集合，用于 Python/UI 端运行时配置
class Properties {
public:
    Properties() = default;

    template<typename T>
    void set(std::string_view key, T value) {
        data_[std::string(key)] = std::move(value);
    }

    template<typename T>
    std::optional<T> tryGet(std::string_view key) const {
        auto it = data_.find(std::string(key));
        if (it == data_.end()) return std::nullopt;
        auto* ptr = std::any_cast<T>(&it->second);
        if (!ptr) return std::nullopt;
        return *ptr;
    }

    template<typename T>
    T get(std::string_view key, T defaultValue = {}) const {
        auto opt = tryGet<T>(key);
        return opt.value_or(defaultValue);
    }

    bool has(std::string_view key) const {
        return data_.find(std::string(key)) != data_.end();
    }

    void remove(std::string_view key) {
        data_.erase(std::string(key));
    }

private:
    std::unordered_map<std::string, std::any> data_;
};

} // namespace sim::core

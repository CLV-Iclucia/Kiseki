// add protection against multiple includes
#pragma once
#include <Core/core.h>
#include <random>
#include <type_traits>
namespace ksk::core {
float randomFloat();
Real randomReal();
template <typename T>
inline T randomScalar() {
    static_assert(std::is_same_v<T, Real> || std::is_same_v<T, float>);
    if constexpr (std::is_same_v<T, Real>) {
        return randomReal();
    } else if constexpr (std::is_same_v<T, float>) {
        return randomFloat();
    }
}
template <typename T, int Dim>
inline Vector<T, Dim> randomVec() {
    Vector<T, Dim> ret;
    for (int i = 0; i < Dim; i++)
        ret[i] = randomScalar<T>();
    return ret;
}

}
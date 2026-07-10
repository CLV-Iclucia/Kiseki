//
// types.h — RPK shared type definitions
//

#pragma once

#include <cstdint>

namespace ksk::rpk {

// Scalar type of the elements being processed.
enum class ScalarType : uint32_t {
    Float32 = 0,
    Float64 = 1,
    Int32   = 2,
    Uint32  = 3,
};

}  // namespace ksk::rpk

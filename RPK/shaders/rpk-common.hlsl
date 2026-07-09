// ============================================================================
// rpk-common.hlsl — Shared type/op definitions for RPK shaders
// ============================================================================

#ifndef RPK_COMMON_HLSL
#define RPK_COMMON_HLSL

// ---- Numeric constants for preprocessor comparison ----
#define FLOAT32 0
#define FLOAT64 1
#define INT32   2
#define UINT32  3

#define SUM 0
#define MAX 1
#define MIN 2

// ---- Scalar type selection via preprocessor ----
// Define SCALAR_TYPE as one of: FLOAT32, FLOAT64, INT32, UINT32

#if SCALAR_TYPE == FLOAT64
    typedef double scalar_t;
    #define SCALAR_ZERO double(0.0)
    #define SCALAR_MAX  double(1.7976931348623158e+308)
    #define SCALAR_MIN  double(-1.7976931348623158e+308)
#elif SCALAR_TYPE == INT32
    typedef int scalar_t;
    #define SCALAR_ZERO 0
    #define SCALAR_MAX  2147483647
    #define SCALAR_MIN  (-2147483647 - 1)
#elif SCALAR_TYPE == UINT32
    typedef uint scalar_t;
    #define SCALAR_ZERO 0u
    #define SCALAR_MAX  4294967295u
    #define SCALAR_MIN  0u
#else
    // Default: FLOAT32 (SCALAR_TYPE == 0 or undefined)
    typedef float scalar_t;
    #define SCALAR_ZERO 0.0f
    #define SCALAR_MAX  3.402823466e+38f
    #define SCALAR_MIN  (-3.402823466e+38f)
#endif

// ---- Reduce/Scan operator selection ----
// Define REDUCE_OP or SCAN_OP as one of: SUM, MAX, MIN

#if defined(REDUCE_OP)
    #define RPK_OP REDUCE_OP
#elif defined(SCAN_OP)
    #define RPK_OP SCAN_OP
#else
    #define RPK_OP SUM
#endif

#if RPK_OP == MAX
    #define OP_IDENTITY SCALAR_MIN
    scalar_t rpk_op(scalar_t a, scalar_t b) { return max(a, b); }
#elif RPK_OP == MIN
    #define OP_IDENTITY SCALAR_MAX
    scalar_t rpk_op(scalar_t a, scalar_t b) { return min(a, b); }
#else
    // Default: SUM
    #define OP_IDENTITY SCALAR_ZERO
    scalar_t rpk_op(scalar_t a, scalar_t b) { return a + b; }
#endif

#define WORKGROUP_SIZE 256

#endif // RPK_COMMON_HLSL

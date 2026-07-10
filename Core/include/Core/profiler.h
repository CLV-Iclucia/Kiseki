//
// Kiseki Profiler — thin wrapper over Tracy
//
// Usage:
//   SIM_PROFILE_FUNCTION()          — 标记整个函数
//   SIM_PROFILE_SCOPE("name")       — 标记任意代码块
//   SIM_PROFILE_FRAME_MARK()        — 标记帧边界（用于 step 结束处）
//   SIM_PROFILE_VALUE("name", val)  — 追踪标量数值（如能量、迭代次数）
//
// 编译时通过 CMake 选项 KISEKI_PROFILE=ON 启用（自动定义 TRACY_ENABLE）。
// 关闭时所有宏展开为空语句，零开销。
//

#pragma once

#ifdef TRACY_ENABLE

#include <tracy/Tracy.hpp>

#define SIM_PROFILE_FUNCTION()      ZoneScoped
#define SIM_PROFILE_SCOPE(name)     ZoneScopedN(name)
#define SIM_PROFILE_FRAME_MARK()    FrameMark
#define SIM_PROFILE_VALUE(name, val) \
    do { TracyPlot(name, static_cast<double>(val)); } while(0)

// 带颜色标记（方便在时间线中区分不同阶段）
#define SIM_PROFILE_SCOPE_COLOR(name, color) ZoneScopedNC(name, color)

// 常用颜色
namespace ksk::core::profiler_colors {
    inline constexpr unsigned int kGradient   = 0xFF6B6B;  // 红
    inline constexpr unsigned int kHessian    = 0x4ECDC4;  // 青
    inline constexpr unsigned int kSolver     = 0x45B7D1;  // 蓝
    inline constexpr unsigned int kCollision  = 0xFFA07A;  // 橙
    inline constexpr unsigned int kLineSearch = 0x98D8C8;  // 绿
    inline constexpr unsigned int kEnergy     = 0xDDA0DD;  // 紫
}

#else // TRACY_ENABLE not defined

#define SIM_PROFILE_FUNCTION()          (void)0
#define SIM_PROFILE_SCOPE(name)         (void)0
#define SIM_PROFILE_FRAME_MARK()        (void)0
#define SIM_PROFILE_VALUE(name, val)    (void)0
#define SIM_PROFILE_SCOPE_COLOR(name, color) (void)0

namespace ksk::core::profiler_colors {
    inline constexpr unsigned int kGradient   = 0;
    inline constexpr unsigned int kHessian    = 0;
    inline constexpr unsigned int kSolver     = 0;
    inline constexpr unsigned int kCollision  = 0;
    inline constexpr unsigned int kLineSearch = 0;
    inline constexpr unsigned int kEnergy     = 0;
}

#endif // TRACY_ENABLE

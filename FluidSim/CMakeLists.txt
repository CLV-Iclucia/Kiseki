project(FluidSim LANGUAGES CXX)

# ============================================================================
# Source collection
# ============================================================================

# Common: backend-agnostic interface types (fluid-types.h, fluid-backend.h, ...)
file(GLOB common_headers CONFIGURE_DEPENDS include/FluidSim/*.h)
file(GLOB common_srcs    CONFIGURE_DEPENDS src/*.cc src/*.cpp)

# CPU backend sources
file(GLOB_RECURSE cpu_srcs    CONFIGURE_DEPENDS src/cpu/*.cc src/cpu/*.cpp)
file(GLOB_RECURSE cpu_headers CONFIGURE_DEPENDS include/FluidSim/cpu/*.h)

# GPU backend sources (Vulkan/RHI)
file(GLOB_RECURSE gpu_srcs    CONFIGURE_DEPENDS src/gpu/*.cc src/gpu/*.cpp)
file(GLOB_RECURSE gpu_headers CONFIGURE_DEPENDS include/FluidSim/gpu/*.h)

# ============================================================================
# FluidSim — unified library (Modular Solver Pipeline Architecture)
#
# Both CPU and GPU backends are compiled into a single target.
# The caller creates a simulation via:
#   - presets::createFreeSurface(config)  (recommended)
#   - Manual FluidSimulator(ctx, solvers) construction
#
# Legacy code (CPUFluidBackend, GPUFluidBackend) is still compiled for
# backward compatibility during the transition period.
#
# If RHI is not available (SIMCRAFT_BUILD_RHI=OFF) the GPU sources are simply
# omitted via the conditional below — the CPU backend remains fully usable.
# ============================================================================
if(SIMCRAFT_BUILD_RHI)
    add_library(FluidSim STATIC
        ${common_srcs}    ${common_headers}
        ${cpu_srcs}       ${cpu_headers}
        ${gpu_srcs}       ${gpu_headers}
    )

    target_link_libraries(FluidSim PUBLIC Core Spatify TBB::tbb RHI)
else()
    message(STATUS "[FluidSim] RHI not available — GPU backend excluded")

    add_library(FluidSim STATIC
        ${common_srcs}    ${common_headers}
        ${cpu_srcs}       ${cpu_headers}
    )

    target_link_libraries(FluidSim PUBLIC Core Spatify TBB::tbb)
endif()

target_include_directories(FluidSim PUBLIC include/)

target_compile_definitions(FluidSim PRIVATE
    FLUIDSIM_SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/shaders"
    SIMCRAFT_ASSETS_DIR="${ASSETS_PATH}"
)



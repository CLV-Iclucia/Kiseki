cmake_minimum_required(VERSION 3.20)

# Auto-detect vcpkg if VCPKG_ROOT is set and no toolchain specified
if(DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
  set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
      CACHE STRING "Vcpkg toolchain file")
endif()

set(CMAKE_CXX_STANDARD 20)
project(SimCraft)

if (MSVC)
    add_compile_options(/Zc:preprocessor)
endif()

find_package(TBB REQUIRED)
find_package(GTest REQUIRED)
find_package(Eigen3 REQUIRED)

# Tracy profiler — enable with -DSIMCRAFT_PROFILE=ON
option(SIMCRAFT_PROFILE "Enable Tracy profiler instrumentation" ON)
if(SIMCRAFT_PROFILE)
  find_package(Tracy CONFIG REQUIRED)
  add_compile_definitions(TRACY_ENABLE)
  message(STATUS "Tracy profiling ENABLED")
endif()

set(EXTERNAL_PATH ${CMAKE_CURRENT_SOURCE_DIR}/external)
set(ASSETS_PATH ${CMAKE_CURRENT_SOURCE_DIR}/assets)

# External libraries
add_subdirectory(${EXTERNAL_PATH}/glm)
add_subdirectory(${EXTERNAL_PATH}/spdlog)
add_subdirectory(${EXTERNAL_PATH}/cxxopts)

# Internal libraries (ordered by dependency)
add_subdirectory(Core)
add_subdirectory(Spatify)
add_subdirectory(Maths)
add_subdirectory(Deform)
add_subdirectory(FEM)

# RHI (Vulkan) module — gated behind option so users without Vulkan SDK
# can still build the rest of the project.
option(SIMCRAFT_BUILD_RHI "Build RHI module" ON)
if(SIMCRAFT_BUILD_RHI)
  add_subdirectory(RHI)
endif()

# FluidSim depends on RHI (GPU backend), so it comes after RHI
add_subdirectory(FluidSim)

add_subdirectory(Renderer)

# Applications
add_subdirectory(Apps)

# Python bindings
option(SIMCRAFT_BUILD_PYTHON "Build Python bindings" ON)
if(SIMCRAFT_BUILD_PYTHON)
  add_subdirectory(python)
endif()

# add_subdirectory(HairSim)  # WIP: requires Intel MKL

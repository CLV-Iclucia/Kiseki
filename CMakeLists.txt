cmake_minimum_required(VERSION 3.20)

# Auto-detect vcpkg if VCPKG_ROOT is set and no toolchain specified
if(DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
  set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
      CACHE STRING "Vcpkg toolchain file")
endif()

set(CMAKE_CXX_STANDARD 20)
project(Kiseki)

if (MSVC)
    add_compile_options(/Zc:preprocessor)
endif()

find_package(TBB REQUIRED)
find_package(GTest REQUIRED)
find_package(Eigen3 REQUIRED)

# Tracy profiler — enable with -DKISEKI_PROFILE=ON
option(KISEKI_PROFILE "Enable Tracy profiler instrumentation" ON)
if(KISEKI_PROFILE)
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

# Runtime buffers use RHI-owned GPU storage for device-resident data.
option(KISEKI_BUILD_RHI "Build RHI module" ON)
if(KISEKI_BUILD_RHI)
  add_subdirectory(RHI)
  add_subdirectory(RPK)
else()
  message(FATAL_ERROR "Kiseki Runtime buffers currently require RHI.")
endif()

# Kiseki-level runtime layer. Keep it before physics modules so FEM/HairSim
# adapters can depend on Kiseki::Runtime without creating reverse dependencies.
add_subdirectory(Engine)

# FEM has an optional GPU backend (RHI + RPK), so it must come after them.
add_subdirectory(FEM)

# FluidSim depends on RHI (GPU backend), so it comes after RHI
add_subdirectory(FluidSim)

add_subdirectory(Renderer)

add_subdirectory(HairSim)

# Applications
add_subdirectory(Apps)

# Python bindings
option(KISEKI_BUILD_PYTHON "Build Python bindings" ON)
if(KISEKI_BUILD_PYTHON)
  add_subdirectory(python)
endif()

project(FEM LANGUAGES CXX)

file(GLOB_RECURSE cpu_srcs CONFIGURE_DEPENDS src/*.cc src/*.cpp include/*.h)
list(FILTER cpu_srcs EXCLUDE REGEX "/gpu/")
file(GLOB_RECURSE gpu_srcs CONFIGURE_DEPENDS src/gpu/*.cc src/gpu/*.cpp include/fem/gpu/*.h)

if(KISEKI_BUILD_RHI)
    add_library(FEM STATIC ${cpu_srcs} ${gpu_srcs})
    target_link_libraries(FEM PUBLIC Kiseki::Maths Kiseki::Core Kiseki::Spatify Kiseki::Deform Kiseki::RHI Kiseki::RPK)
    target_compile_definitions(FEM PUBLIC FEM_GPU_ENABLED)
    target_compile_definitions(FEM PRIVATE FEM_SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/shaders")
    # Lets GPU modules add FEM/include to the shader compiler's -I so kernels can
    # #include <fem/shared/*.h> (C++/HLSL cross-language headers).
    target_compile_definitions(FEM PRIVATE FEM_INCLUDE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/include")
else()
    add_library(FEM STATIC ${cpu_srcs})
    target_link_libraries(FEM PUBLIC Kiseki::Maths Kiseki::Core Kiseki::Spatify Kiseki::Deform)
endif()
add_library(Kiseki::FEM ALIAS FEM)
target_include_directories(FEM PUBLIC include)

set(FEM_ASSETS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/assets)
set(FEM_TETS_DIR ${FEM_ASSETS_DIR}/tets)
target_compile_definitions(FEM PUBLIC
    FEM_ASSETS_DIR="${FEM_ASSETS_DIR}"
    FEM_TETS_DIR="${FEM_TETS_DIR}"
)

# Developer test apps (headless)
add_executable(toy-ipc apps/toy-ipc.cc)
target_link_libraries(toy-ipc PUBLIC Kiseki::FEM)

enable_testing()
file(GLOB_RECURSE test_srcs CONFIGURE_DEPENDS tests/*.cc)
foreach(test_src ${test_srcs})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} PUBLIC Kiseki::FEM GTest::gtest GTest::gtest_main)
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

# Benchmarks
file(GLOB bench_srcs CONFIGURE_DEPENDS benchmarks/*.cc)
foreach(bench_src ${bench_srcs})
    get_filename_component(bench_name ${bench_src} NAME_WE)
    add_executable(${bench_name} ${bench_src})
    target_link_libraries(${bench_name} PUBLIC Kiseki::FEM)
endforeach()

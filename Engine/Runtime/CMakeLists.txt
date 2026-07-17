project(Runtime LANGUAGES CXX)

set(RUNTIME_SOURCES
    src/dof-layout.cc
    src/contact-detection.cc
    src/cpu/contact-detection-cpu.cc
    src/gpu/contact-detection-gpu.cc
        src/contact-barrier-energy.cc
    src/global-geometry-manager.cc
    src/subsystem-backend.cc
    src/runtime-scene.cc
        src/simulation-context.cc
    src/global-solver.cc
)

set(RUNTIME_HEADERS
    include/Runtime/types.h
    include/Runtime/buffers.h
    include/Runtime/contact-detection.h
        include/Runtime/contact-barrier-energy.h
    include/Runtime/dof-layout.h
    include/Runtime/global-geometry-manager.h
    include/Runtime/subsystem-backend.h
    include/Runtime/contact-table.h
    include/Runtime/subsystem.h
    include/Runtime/runtime-scene.h
        include/Runtime/simulation-context.h
    include/Runtime/global-solver.h
    include/Runtime/solver-config.h
    src/contact-detection-backends.h
)

add_library(Runtime STATIC ${RUNTIME_SOURCES} ${RUNTIME_HEADERS})
add_library(Kiseki::Runtime ALIAS Runtime)
target_include_directories(Runtime PUBLIC include)
target_link_libraries(Runtime PUBLIC Kiseki::Core Kiseki::RHI Kiseki::Spatify Kiseki::Maths Kiseki::EngineContactIPC Eigen3::Eigen)
target_compile_features(Runtime PUBLIC cxx_std_20)

enable_testing()
file(GLOB_RECURSE test_srcs CONFIGURE_DEPENDS tests/*.cc)
foreach(test_src ${test_srcs})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    target_include_directories(${test_name} PRIVATE ${CMAKE_SOURCE_DIR}/Engine/Scene/include)
    target_link_libraries(${test_name} PUBLIC Kiseki::Runtime GTest::gtest GTest::gtest_main)
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

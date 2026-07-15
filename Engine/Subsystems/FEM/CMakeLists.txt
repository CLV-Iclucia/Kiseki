project(EngineFEM LANGUAGES CXX)

set(ENGINE_FEM_SOURCES
    src/fem-scene.cc
    src/fem-subsystem.cc
    src/cpu/fem-cpu-backend.cc
)

set(ENGINE_FEM_HEADERS
    include/FEM/fem-scene.h
    include/FEM/fem-subsystem.h
    include/FEM/cpu/fem-cpu-backend.h
)

add_library(EngineFEM STATIC ${ENGINE_FEM_SOURCES} ${ENGINE_FEM_HEADERS})
add_library(Kiseki::EngineFEM ALIAS EngineFEM)
target_include_directories(EngineFEM PUBLIC include)
target_link_libraries(EngineFEM PUBLIC Kiseki::Core Kiseki::Runtime Kiseki::Deform Eigen3::Eigen)
target_compile_features(EngineFEM PUBLIC cxx_std_20)

enable_testing()
file(GLOB_RECURSE test_srcs CONFIGURE_DEPENDS tests/*.cc)
foreach(test_src ${test_srcs})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} PUBLIC
        Kiseki::EngineFEM
        GTest::gtest
        GTest::gtest_main
    )
    target_compile_features(${test_name} PUBLIC cxx_std_20)
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

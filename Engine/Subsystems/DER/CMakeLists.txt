project(DER LANGUAGES CXX)

set(DER_SOURCES
    src/rod.cc
    src/hair-scene.cc
    src/der-subsystem.cc
    src/der-scene.cc
    src/cpu/der-cpu-backend.cc
    src/gpu/der-gpu-backend.cc
)

set(DER_HEADERS
    include/DER/rod.h
    include/DER/der-subsystem.h
    include/DER/der-scene.h
    include/DER/cpu/der-cpu-backend.h
    include/DER/gpu/der-gpu-backend.h
)

add_library(DER STATIC ${DER_SOURCES} ${DER_HEADERS})
add_library(Kiseki::DER ALIAS DER)
target_include_directories(DER PUBLIC include)
target_link_libraries(DER PUBLIC Kiseki::Core Kiseki::Runtime Kiseki::SceneObjects Kiseki::EngineContactIPC Eigen3::Eigen)
target_compile_features(DER PUBLIC cxx_std_20)

enable_testing()
file(GLOB_RECURSE test_srcs CONFIGURE_DEPENDS tests/*.cc)
foreach(test_src ${test_srcs})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} PUBLIC
        Kiseki::DER
        GTest::gtest
        GTest::gtest_main
    )
    target_compile_features(${test_name} PUBLIC cxx_std_20)
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

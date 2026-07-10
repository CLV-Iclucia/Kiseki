project(HairSim LANGUAGES CXX)

add_library(HairSim STATIC
    src/rod.cc
    src/stretching-energy.cc
    src/bending-energy.cc
    src/twisting-energy.cc
    src/implicit-integrator.cc
)
add_library(Kiseki::Hair ALIAS HairSim)
target_include_directories(HairSim PUBLIC include)
target_link_libraries(HairSim PUBLIC Kiseki::Core Eigen3::Eigen)
target_compile_features(HairSim PUBLIC cxx_std_20)

add_executable(hair-sim apps/hair-sim-app.cc)
target_link_libraries(hair-sim PRIVATE Kiseki::Hair Kiseki::Renderer cxxopts)
target_compile_features(hair-sim PRIVATE cxx_std_20)

enable_testing()
add_executable(test-twisting-energy tests/test-twisting-energy.cc)
target_link_libraries(test-twisting-energy PRIVATE
    Kiseki::Hair
    GTest::gtest
    GTest::gtest_main
)
target_compile_features(test-twisting-energy PRIVATE cxx_std_20)
add_test(NAME test-twisting-energy COMMAND test-twisting-energy)

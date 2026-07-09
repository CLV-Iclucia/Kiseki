project(HairSim LANGUAGES CXX)

add_library(HairSim STATIC
    src/rod.cc
    src/stretching-energy.cc
    src/bending-energy.cc
    src/twisting-energy.cc
    src/implicit-integrator.cc
)
target_include_directories(HairSim PUBLIC include)
target_link_libraries(HairSim PUBLIC Core Eigen3::Eigen)
target_compile_features(HairSim PUBLIC cxx_std_20)

add_executable(hair-sim apps/hair-sim-app.cc)
target_link_libraries(hair-sim PRIVATE HairSim Renderer cxxopts)
target_compile_features(hair-sim PRIVATE cxx_std_20)

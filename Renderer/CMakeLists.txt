project(Renderer LANGUAGES CXX)

find_package(glfw3 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(glad CONFIG REQUIRED)
find_package(glad CONFIG REQUIRED)

# Collect source files
file(GLOB_RECURSE srcs CONFIGURE_DEPENDS src/*.cc src/*.cpp)

# Create static library
add_library(Renderer STATIC ${srcs})
add_library(Kiseki::Renderer ALIAS Renderer)

target_include_directories(Renderer PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(Renderer PUBLIC Kiseki::Core glfw OpenGL::GL glad::glad)

target_compile_features(Renderer PUBLIC cxx_std_20)

message(STATUS "Renderer configured with GLFW and OpenGL")

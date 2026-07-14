project(SceneObjects LANGUAGES CXX)

add_library(SceneObjects INTERFACE)
add_library(Kiseki::SceneObjects ALIAS SceneObjects)

target_include_directories(SceneObjects INTERFACE include)

target_link_libraries(SceneObjects INTERFACE Kiseki::Runtime)

project(SceneObjects LANGUAGES CXX)

file (GLOB_RECURSE SCENE_OBJECTS_SOURCES CONFIGURE_DEPENDS src/*.cc)
file (GLOB_RECURSE SCENE_OBJECTS_HEADERS CONFIGURE_DEPENDS include/*.h)

add_library(SceneObjects STATIC ${SCENE_OBJECTS_SOURCES} ${SCENE_OBJECTS_HEADERS})
add_library(Kiseki::SceneObjects ALIAS SceneObjects)

target_include_directories(SceneObjects PUBLIC include)

target_link_libraries(SceneObjects PUBLIC Kiseki::Runtime)
target_compile_features(SceneObjects PUBLIC cxx_std_20)

project(Scene LANGUAGES CXX)

set(SCENE_SOURCES
    ../SceneObjects/src/hair.cc
)

set(SCENE_HEADERS
    include/Scene/hair.h
        ../SceneObjects/include/SceneObjects/hair.h
)

add_library(Scene STATIC ${SCENE_SOURCES} ${SCENE_HEADERS})
add_library(Kiseki::Scene ALIAS Scene)

target_include_directories(Scene PUBLIC include ../SceneObjects/include)

target_link_libraries(Scene PUBLIC Kiseki::Runtime Kiseki::DER Kiseki::SceneObjects)

target_compile_features(Scene PUBLIC cxx_std_20)

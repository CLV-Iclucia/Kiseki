project(Core LANGUAGES CXX)

if (MSVC)
    add_compile_options(/Zc:preprocessor)
endif()

set(CORE_SOURCES
    src/json.cc
    src/obj-io.cc
    src/rand-gen.cc
)

add_library(Core STATIC ${CORE_SOURCES})
target_include_directories(Core PUBLIC include)
target_include_directories(Core PUBLIC ${EXTERNAL_PATH}/glm)
target_link_libraries(Core PUBLIC spdlog)

if(SIMCRAFT_PROFILE)
  target_link_libraries(Core PUBLIC Tracy::TracyClient)
endif()

enable_testing()
file(GLOB_RECURSE test_srcs CONFIGURE_DEPENDS tests/*.cc)
foreach(test_src ${test_srcs})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} PUBLIC Core GTest::gtest GTest::gtest_main)
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

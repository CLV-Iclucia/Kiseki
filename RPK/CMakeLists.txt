project(RPK LANGUAGES CXX)

# ============================================================================
# RPK — RHI Parallel Kit
# GPU parallel primitives: reduce, scan (prefix sum), radix sort.
# Built on top of the RHI compute dispatch infrastructure.
# ============================================================================

file(GLOB_RECURSE rpk_srcs    CONFIGURE_DEPENDS src/*.cc)
file(GLOB_RECURSE rpk_headers CONFIGURE_DEPENDS include/RPK/*.h)

add_library(RPK STATIC ${rpk_srcs} ${rpk_headers})

target_include_directories(RPK PUBLIC include/)

target_link_libraries(RPK PUBLIC RHI)

target_compile_features(RPK PUBLIC cxx_std_20)

target_compile_definitions(RPK PRIVATE
    RPK_SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/shaders"
)

# ---- Tests -----------------------------------------------------------------
enable_testing()
file(GLOB rpk_test_srcs CONFIGURE_DEPENDS tests/*.cc)
foreach(test_src ${rpk_test_srcs})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} PRIVATE RPK GTest::gtest GTest::gtest_main)
    add_test(NAME ${test_name} COMMAND ${test_name})

    if(WIN32 AND CMAKE_VERSION VERSION_GREATER_EQUAL "3.21")
        add_custom_command(TARGET ${test_name} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_RUNTIME_DLLS:${test_name}>"
                "$<TARGET_FILE_DIR:${test_name}>"
            COMMAND_EXPAND_LISTS
        )
    endif()
endforeach()

message(STATUS "RPK (RHI Parallel Kit) configured")

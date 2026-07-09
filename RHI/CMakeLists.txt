project(RHI LANGUAGES CXX)

if (MSVC)
    add_compile_options(/Zc:preprocessor)
endif()

# ---- Dependencies ----------------------------------------------------------
find_package(Vulkan REQUIRED)
find_package(VulkanMemoryAllocator CONFIG REQUIRED)
find_package(vk-bootstrap CONFIG REQUIRED)

# DXC for HLSL → SPIR-V/DXIL compilation (R3+).
# vcpkg port `directx-dxc` exports the Microsoft::DirectXShaderCompiler target
# along with the dxcompiler runtime DLL/SO. We link it PRIVATE because the
# DXC headers are an implementation detail of shader-compiler.cc; consumers
# of RHI never see <dxc/dxcapi.h>.
find_package(directx-dxc CONFIG REQUIRED)

# SPIRV-Reflect for SPIR-V → ReflectionInfo (R4+). vcpkg port `spirv-reflect`
# exports its CMake config under the `unofficial-spirv-reflect` package name
# with target `unofficial::spirv-reflect` (NOT `spirv-reflect-static` as plan
# §6.1 originally guessed — vcpkg port naming differs from the upstream lib's
# CMake target). PRIVATE linkage: reflect-spirv.cc is the only TU that
# #includes <spirv_reflect.h>; our public reflection.h carries POD types only.
find_package(unofficial-spirv-reflect CONFIG REQUIRED)

# ---- Source globs ----------------------------------------------------------
file(GLOB_RECURSE rhi_common  CONFIGURE_DEPENDS src/common/*.cc)
file(GLOB_RECURSE rhi_vulkan  CONFIGURE_DEPENDS src/vulkan/*.cc src/vulkan/*.h)
file(GLOB_RECURSE rhi_headers CONFIGURE_DEPENDS include/RHI/*.h)

# Make sure src/common still builds even if empty in this tranche.
if(NOT rhi_common)
    set(rhi_common "")
endif()

add_library(RHI STATIC ${rhi_common} ${rhi_vulkan} ${rhi_headers})

target_compile_definitions(RHI PRIVATE
    RHI_SHADER_INCLUDE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/shaders"
)

target_include_directories(RHI
    PUBLIC  include/
    PRIVATE src/
)

target_link_libraries(RHI
    PUBLIC  Core
            Vulkan::Vulkan
    PRIVATE GPUOpen::VulkanMemoryAllocator
            vk-bootstrap::vk-bootstrap
            Microsoft::DirectXShaderCompiler
            unofficial::spirv-reflect
)

target_compile_features(RHI PUBLIC cxx_std_20)

# ---- Tests -----------------------------------------------------------------
enable_testing()
file(GLOB test_srcs CONFIGURE_DEPENDS tests/*.cc)
foreach(test_src ${test_srcs})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} PUBLIC RHI GTest::gtest GTest::gtest_main)
    add_test(NAME ${test_name} COMMAND ${test_name})

    # Deploy runtime DLLs alongside the test exe on Windows.
    # dxcompiler.dll comes from vcpkg's directx-dxc port and isn't on PATH.
    # TARGET_RUNTIME_DLLS requires CMake 3.21+; the project requires 3.20, so
    # we guard.
    if(WIN32 AND CMAKE_VERSION VERSION_GREATER_EQUAL "3.21")
        add_custom_command(TARGET ${test_name} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_RUNTIME_DLLS:${test_name}>"
                "$<TARGET_FILE_DIR:${test_name}>"
            COMMAND_EXPAND_LISTS
        )
    endif()
endforeach()

message(STATUS "RHI configured (Vulkan ${Vulkan_VERSION}, VMA + vk-bootstrap, DXC)")

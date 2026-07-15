project(KisekiApps LANGUAGES CXX)

# IPC FEM Simulator with optional rendering
add_executable(ipc-fem ipc-fem/main.cc)
target_link_libraries(ipc-fem PUBLIC Kiseki::FEM cxxopts Kiseki::Renderer)
target_compile_features(ipc-fem PUBLIC cxx_std_20)

# Cube Drop example (equivalent to python/examples/cube_drop.py)
add_executable(cube-drop cube-drop/main.cc)
target_link_libraries(cube-drop PUBLIC Kiseki::FEM cxxopts Kiseki::Renderer)
target_compile_features(cube-drop PUBLIC cxx_std_20)

# Multi-Body Collision example (equivalent to python/examples/multi_body.py)
add_executable(multi-body multi-body/main.cc)
target_link_libraries(multi-body PUBLIC Kiseki::FEM cxxopts Kiseki::Renderer)
target_compile_features(multi-body PUBLIC cxx_std_20)

# Prescribed Motion example (equivalent to python/examples/prescribed_motion.py)
add_executable(prescribed-motion prescribed-motion/main.cc)
target_link_libraries(prescribed-motion PUBLIC Kiseki::FEM cxxopts Kiseki::Renderer)
target_compile_features(prescribed-motion PUBLIC cxx_std_20)

# Energy Conservation Check (equivalent to python/examples/energy_conservation.py)
add_executable(energy-check energy-check/main.cc)
target_link_libraries(energy-check PUBLIC Kiseki::FEM cxxopts Kiseki::Renderer)
target_compile_features(energy-check PUBLIC cxx_std_20)

# FEM Showcase: comprehensive multi-body + mixed-collider test scene
add_executable(fem-showcase fem-showcase/main.cc)
target_link_libraries(fem-showcase PUBLIC Kiseki::FEM cxxopts Kiseki::Renderer)
target_compile_features(fem-showcase PUBLIC cxx_std_20)

# DER Hair runtime smoke scene
add_executable(der-hair der-hair/main.cc)
    target_link_libraries(der-hair PUBLIC Kiseki::DER Kiseki::SceneObjects cxxopts Kiseki::Renderer)

# Fluid Dam Break simulation
add_executable(fluid-sim fluid-sim/main.cc)
target_link_libraries(fluid-sim PUBLIC Kiseki::Fluid cxxopts Kiseki::Renderer)
target_compile_features(fluid-sim PUBLIC cxx_std_20)

message(STATUS "Apps configured: ipc-fem, cube-drop, multi-body, prescribed-motion, energy-check, fem-showcase, der-hair, fluid-sim")

# ---- RHI demo apps (require Vulkan + GLFW, no OpenGL Renderer) ----
if(KISEKI_BUILD_RHI)
  find_package(glfw3 REQUIRED)

  add_executable(vk-triangle vk-triangle/main.cc)
  target_link_libraries(vk-triangle PUBLIC Kiseki::RHI glfw)
  target_compile_features(vk-triangle PUBLIC cxx_std_20)

  # GPU FEM backend driver (headless): CPU assembly + GPU-resident PCG.
  add_executable(gpu-fem gpu-fem/main.cc)
  target_link_libraries(gpu-fem PUBLIC Kiseki::FEM Kiseki::RHI cxxopts)
  target_compile_features(gpu-fem PUBLIC cxx_std_20)

  # Deploy runtime DLLs (dxcompiler.dll etc.) alongside the exe.
  if(WIN32 AND CMAKE_VERSION VERSION_GREATER_EQUAL "3.21")
    add_custom_command(TARGET vk-triangle POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_RUNTIME_DLLS:vk-triangle>"
            "$<TARGET_FILE_DIR:vk-triangle>"
        COMMAND_EXPAND_LISTS
    )
    add_custom_command(TARGET gpu-fem POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_RUNTIME_DLLS:gpu-fem>"
            "$<TARGET_FILE_DIR:gpu-fem>"
        COMMAND_EXPAND_LISTS
    )
  endif()

  message(STATUS "  + vk-triangle (RHI R7 demo)")
  message(STATUS "  + gpu-fem (GPU FEMBackend driver)")
endif()

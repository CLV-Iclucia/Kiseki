project(EngineContactIPC LANGUAGES CXX)

set(ENGINE_CONTACT_IPC_SOURCES
    src/contact-barrier.cc
    src/gipc/pfpx.cc
)

set(ENGINE_CONTACT_IPC_HEADERS
    include/Contact/contact-barrier.h
    include/Contact/types.h
    include/Contact/gipc/barrier.h
    include/Contact/gipc/hessian.h
    include/Contact/gipc/mollifier.h
    include/Contact/gipc/pfpx.h
    include/Contact/gipc/external/GIPC_PDerivative.h
    include/Contact/shared/cross-lang.h
    include/Contact/shared/gipc-barrier.h
    include/Contact/shared/gipc-mollifier.h
    include/Contact/shared/gipc-pfpx.h
    include/Contact/shared/ipc-activation.h
    include/Contact/shared/ipc-ccd.h
    include/Contact/shared/ipc-distance-type.h
    include/Contact/shared/ipc-distance.h
)

add_library(EngineContactIPC STATIC
    ${ENGINE_CONTACT_IPC_SOURCES}
    ${ENGINE_CONTACT_IPC_HEADERS}
)
add_library(Kiseki::EngineContactIPC ALIAS EngineContactIPC)
target_include_directories(EngineContactIPC PUBLIC include ${EXTERNAL_PATH}/glm)
target_link_libraries(EngineContactIPC PUBLIC Kiseki::Maths Eigen3::Eigen)
target_compile_features(EngineContactIPC PUBLIC cxx_std_20)
if(MSVC)
    target_compile_options(EngineContactIPC PUBLIC /utf-8)
endif()

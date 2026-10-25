set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(GVS_GPU "Build the Vulkan GPU backend" ON)

find_package(Threads REQUIRED)

add_library(gvs_core STATIC
    src/cpu_engine.cpp
    src/engine.cpp
    src/net.cpp
    src/protocol.cpp
    src/sim.cpp
    src/view.cpp)
target_include_directories(gvs_core PUBLIC src)
# CPU and GPU must round floats identically for the results to match bit for bit.
target_compile_options(gvs_core PUBLIC -ffp-contract=off -Wall -Wextra)
target_link_libraries(gvs_core PUBLIC Threads::Threads)

if(GVS_GPU)
    find_package(Vulkan REQUIRED)
    find_program(GLSLANG_VALIDATOR glslangValidator REQUIRED)
    set(SPV_DIR ${CMAKE_CURRENT_BINARY_DIR}/spv)
    file(MAKE_DIRECTORY ${SPV_DIR})
    set(SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
    set(SPV_HEADERS)
    foreach(s upd derive clear hist0 decide scatter0 advance hist scatter final)
        set(out ${SPV_DIR}/${s}.spv.h)
        add_custom_command(
            OUTPUT ${out}
            COMMAND ${GLSLANG_VALIDATOR} -V --target-env vulkan1.1 --vn spv_${s} -o ${out} ${SHADER_DIR}/${s}.comp
            DEPENDS ${SHADER_DIR}/${s}.comp ${SHADER_DIR}/common.glsl
            COMMENT "glslang ${s}.comp"
            VERBATIM)
        list(APPEND SPV_HEADERS ${out})
    endforeach()
    target_sources(gvs_core PRIVATE src/gpu_engine.cpp ${SPV_HEADERS})
    target_include_directories(gvs_core PRIVATE ${SPV_DIR})
    target_compile_definitions(gvs_core PUBLIC GVS_GPU=1)
    target_link_libraries(gvs_core PUBLIC Vulkan::Vulkan)
endif()

add_executable(gvs_server src/server_main.cpp)
target_link_libraries(gvs_server PRIVATE gvs_core)

add_executable(gvs_bench src/bench_main.cpp)
target_link_libraries(gvs_bench PRIVATE gvs_core)

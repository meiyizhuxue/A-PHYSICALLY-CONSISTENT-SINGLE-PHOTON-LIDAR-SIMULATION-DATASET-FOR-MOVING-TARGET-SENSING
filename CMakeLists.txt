cmake_minimum_required(VERSION 3.18)

project(sim LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

# Configure CUDA compilation options
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

# Locate the CUDA toolkit (required)
find_package(CUDAToolkit REQUIRED)

add_executable(sim
    main.cpp
    pixel_simulator.cpp
    spad_simulator.cu
)

# Engineering Header File
target_include_directories(sim PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    D:/Software/VScode/include/eigen-3.4.0
)

# Link CUDA libraries
target_link_libraries(sim PRIVATE CUDA::cudart)

# CUDA Architecture Configuration (Explicitly Specifying RTX 4060's 8.9)
set_target_properties(sim PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "86;89"  # 8.6作为兼容，8.9为最新
)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(sim PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 -use_fast_math>
    )
endif()


if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(sim PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-G -lineinfo>
    )
endif()
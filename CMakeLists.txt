cmake_minimum_required(VERSION 3.18)
project(cvdv LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set CUDA architectures - auto-detect or use common architectures
set(CMAKE_CUDA_ARCHITECTURES 89)  # For compute capability 8.9

find_package(CUDAToolkit REQUIRED)

add_library(cvdv SHARED src/cvdv.cu)

target_link_libraries(cvdv PRIVATE
    CUDA::cudart
    CUDA::cufft
)

set_target_properties(cvdv PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

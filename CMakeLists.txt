cmake_minimum_required(VERSION 3.18)
project(cvdv LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set CUDA architectures - auto-detect or use common architectures
set(CMAKE_CUDA_ARCHITECTURES native)

find_package(CUDAToolkit REQUIRED)

# Manually find cuTENSOR (installed via apt)
find_path(CUTENSOR_INCLUDE_DIR NAMES cutensor.h PATHS /usr/include/libcutensor/13)
find_library(CUTENSOR_LIBRARY NAMES cutensor PATHS /usr/lib/x86_64-linux-gnu/libcutensor/13)
if(NOT CUTENSOR_INCLUDE_DIR OR NOT CUTENSOR_LIBRARY)
    message(FATAL_ERROR "cuTENSOR not found. Install: sudo apt-get install libcutensor2-dev-cuda-13")
endif()

add_library(cvdv SHARED src/cvdv.cu)

target_include_directories(cvdv PRIVATE ${CUTENSOR_INCLUDE_DIR})
target_link_libraries(cvdv PRIVATE
    CUDA::cudart
    CUDA::cufft
    ${CUTENSOR_LIBRARY}
)

set_target_properties(cvdv PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

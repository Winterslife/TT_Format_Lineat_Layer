# Author:  Johannes de Fine Licht (definelicht@inf.ethz.ch)
# Copyright: This software is copyrighted under the BSD 3-Clause License.

cmake_minimum_required(VERSION 3.0)
project(sdaccel_mm)

include(${CMAKE_SOURCE_DIR}/config.cmake)

# 设置Vitis HLS路径
set(VITIS_HLS_ROOT $ENV{XILINX_HLS})
if(NOT VITIS_HLS_ROOT)
    message(FATAL_ERROR "XILINX_HLS environment variable not set")
endif()
# 需要添加 TT 相关的源文件
set(TT_KERNEL_SRC
    ${CMAKE_SOURCE_DIR}/kernel/TTLinear.cpp)

# TT 配置选项
set(TT_MAX_DIMS "8" CACHE STRING "Maximum number of TT cores")
set(TT_MAX_RANK "128" CACHE STRING "Maximum TT rank")
set(TT_MAX_MODE_SIZE "512" CACHE STRING "Maximum size per mode")
set(TT_MAX_BATCH_SIZE "1" CACHE STRING "Maximum batch size")

# Additional TT configuration options (matching Tucker's style)
set(TT_INPUT_DIM "1024" CACHE STRING "Input dimension")
set(TT_OUTPUT_DIM "512" CACHE STRING "Input dimension")
set(TT_MEMORY_WIDTH_BYTES "64" CACHE STRING "Memory width in bytes")

# Memory configurations for TT
set(TT_MEMORY_BUS_WIDTH_INPUT "64" CACHE STRING "Width of memory bus in bytes for input")
set(TT_MEMORY_BUS_WIDTH_OUTPUT "64" CACHE STRING "Width of memory bus in bytes for output")
set(TT_MEMORY_BUS_WIDTH_CORES "64" CACHE STRING "Width of memory bus in bytes for TT cores")


# 确保在configure_file之前设置这些变量
configure_file(
    ${CMAKE_SOURCE_DIR}/include/Config.h.in
    ${CMAKE_BINARY_DIR}/Config.h
)

set(VITIS_HLS_LIBS
    ${VITIS_HLS_ROOT}/lib/lnx64.o/libhlsmath.so
    ${VITIS_HLS_ROOT}/lnx64/tools/fpo_v7_1/libIp_floating_point_v7_1_bitacc_cmodel.so
    ${VITIS_HLS_ROOT}/lnx64/tools/fpo_v7_1/libgmp.so
    ${VITIS_HLS_ROOT}/lnx64/tools/fpo_v7_1/libmpfr.so
    /opt/xilinx/xrt/lib/libxilinxopencl.so
    stdc++
    m
    pthread
)
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}    # 包含生成的Config.h
    ${VITIS_HLS_ROOT}/include
)
# CMakeLists.txt

# 设置数据类型选项
set(MM_DATA_TYPE "half" CACHE STRING "Matrix data type.")

# 当选择half类型时开启MM_HALF_PRECISION
if(MM_DATA_TYPE STREQUAL "half")
    add_definitions(-DMM_HALF_PRECISION)
    
    # 确保包含half相关的库和头文件
    include_directories(${VITIS_HLS_ROOT}/include)
    link_libraries(${VITIS_HLS_ROOT}/lib/lnx64.o/libhlsmath.so)
endif()

# 配置头文件
configure_file(
    ${CMAKE_SOURCE_DIR}/include/Config.h.in
    ${CMAKE_BINARY_DIR}/Config.h
)

# 设置模块路径
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake
    ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/hlslib/cmake)

# 包含目录设置
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}
    ${CMAKE_SOURCE_DIR}/hlslib/include  # 确保这一行存在
    ${VITIS_HLS_ROOT}/include
)

# 配置头文件
configure_file(
    ${CMAKE_SOURCE_DIR}/include/Config.h.in
    ${CMAKE_BINARY_DIR}/Config.h
)
set(MM_KERNEL_SRC
    ${CMAKE_SOURCE_DIR}/kernel/Compute.cpp
    ${CMAKE_SOURCE_DIR}/kernel/Memory.cpp
    ${CMAKE_SOURCE_DIR}/kernel/Top.cpp)

# Tucker source files
set(TUCKER_KERNEL_SRC
    ${CMAKE_SOURCE_DIR}/kernel/TuckerLinear.cpp)

# Tucker configuration options
set(TUCKER_INPUT_DIM "1024" CACHE STRING "Input dimension I")
set(TUCKER_OUTPUT_DIM "512" CACHE STRING "Output dimension O") 
# Batch size固定为1
set(TUCKER_BATCH_SIZE "1" CACHE STRING "Batch size fixed to 1")
set(TUCKER_MAX_RANK_1 "128" CACHE STRING "First Tucker rank R1")
set(TUCKER_MAX_RANK_2 "128" CACHE STRING "Second Tucker rank R2")
set(MM_SIZE_R2 "64" CACHE STRING "Size of Tucker rank R2")
# Add Tucker configuration to synthesis flags
set(MM_SYNTHESIS_FLAGS "${MM_SYNTHESIS_FLAGS} \
    -DTUCKER_MAX_INPUT_DIM=${TUCKER_MAX_INPUT_DIM} \
    -DTUCKER_MAX_OUTPUT_DIM=${TUCKER_MAX_OUTPUT_DIM} \
    -DTUCKER_MAX_BATCH_SIZE=${TUCKER_MAX_BATCH_SIZE} \
    -DTUCKER_MAX_RANK_1=${TUCKER_MAX_RANK_1} \
    -DTUCKER_MAX_RANK_2=${TUCKER_MAX_RANK_2}")

# Add TT configuration to synthesis flags
set(MM_SYNTHESIS_FLAGS "${MM_SYNTHESIS_FLAGS} \
    -DTT_MAX_DIMS=${TT_MAX_DIMS} \
    -DTT_MAX_RANK=${TT_MAX_RANK} \
    -DTT_MAX_MODE_SIZE=${TT_MAX_MODE_SIZE} \
    -DTT_MAX_BATCH_SIZE=${TT_MAX_BATCH_SIZE} \
    -DTT_INPUT_DIM=${TT_INPUT_DIM} \
    -DTT_OUTPUT_DIM=${TT_OUTPUT_DIM} \
    -DTT_MEMORY_WIDTH_BYTES=${TT_MEMORY_WIDTH_BYTES}")

# Target options 
set(MM_PLATFORM "xilinx_u50_gen3x16_xdma_5_202210_1" CACHE STRING "Platform string for Vitis.")
set(MM_TARGET_CLOCK "" CACHE STRING "Target clock for kernel (uses default is left empty).")
set(MM_ENABLE_PROFILING OFF CACHE BOOL "Collect profiling information.")
set(MM_ENABLE_DEBUGGING OFF CACHE BOOL "Inject debugging cores to design.")
set(MM_VITIS_FLAGS "" CACHE STRING "Extra flags for Vitis.")
set(MM_POWER_METER OFF CACHE BOOL "Enable Corsair power meter for measuring power consumption with Corsair RMi power supplies.")
set(MM_ENABLE_BLAS ON CACHE BOOL "Use BLAS library for validation if available.")

# Domain options
set(MM_DATA_TYPE "half" CACHE STRING "Matrix data type.")
set(MM_MEMORY_BUS_WIDTH_N 64 CACHE STRING "Width of memory bus in bytes in N.")
set(MM_MEMORY_BUS_WIDTH_K 64 CACHE STRING "Width of memory bus in bytes in K.")
set(MM_MEMORY_BUS_WIDTH_M 64 CACHE STRING "Width of memory bus in bytes in M.")
set(MM_DYNAMIC_SIZES ON CACHE BOOL "Use dynamic matrix dimension sizes.")
set(MM_SIZE_N 512 CACHE STRING "Size of matrix dimension.")
set(MM_SIZE_K 512 CACHE STRING "Size of matrix dimension.")
set(MM_SIZE_M 512 CACHE STRING "Size of matrix dimension.")
set(MM_MEMORY_TILE_SIZE_N 256 CACHE STRING "Tile size of outer memory tile in N.")
set(MM_MEMORY_TILE_SIZE_M 256 CACHE STRING "Tile size of outer memory tile in M.")
set(MM_PARALLELISM_N 32 CACHE STRING "Number of parallel compute in N.")
set(MM_PARALLELISM_M 8 CACHE STRING "Number of parallel compute in M.")
# set(MM_GRANULARITY_N 1 CACHE STRING "Granularity of processing elements in N.")
set(MM_TRANSPOSED_A OFF CACHE BOOL "Assume the input matrix A is transposed.")
set(MM_TRANSPOSE_WIDTH 64 CACHE STRING "Burst width when transposing reads from A.")
set(MM_TWO_DIMMS OFF CACHE BOOL "Use two DDR DIMMs instead of one")
set(MM_MAP_OP "Multiply" CACHE STRING "Map operation to perform between A and B matrices.")
set(MM_REDUCE_OP "Add" CACHE STRING "Reduction operation to write back to C.")
set(MM_ADD_RESOURCE OFF CACHE STRING "")
set(MM_MULT_RESOURCE OFF CACHE STRING "")  

# Internal
set(MM_KERNEL_NAME MatrixMultiplicationKernel)
include(CheckTypeSize)
check_type_size(${MM_DATA_TYPE} MM_DATA_WIDTH_${MM_DATA_TYPE})
if(NOT MM_DATA_WIDTH_${MM_DATA_TYPE}) # Non-primitive data type
  if(MM_DATA_TYPE STREQUAL "half")
    set(MM_DATA_WIDTH_${MM_DATA_TYPE} 2)
  elseif(MM_DATA_TYPE STREQUAL "uint8_t")
    set(MM_DATA_WIDTH_${MM_DATA_TYPE} 1)
  else()
    message(FATAL_ERROR "Could not get size of data type ${MM_DATA_TYPE}.") 
  endif()
endif()
math(EXPR MM_KERNEL_WIDTH_N "${MM_DATA_WIDTH_${MM_DATA_TYPE}} * ${MM_PARALLELISM_N}")
math(EXPR MM_KERNEL_WIDTH_M "${MM_DATA_WIDTH_${MM_DATA_TYPE}} * ${MM_PARALLELISM_M}")

link_directories(
    ${VITIS_HLS_ROOT}/lib/lnx64.o
    ${VITIS_HLS_ROOT}/lnx64/tools/fpo_v7_1
)

# Validation checks
math(EXPR MM_INNER_TILES "(${MM_MEMORY_TILE_SIZE_N} / ${MM_PARALLELISM_N}) * (${MM_MEMORY_TILE_SIZE_M} / ${MM_PARALLELISM_M})")
if(NOT MM_TRANSPOSED_A AND MM_MEMORY_TILE_SIZE_N GREATER MM_INNER_TILES)
  message(WARNING "In-memory transposition for A cannot keep up with the instantiated number of compute units. The number of inner tiles (currently ${MM_INNER_TILES}) must be greater than or equal to the outer tile size in N (currently ${MM_MEMORY_TILE_SIZE_N}).")
endif()
math(EXPR MM_MEMORY_WIDTH_M "${MM_MEMORY_BUS_WIDTH_M} / ${MM_DATA_WIDTH_${MM_DATA_TYPE}}")
math(EXPR MM_REM "${MM_MEMORY_WIDTH_M} % ${MM_PARALLELISM_M}")
if(MM_REM GREATER 0)
  message(FATAL_ERROR "Bus width in M (${MM_MEMORY_WIDTH_M}) must be a multiple of the parallelism in M (${MM_PARALLELISM_M}).")
endif()
math(EXPR MM_REM "${MM_MEMORY_TILE_SIZE_M} % ${MM_MEMORY_WIDTH_M}")
if(MM_REM GREATER 0)
  message(FATAL_ERROR "Outer memory tile size must be divisible by element width of memory bus.")
endif()

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake
    ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/hlslib/cmake)

find_package(Vitis REQUIRED)
find_package(Threads REQUIRED)

# Check if BLAS is available
if(MM_ENABLE_BLAS)
  find_package(BLAS)
  if(BLAS_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMM_HAS_BLAS")
  else()
    set(BLAS_LIBRARIES)
  endif()
else()
  set(BLAS_LIBRARIES)
endif()

# Include power meter
if(MM_POWER_METER)
  add_subdirectory(powermeter)
  include_directories(SYSTEM powermeter/include powermeter/OpenCorsairLink/include ${LIBUSB_INCLUDE_DIR})
  add_definitions("-DMM_POWER_METER")
endif()

# Testing configuration for TT
if(MM_DYNAMIC_SIZES)
    add_test(TestTTSimulation TestTTSimulation 
             ${TT_MAX_DIMS} ${TT_MAX_RANK} ${TT_MAX_MODE_SIZE})
else()
    add_test(TestTTSimulation TestTTSimulation)
endif()

include_directories(include ${CMAKE_BINARY_DIR} SYSTEM hlslib/include ${Vitis_INCLUDE_DIRS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
if(MM_DYNAMIC_SIZES)
  add_definitions("-DMM_DYNAMIC_SIZES")
  set(MM_SYNTHESIS_FLAGS "${MM_SYNTHESIS_FLAGS} -DMM_DYNAMIC_SIZES")
endif()
if(MM_TRANSPOSED_A)
  add_definitions("-DMM_TRANSPOSED_A")
  set(MM_SYNTHESIS_FLAGS "${MM_SYNTHESIS_FLAGS} -DMM_TRANSPOSED_A")
endif()
add_definitions("-DHLSLIB_STREAM_TIMEOUT=16")
if(((${Vitis_MAJOR_VERSION} LESS 2018) AND (${Vitis_MINOR_VERSION} LESS 3)) OR ${Vitis_MAJOR_VERSION} LESS 2017)
  add_definitions(-DHLSLIB_LEGACY_SDX=1)
else()
  add_definitions(-DHLSLIB_LEGACY_SDX=0)
endif()
if(MM_DATA_TYPE STREQUAL "half")
  add_definitions("-DMM_HALF_PRECISION")
endif()

# Query default clock of platform
if(NOT "${MM_TARGET_CLOCK}")
  if(NOT "${MM_PLATFORM_INTERNAL}" STREQUAL "${MM_PLATFORM}")
    message(STATUS "Querying default clock for ${MM_PLATFORM}.")
    execute_process(COMMAND ${Vitis_PLATFORMINFO} --platform ${MM_PLATFORM} -jhardwarePlatform.systemClocks
                    OUTPUT_VARIABLE SYSTEM_CLOCKS
                    RESULT_VARIABLE RET)
    if(RET EQUAL 0)
      string(FIND "${SYSTEM_CLOCKS}" "\"default\": \"true\"" LOC)
      string(SUBSTRING "${SYSTEM_CLOCKS}" ${LOC} -1 SYSTEM_CLOCKS)
      string(REGEX MATCH "\"frequency\": \"[0-9\\.]+\"" SYSTEM_CLOCKS "${SYSTEM_CLOCKS}")
      string(REGEX REPLACE "\"frequency\": \"([0-9\\.]+)\"" "\\1" CLOCK "${SYSTEM_CLOCKS}")
      set(MM_CLOCK_INTERNAL "${CLOCK}" CACHE INTERNAL "")
      set(MM_PLATFORM_INTERNAL "${MM_PLATFORM}" CACHE INTERNAL "")
    else()
      message(WARNING "Failed to query default frequency for platform ${MM_PLATFORM}. Assuming 300 MHz.")
      set(MM_CLOCK_INTERNAL "300" CACHE INTERNAL "")
    endif()
  endif()
endif()

# Hardware configuration header for HLS
configure_file(include/Config.h.in Config.h)

# C++ source code 
set(MM_KERNEL_SRC
    ${CMAKE_SOURCE_DIR}/kernel/Compute.cpp
    ${CMAKE_SOURCE_DIR}/kernel/Memory.cpp
    ${CMAKE_SOURCE_DIR}/kernel/Top.cpp)
add_library(mmkernel ${MM_KERNEL_SRC})
target_link_libraries(mmkernel 
    PRIVATE ${CMAKE_THREAD_LIBS_INIT}
)

# Executables 
add_executable(PrintSpecifications src/PrintSpecifications.cpp)

# Software test
add_executable(TestSimulation test/TestSimulation.cpp)
target_link_libraries(TestSimulation PRIVATE ${Vitis_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${BLAS_LIBRARIES} ${Vitis_LIBRARIES} mmkernel)

# Create a library for TT implementation
add_library(tt_lib ${TT_KERNEL_SRC})
target_link_libraries(tt_lib 
    PRIVATE 
    ${Vitis_LIBRARIES} 
    ${CMAKE_THREAD_LIBS_INIT} 
    ${BLAS_LIBRARIES} 
    mmkernel)

# Update the TestTTSimulation target
add_executable(TestTTSimulation test/TestTTSimulation.cpp)
target_link_libraries(TestTTSimulation 
    PRIVATE
    tt_lib          # Link with our TT library
    ${Vitis_LIBRARIES} 
    ${CMAKE_THREAD_LIBS_INIT} 
    ${BLAS_LIBRARIES} 
    mmkernel)

# Make sure include directories are set correctly for both targets
target_include_directories(tt_lib PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/hlslib/include
    ${CMAKE_BINARY_DIR})

target_include_directories(TestTTSimulation PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/hlslib/include
    ${CMAKE_BINARY_DIR})

  # If using half precision
if(MM_DATA_TYPE STREQUAL "half")
    target_link_libraries(tt_lib PRIVATE ${VITIS_HLS_LIBS})
    target_link_libraries(TestTTSimulation PRIVATE ${VITIS_HLS_LIBS})
endif()
# Half precision support
if(MM_DATA_TYPE STREQUAL "half")
    add_definitions("-DMM_HALF_PRECISION")
    target_link_libraries(mmkernel 
        PRIVATE 
        ${VITIS_HLS_LIBS}
    )
    target_link_libraries(TestSimulation 
        PRIVATE 
        ${VITIS_HLS_LIBS}
    )
endif()

add_library(tuckerkernel ${TUCKER_KERNEL_SRC})
target_link_libraries(tuckerkernel 
    ${CMAKE_THREAD_LIBS_INIT}
    mmkernel)  # Link against original matrix multiplication

# Tucker testing
add_executable(TestTuckerSimulation test/TestTuckerSimulation.cpp)
target_link_libraries(TestTuckerSimulation 
    ${Vitis_LIBRARIES} 
    ${CMAKE_THREAD_LIBS_INIT} 
    ${BLAS_LIBRARIES} 
    tuckerkernel)  # Link against Tucker kernel

if(MM_DATA_TYPE STREQUAL "half")
    target_link_libraries(TestTuckerSimulation ${Vitis_FLOATING_POINT_LIBRARY})
endif()

# Make sure include directories are set correctly
target_include_directories(mmkernel PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/hlslib/include
    ${CMAKE_BINARY_DIR})

target_include_directories(tuckerkernel PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/hlslib/include
    ${CMAKE_BINARY_DIR})

target_include_directories(TestTuckerSimulation PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/hlslib/include
    ${CMAKE_BINARY_DIR})

# Testing configuration
if(MM_DYNAMIC_SIZES)
    math(EXPR TEST_SIZE_N "2 * ${MM_MEMORY_TILE_SIZE_N} + 1")
    math(EXPR TEST_SIZE_K "2 * ${MM_PARALLELISM_N} * ${MM_PARALLELISM_M} + ${MM_MEMORY_BUS_WIDTH_K} / ${MM_DATA_WIDTH_${MM_DATA_TYPE}}")
    math(EXPR TEST_SIZE_M "2 * ${MM_MEMORY_TILE_SIZE_M} + ${MM_MEMORY_WIDTH_M}")
    add_test(TestTuckerSimulation TestTuckerSimulation ${TEST_SIZE_N} ${TEST_SIZE_K} ${TEST_SIZE_M})
else()
    add_test(TestTuckerSimulation TestTuckerSimulation)
endif()

# 设置运行时链接路径
set_target_properties(mmkernel PROPERTIES
    INSTALL_RPATH "${VITIS_HLS_ROOT}/lib/lnx64.o;${VITIS_HLS_ROOT}/lnx64/tools/fpo_v7_1"
    BUILD_WITH_INSTALL_RPATH TRUE
    LINK_FLAGS "-Wl,--no-as-needed"
)

set_target_properties(TestSimulation PROPERTIES
    INSTALL_RPATH "${VITIS_HLS_ROOT}/lib/lnx64.o;${VITIS_HLS_ROOT}/lnx64/tools/fpo_v7_1"
    BUILD_WITH_INSTALL_RPATH TRUE
    LINK_FLAGS "-Wl,--no-as-needed"
)


# Synthesis flags
set(MM_SYNTHESIS_FLAGS "${MM_SYNTHESIS_FLAGS} -O3 -DMM_SYNTHESIS")
if(MM_DATA_TYPE STREQUAL "half")
  set(MM_SYNTHESIS_FLAGS "${MM_SYNTHESIS_FLAGS} -DMM_HALF_PRECISION")
endif()

if(MM_ADD_RESOURCE)
  set(MM_SYNTHESIS_FLAGS "${MM_SYNTHESIS_FLAGS} -DMM_ADD_RESOURCE=${MM_ADD_RESOURCE}") 
endif() 
if(MM_MULT_RESOURCE) 
  set(MM_SYNTHESIS_FLAGS "${MM_SYNTHESIS_FLAGS} -DMM_MULT_RESOURCE=${MM_MULT_RESOURCE}") 
endif() 

if(MM_TWO_DIMMS)
  set(MM_PORT_MAPPING m_axi_gmem0:DDR[0] m_axi_gmem1:DDR[1] m_axi_gmem2:DDR[1])
else()
  set(MM_PORT_MAPPING m_axi_gmem0:DDR[1] m_axi_gmem1:DDR[1] m_axi_gmem2:DDR[1])
endif()
# Hardware kernel
add_vitis_kernel(TuckerKernel
                KERNEL "TuckerLinearKernel"
                FILES ${TUCKER_KERNEL_SRC}
                HLS_FLAGS ${MM_SYNTHESIS_FLAGS}
                INCLUDE_DIRS include
                            hlslib/include
                            ${CMAKE_BINARY_DIR})

# Add TT hardware kernel configuration
add_vitis_kernel(TTKernel
                KERNEL "TTLinearKernel"
                FILES ${TT_KERNEL_SRC}
                HLS_FLAGS ${MM_SYNTHESIS_FLAGS}
                INCLUDE_DIRS include
                            hlslib/include
                            ${CMAKE_BINARY_DIR})

# 添加综合和实现目标
add_vitis_program(TuckerKernel ${MM_PLATFORM} 
                 CLOCK ${MM_TARGET_CLOCK}
                 BUILD_FLAGS ${MM_VITIS_FLAGS}
                 PROFILING ${MM_ENABLE_PROFILING}
                 DEBUGGING ${MM_ENABLE_DEBUGGING})
              
# Add synthesis and implementation targets
add_vitis_program(TTKernel ${MM_PLATFORM} 
                 CLOCK ${MM_TARGET_CLOCK}
                 BUILD_FLAGS ${MM_VITIS_FLAGS}
                 PROFILING ${MM_ENABLE_PROFILING}
                 DEBUGGING ${MM_ENABLE_DEBUGGING})

# Hardware kernel
add_vitis_kernel(MatrixMultiplication
                 KERNEL "MatrixMultiplicationKernel"
                 FILES ${MM_KERNEL_SRC}
                 HLS_FLAGS ${MM_SYNTHESIS_FLAGS}
                 INCLUDE_DIRS include
                              hlslib/include
                              ${CMAKE_BINARY_DIR}
                 PORT_MAPPING ${MM_PORT_MAPPING} 
                 DEPENDS include/Compute.h
                         include/MatrixMultiplication.h
                         include/Memory.h
                         ${CMAKE_BINARY_DIR}/Config.h)
add_vitis_program(MatrixMultiplication ${MM_PLATFORM}
                  CLOCK ${MM_TARGET_CLOCK}
                  BUILD_FLAGS ${MM_VITIS_FLAGS}
                  PROFILING ${MM_ENABLE_PROFILING}
                  DEBUGGING ${MM_ENABLE_DEBUGGING})

# Host code to launch kernel
add_executable(RunHardware.exe host/RunHardware.cpp)
target_link_libraries(RunHardware.exe ${Vitis_LIBRARIES} ${BLAS_LIBRARIES} mmkernel)
if(MM_TWO_DIMMS)
  target_compile_definitions(RunHardware.exe PRIVATE MM_TWO_DIMMS)
endif()
if(MM_DATA_TYPE STREQUAL "half")
  target_link_libraries(RunHardware.exe ${Vitis_FLOATING_POINT_LIBRARY})
endif()
if(MM_POWER_METER)
  target_link_libraries(RunHardware.exe powermeter)
endif()

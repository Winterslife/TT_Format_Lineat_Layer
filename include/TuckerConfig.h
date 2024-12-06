/// @author    Your Name
/// @copyright This software is copyrighted under the BSD 3-Clause License.

#pragma once

#include "Config.h"
#include "hlslib/xilinx/DataPack.h"

// Tucker configuration parameters 
// Can be modified through CMake
#ifndef TUCKER_MAX_INPUT_DIM
#define TUCKER_MAX_INPUT_DIM 1024
#endif

#ifndef TUCKER_MAX_OUTPUT_DIM  
#define TUCKER_MAX_OUTPUT_DIM 1024
#endif

#ifndef TUCKER_MAX_BATCH_SIZE
#define TUCKER_MAX_BATCH_SIZE 256  
#endif

#ifndef TUCKER_MAX_RANK_1
#define TUCKER_MAX_RANK_1 128
#endif

#ifndef TUCKER_MAX_RANK_2  
#define TUCKER_MAX_RANK_2 128
#endif

// Memory width configurations for Tucker matrices
constexpr int kTuckerMemoryWidthU1 = kMemoryWidthBytesK / sizeof(Data_t);
constexpr int kTuckerMemoryWidthU2 = kMemoryWidthBytesM / sizeof(Data_t);
constexpr int kTuckerMemoryWidthS = kMemoryWidthBytesM / sizeof(Data_t);

// Memory pack types for Tucker matrices
using TuckerPackU1_t = hlslib::DataPack<Data_t, kTuckerMemoryWidthU1>;
using TuckerPackU2_t = hlslib::DataPack<Data_t, kTuckerMemoryWidthU2>;
using TuckerPackS_t = hlslib::DataPack<Data_t, kTuckerMemoryWidthS>;
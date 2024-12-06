/// @author    Your Name
/// @copyright This software is copyrighted under the BSD 3-Clause License.

#pragma once

#include "Config.h"
#include "hlslib/xilinx/DataPack.h"

// TT层的基本配置参数
// 可以通过CMake修改
#ifndef TT_MAX_DIMS
#define TT_MAX_DIMS 8           // 最大维度数(cores数量)
#endif

#ifndef TT_MAX_RANK
#define TT_MAX_RANK 128         // 最大TT秩
#endif

#ifndef TT_MAX_MODE_SIZE
#define TT_MAX_MODE_SIZE 128   // 每个模式的最大大小
#endif

#ifndef TT_MAX_BATCH_SIZE
#define TT_MAX_BATCH_SIZE 32   // 最大batch size
#endif

// Use TTMemoryPack_t defined with kTTMemoryWidth from Config.h
using TTMemoryPack_t = hlslib::DataPack<Data_t, kTTMemoryWidth>;

// Buffer size calculations
constexpr unsigned kMaxCoreSizeMemory = 
    (TT_MAX_RANK * TT_MAX_MODE_SIZE * TT_MAX_RANK * TT_MAX_MODE_SIZE 
    + kTTMemoryWidth - 1) / kTTMemoryWidth;

// Temporary buffer size calculation
constexpr unsigned kMaxIntermediateSize = 
    TT_MAX_MODE_SIZE * TT_MAX_MODE_SIZE * TT_MAX_RANK;

    
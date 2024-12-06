/// @author    Your Name
/// @copyright This software is copyrighted under the BSD 3-Clause License.

#pragma once

#include "TTTypes.h"
#include "TTConfig.h"
#include "MatrixMultiplication.h"
#include "hlslib/xilinx/Stream.h"
#include "hlslib/xilinx/DataPack.h"

namespace tt {

// TT层的初始化和计算函数
void initialize_tt_weights(
    TTCores& cores,
    const TTConfig& config,
    const bool use_xavier_init = true
);

// 核心计算函数
void tt_linear_compute(
    MemoryPackM_t const input[],
    MemoryPackM_t output[],
    const TTConfig& config,
    const TTCores& cores
);

// 维度计算辅助函数
void calculate_gemm_dims(
    const TTConfig& config,
    const unsigned core_idx,
    unsigned& m,
    unsigned& n,
    unsigned& k
);

// Reshape辅助函数
void reshape_for_next_multiply(
    Data_t* input,
    Data_t* output,
    const TTConfig& config,
    const unsigned core_idx
);

} // namespace tt

// Top level kernel interface
extern "C" {

void TTLinearKernel(
    MemoryPackM_t const input[],      
    MemoryPackM_t output[],           
    const unsigned num_dims,           // 总维度数
    const unsigned* input_modes,       // 输入维度数组
    const unsigned* output_modes,      // 输出维度数组
    const unsigned* ranks,             // TT秩数组
    const bool init = false           // 是否初始化权重
);

} // extern "C"
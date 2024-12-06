/// @author    Your Name
/// @copyright This software is copyrighted under the BSD 3-Clause License.

#pragma once

#include "TuckerTypes.h"
#include "MatrixMultiplication.h"
#include "hlslib/xilinx/Stream.h"

namespace tucker {

// Initialize Tucker matrices with random values
void initialize_tucker_weights(
    TuckerPackU1_t U1[],       // [R1 x I] packed matrix
    TuckerPackU2_t U2[],       // [O x R2] packed matrix 
    TuckerPackS_t S[],         // [R2 x R1] packed matrix
    const TuckerParams<Data_t>& params
);

// Convert between packed and unpacked data
void unpack_input_features(
    MemoryPackM_t const input[],
    Data_t unpacked_input[],
    const TuckerParams<Data_t>& params
);

void pack_output_features(
    Data_t const unpacked_output[],
    MemoryPackM_t output[],
    const TuckerParams<Data_t>& params
);

// Perform matrix multiplication steps using MatrixMultiplication
void compute_u1_multiply(
    TuckerPackU1_t const U1[],
    Data_t const input[],
    Data_t tmp1[],
    const TuckerParams<Data_t>& params
);

void compute_s_multiply(
    TuckerPackS_t const S[],
    Data_t const tmp1[],
    Data_t tmp2[],
    const TuckerParams<Data_t>& params  
);

void compute_u2_multiply(
    TuckerPackU2_t const U2[],
    Data_t const tmp2[],
    Data_t output[],
    const TuckerParams<Data_t>& params
);

// Core computation function
void tucker_linear_compute(
    MemoryPackM_t const input[],
    TuckerPackU1_t const U1[],
    TuckerPackU2_t const U2[],
    TuckerPackS_t const S[],
    MemoryPackM_t output[],
    const TuckerParams<Data_t>& params
);

} // namespace tucker
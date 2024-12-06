/// @author    Your Name
/// @copyright This software is copyrighted under the BSD 3-Clause License.

#pragma once

#include "TuckerTypes.h"
#include "MatrixMultiplication.h"
#include "hlslib/xilinx/Stream.h"
#include "hlslib/xilinx/DataPack.h"

// Forward declarations for internal classes/functions
namespace tucker {
void initialize_tucker_weights(const TuckerParams<Data_t>& params);

void tucker_linear_compute(
    MemoryPackM_t const input[],
    MemoryPackM_t output[],
    const TuckerParams<Data_t>& params);

} // namespace tucker


// Top level kernel interface for Tucker-based linear layer
extern "C" {

void TuckerLinearKernel(
    MemoryPackM_t const input[],      // [I × B] Input features
    MemoryPackM_t output[],           // [O × B] Output features
    const unsigned input_dim,          // I: Input dimension
    const unsigned output_dim,         // O: Output dimension
    const unsigned batch_size,         // B: Batch size
    const unsigned rank_1,             // R1: First Tucker rank
    const unsigned rank_2,             // R2: Second Tucker rank
    const bool init = false           // Initialize weights if true
);

} // extern "C"

// Public interface for host code
namespace tucker {


bool initialize_weights(const TuckerParams<Data_t>& params);


bool compute(
    MemoryPackM_t const input[],
    MemoryPackM_t output[],
    const TuckerParams<Data_t>& params
);


void get_weights(
    MemoryPackM_t U1[],
    MemoryPackM_t U2[], 
    MemoryPackM_t S[],
    const TuckerParams<Data_t>& params
);


void set_weights(
    MemoryPackM_t const U1[],
    MemoryPackM_t const U2[],
    MemoryPackM_t const S[],
    const TuckerParams<Data_t>& params
);

} // namespace tucker

// Implementation details
namespace tucker {

/// Maximum sizes for statically allocated arrays
constexpr unsigned kMaxInputDim = TUCKER_MAX_INPUT_DIM;
constexpr unsigned kMaxOutputDim = TUCKER_MAX_OUTPUT_DIM;
constexpr unsigned kMaxBatchSize = TUCKER_MAX_BATCH_SIZE;
constexpr unsigned kMaxRank1 = TUCKER_MAX_RANK_1;
constexpr unsigned kMaxRank2 = TUCKER_MAX_RANK_2;

/// Buffer sizes for packed memory formats
constexpr unsigned kU1BufferSize = 
    (kMaxInputDim * kMaxRank1 + kMemoryWidthK - 1) / kMemoryWidthK;
constexpr unsigned kU2BufferSize =
    (kMaxOutputDim * kMaxRank2 + kMemoryWidthM - 1) / kMemoryWidthM;
constexpr unsigned kSBufferSize =
    (kMaxRank1 * kMaxRank2 + kMemoryWidthM - 1) / kMemoryWidthM;
constexpr unsigned kTempBufferSize =
    (kMaxRank1 * kMaxBatchSize + kMemoryWidthM - 1) / kMemoryWidthM;

/// Helper function to compute packed buffer sizes
constexpr unsigned get_packed_size(unsigned elements, unsigned width) {
    return (elements + width - 1) / width;
}

} // namespace tucker
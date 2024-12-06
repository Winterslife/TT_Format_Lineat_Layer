/// @author    Johannes de Fine Licht (definelicht@inf.ethz.ch)
/// @copyright This software is copyrighted under the BSD 3-Clause License.

#include "hlslib/xilinx/Stream.h"
#include "hlslib/xilinx/Simulation.h"
#include "hlslib/xilinx/Utility.h"
#include "MatrixMultiplication.h"
#include "Compute.h"
#include <cassert>
#ifdef MM_HALF_PRECISION
#include "hls_half.h"
#endif
inline float to_float(const Data_t& val) {
    #ifdef MM_HALF_PRECISION
    return static_cast<float>(val);  // half class has implicit conversion to float
    #else
    return static_cast<float>(val);
    #endif
}

inline Data_t from_float(float val) {
    #ifdef MM_HALF_PRECISION
    return Data_t(val);  // Using half constructor
    #else
    return static_cast<Data_t>(val);
    #endif
}
inline Data_t safe_dot_product(Data_t const* a, Data_t const* b, unsigned size) {
    float acc = 0.0f;
    for(unsigned i = 0; i < size; ++i) {
        float a_val = to_float(a[i]);
        float b_val = to_float(b[i]);
        float prod = a_val * b_val;
        // Prevent accumulation from growing too large
        if(std::abs(prod) > 1e3f) {
            prod = (prod > 0) ? 1e3f : -1e3f;
        }
        acc += prod;
        // Clamp intermediate sums
        if(std::abs(acc) > 1e3f) {
            acc = (acc > 0) ? 1e3f : -1e3f;
        }
    }
    return from_float(std::max(-10.0f, std::min(10.0f, acc)));
}
// Then replace the existing MatrixMultiplicationKernel with this version
void MatrixMultiplicationKernel(
    MemoryPackK_t const input_a[],
    MemoryPackM_t const input_b[],
    MemoryPackM_t output[],
    const unsigned size_n,
    const unsigned size_k, 
    const unsigned size_m) {

    #pragma HLS INTERFACE m_axi port=input_a offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi port=input_b offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi port=output  offset=slave bundle=gmem2 
    #pragma HLS INTERFACE s_axilite port=size_n bundle=control
    #pragma HLS INTERFACE s_axilite port=size_k bundle=control
    #pragma HLS INTERFACE s_axilite port=size_m bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    static Data_t a_local[MAX_SIZE_N * MAX_SIZE_K];
    static Data_t b_local[MAX_SIZE_K * MAX_SIZE_M];
    static Data_t c_local[MAX_SIZE_N * MAX_SIZE_M];

    #pragma HLS ARRAY_PARTITION variable=a_local cyclic factor=8
    #pragma HLS ARRAY_PARTITION variable=b_local cyclic factor=8
    #pragma HLS ARRAY_PARTITION variable=c_local cyclic factor=8

    // Initialize to zero
    for(unsigned i = 0; i < MAX_SIZE_N * MAX_SIZE_M; ++i) {
        c_local[i] = 0;
    }

    // Load input matrix A with bounds checking
    LoadInputA: for(unsigned i = 0; i < (size_n * size_k + kMemoryWidthK - 1) / kMemoryWidthK; ++i) {
        #pragma HLS PIPELINE II=1
        if(i < size_n * size_k / kMemoryWidthK) {
            auto pack = input_a[i];
            for(unsigned j = 0; j < kMemoryWidthK; ++j) {
                unsigned idx = i * kMemoryWidthK + j;
                if(idx < size_n * size_k) {
                    float val = to_float(pack[j]);
                    a_local[idx] = from_float(std::max(-10.0f, std::min(10.0f, val)));
                }
            }
        }
    }

    // Load input matrix B with bounds checking
    LoadInputB: for(unsigned i = 0; i < (size_k * size_m + kMemoryWidthM - 1) / kMemoryWidthM; ++i) {
        #pragma HLS PIPELINE II=1
        if(i < size_k * size_m / kMemoryWidthM) {
            auto pack = input_b[i];
            for(unsigned j = 0; j < kMemoryWidthM; ++j) {
                unsigned idx = i * kMemoryWidthM + j;
                if(idx < size_k * size_m) {
                    float val = to_float(pack[j]);
                    b_local[idx] = from_float(std::max(-10.0f, std::min(10.0f, val)));
                }
            }
        }
    }

    // Matrix multiplication with safety checks
    ComputeLoop_N: for(unsigned n = 0; n < size_n; ++n) {
        ComputeLoop_M: for(unsigned m = 0; m < size_m; ++m) {
            #pragma HLS PIPELINE II=1
            Data_t row[MAX_SIZE_K], col[MAX_SIZE_K];
            
            // Gather row and column
            for(unsigned k = 0; k < size_k; ++k) {
                row[k] = a_local[n * size_k + k];
                col[k] = b_local[k * size_m + m];
            }
            
            // Use safe dot product
            c_local[n * size_m + m] = safe_dot_product(row, col, size_k);
        }
    }

    // Write results back to output with bounds checking
    WriteOutput: for(unsigned i = 0; i < (size_n * size_m + kMemoryWidthM - 1) / kMemoryWidthM; ++i) {
        #pragma HLS PIPELINE II=1
        if(i < size_n * size_m / kMemoryWidthM) {
            MemoryPackM_t pack;
            for(unsigned j = 0; j < kMemoryWidthM; ++j) {
                unsigned idx = i * kMemoryWidthM + j;
                if(idx < size_n * size_m) {
                    pack[j] = c_local[idx];
                } else {
                    pack[j] = 0;
                }
            }
            output[i] = pack;
        }
    }
}

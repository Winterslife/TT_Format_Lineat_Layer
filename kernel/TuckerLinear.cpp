#include "TuckerLinear.h"
#include "TuckerTypes.h"
#include "MatrixMultiplication.h"

// Static storage for Tucker matrices
namespace {
  TuckerPackU1_t U1_global[TUCKER_MAX_RANK_1 * TUCKER_MAX_INPUT_DIM / kTuckerMemoryWidthU1];
  TuckerPackU2_t U2_global[TUCKER_MAX_OUTPUT_DIM * TUCKER_MAX_RANK_2 / kTuckerMemoryWidthU2];
  TuckerPackS_t S_global[TUCKER_MAX_RANK_2 * TUCKER_MAX_RANK_1 / kTuckerMemoryWidthS];
} // anonymous namespace

namespace tucker {


void get_weights(
    MemoryPackM_t U1[],
    MemoryPackM_t U2[],
    MemoryPackM_t S[],
    const TuckerParams<Data_t>& params) {
    
    const auto& tparams = params;  // Rename for clarity

    // Copy U1
    for(unsigned i = 0; i < tparams.u1_size_memory(); ++i) {
        #pragma HLS PIPELINE
        U1[i] = static_cast<MemoryPackM_t>(::U1_global[i]);
    }

    // Copy U2
    for(unsigned i = 0; i < tparams.u2_size_memory(); ++i) {
        #pragma HLS PIPELINE
        U2[i] = static_cast<MemoryPackM_t>(::U2_global[i]);
    }

    // Copy S
    for(unsigned i = 0; i < tparams.s_size_memory(); ++i) {
        #pragma HLS PIPELINE
        S[i] = static_cast<MemoryPackM_t>(::S_global[i]);
    }
}

void tucker_linear_compute(
    MemoryPackM_t const input[],
    MemoryPackM_t output[],
    const TuckerParams<Data_t>& params) {

    #pragma HLS INLINE off
    #pragma HLS DATAFLOW

    // 打印维度信息
    std::cout << "\nTucker computation dimensions:\n";
    std::cout << "Input: " << params.input_dim << " x " << params.batch_size << "\n";
    std::cout << "U1: " << params.rank_1 << " x " << params.input_dim << "\n";
    std::cout << "S: " << params.rank_2 << " x " << params.rank_1 << "\n";
    std::cout << "U2: " << params.output_dim << " x " << params.rank_2 << "\n";

    // 添加一个辅助函数来检查数值范围
    auto check_range = [](const char* name, const MemoryPackM_t data[], size_t size) {
        float min_val = std::numeric_limits<float>::max();
        float max_val = std::numeric_limits<float>::lowest();
        float sum = 0.0f;
        
        for(size_t i = 0; i < size; ++i) {
            for(unsigned j = 0; j < kMemoryWidthM; ++j) {
                float val = static_cast<float>(data[i][j]);
                min_val = std::min(min_val, val);
                max_val = std::max(max_val, val);
                sum += val;
            }
        }
        float mean = sum / (size * kMemoryWidthM);
        
        std::cout << name << " stats - min: " << min_val 
                 << ", max: " << max_val 
                 << ", mean: " << mean << "\n";
    };

    // 打印输入样本
    std::cout << "\nInput samples: " << input[0][0] << ", " << input[0][1] << "\n";

    // Step 1: U1 × input
    static MemoryPackM_t tmp1[TUCKER_MAX_RANK_1 * TUCKER_MAX_BATCH_SIZE / kMemoryWidthM];
    MatrixMultiplicationKernel(
        reinterpret_cast<MemoryPackK_t const*>(U1_global),
        input,
        tmp1,
        params.rank_1, 
        params.input_dim, 
        params.batch_size
    );
    
    // 检查中间结果
    std::cout << "After U1 multiply: ";
    check_range("tmp1", tmp1, params.rank_1 * params.batch_size / kMemoryWidthM);

    // Step 2: S × tmp1
    static MemoryPackM_t tmp2[TUCKER_MAX_RANK_2 * TUCKER_MAX_BATCH_SIZE / kMemoryWidthM];
    MatrixMultiplicationKernel(
        reinterpret_cast<MemoryPackK_t const*>(S_global),
        tmp1,
        tmp2,
        params.rank_2, 
        params.rank_1, 
        params.batch_size
    );
    
    // 检查中间结果
    std::cout << "After S multiply: ";
    check_range("tmp2", tmp2, params.rank_2 * params.batch_size / kMemoryWidthM);

    // Step 3: U2 × tmp2
    MatrixMultiplicationKernel(
        reinterpret_cast<MemoryPackK_t const*>(U2_global),
        tmp2,
        output,
        params.output_dim, 
        params.rank_2, 
        params.batch_size
    );
    
    // 检查最终输出
    std::cout << "Final output: ";
    check_range("output", output, params.output_dim * params.batch_size / kMemoryWidthM);
}

// Processing Element wrapper for Tucker computation
void TuckerProcessingElement(
    Stream<ComputePackN_t> &aIn,
    Stream<ComputePackN_t> &aOut,
    Stream<ComputePackM_t> &bIn,
    Stream<ComputePackM_t> &bOut,
    Stream<ComputePackM_t> &cOut,
    Stream<ComputePackM_t> &cIn,
    const unsigned locationN,
    const TuckerParams<Data_t>& params) {

    #pragma HLS INLINE

    // Buffer for intermediate results
    ComputePackN_t aBuffer[2 * kInnerTilesN];
    ComputePackM_t cBuffer[kInnerTilesN * kInnerTilesM][kComputeTileSizeN];

    #pragma HLS ARRAY_PARTITION variable=cBuffer complete dim=2

InitializeABuffer_Inner:
    for (unsigned n2 = 0; n2 < kInnerTilesN; ++n2) {
        if (locationN < kComputeTilesN - 1) {
        InitializeABuffer_Outer:
            for (unsigned n1 = 0; n1 < kComputeTilesN - locationN; ++n1) {
                #pragma HLS PIPELINE II=1
                #pragma HLS LOOP_FLATTEN
                const auto read = aIn.Pop();
                if (n1 == 0) {
                    aBuffer[n2] = read;
                } else {
                    aOut.Push(read);
                }
            }
        } else {
            #pragma HLS PIPELINE II=1
            aBuffer[n2] = aIn.Pop();
        }
    }

OuterTile_N:
    for (unsigned n0 = 0; n0 < OuterTilesN(params.output_dim); ++n0) {
    OuterTile_M:
        for (unsigned m0 = 0; m0 < OuterTilesM(params.batch_size); ++m0) {
        
        Collapse_K:
            for (unsigned k = 0; k < params.rank_2; ++k) {
            Pipeline_N:
                for (unsigned n1 = 0; n1 < kInnerTilesN; ++n1) {
                Pipeline_M:
                    for (unsigned m1 = 0; m1 < kInnerTilesM; ++m1) {
                        #pragma HLS PIPELINE II=1
                        #pragma HLS LOOP_FLATTEN

                        // Double-buffering for A
                        if ((n0 < OuterTilesN(params.output_dim) - 1 || 
                             m0 < OuterTilesM(params.batch_size) - 1 ||
                             k < params.rank_2 - 1) &&
                            m1 >= locationN && 
                            m1 < kComputeTilesN) {
                            const auto read = aIn.Pop();
                            if (m1 == locationN) {
                                aBuffer[n1 + (k % 2 == 0 ? kInnerTilesN : 0)] = read;
                                #pragma HLS DEPENDENCE variable=aBuffer false
                            } else {
                                if (locationN < kComputeTilesN - 1) {
                                    aOut.Push(read);
                                }
                            }
                        }

                        // Compute tile
                        const auto aVal = aBuffer[n1 + (k % 2 == 0 ? 0 : kInnerTilesN)];
                        #pragma HLS DEPENDENCE variable=aBuffer false
                        const auto bVal = bIn.Pop();
                        if (locationN < kComputeTilesN - 1) {
                            bOut.Push(bVal);
                        }

                    Unroll_N:
                        for (unsigned n2 = 0; n2 < kComputeTileSizeN; ++n2) {
                            #pragma HLS UNROLL

                            const bool inBoundsN = ((n0 * kInnerTilesN * kComputeTileSizeN +
                                                   n1 * kComputeTileSizeN + n2) < params.output_dim);

                            ComputePackM_t cStore;
                            const auto cPrev = (k > 0)
                                            ? cBuffer[n1 * kInnerTilesM + m1][n2]
                                            : ComputePackM_t(static_cast<Data_t>(0));

                        Unroll_M:
                            for (unsigned m2 = 0; m2 < kComputeTileSizeM; ++m2) {
                                #pragma HLS UNROLL

                                const bool inBoundsM = ((m0 * kInnerTilesM * kComputeTileSizeM +
                                                       m1 * kComputeTileSizeM + m2) < params.batch_size);

                                const bool inBounds = inBoundsN && inBoundsM;

                                const auto mapped = OperatorMap::Apply(aVal[n2], bVal[m2]);
                                const auto prev = cPrev[m2];
                                const auto reduced = OperatorReduce::Apply(prev, mapped);
                                cStore[m2] = inBounds ? reduced : prev;
                                #pragma HLS DEPENDENCE variable=cBuffer false
                            }
                            cBuffer[n1 * kInnerTilesM + m1][n2] = cStore;
                        }
                    }
                }
            }

            // Write back results using bit width appropriate data types
            unsigned n1 = 0;
            unsigned n2 = 0;
            unsigned m1 = 0;
            unsigned inner = 0;

            const unsigned writeFlattenedInner =
                (kComputeTileSizeN * kInnerTilesM +
                 (kComputeTilesN - locationN - 1) * kComputeTileSizeN * kInnerTilesM);
            const unsigned writeFlattened = kInnerTilesN * writeFlattenedInner;

        WriteC_Flattened:
            for (unsigned i = 0; i < writeFlattened; ++i) {
                #pragma HLS PIPELINE II=1
                if (inner < kComputeTileSizeN * kInnerTilesM) {
                    cOut.Push(cBuffer[n1 * kInnerTilesM + m1][n2]);
                    if (m1 == kInnerTilesM - 1) {
                        m1 = 0;
                        if (n2 == kComputeTileSizeN - 1) {
                            n2 = 0;
                        } else {
                            ++n2;
                        }
                    } else {
                        ++m1;
                    }
                } else {
                    if (locationN < kComputeTilesN - 1) {
                        cOut.Push(cIn.Pop());
                    }
                }
                if (inner == writeFlattenedInner - 1) {
                    inner = 0;
                    ++n1;
                } else {
                    ++inner;
                }
            }
        }
    }
}


void initialize_tucker_weights(
    TuckerPackU1_t U1[], 
    TuckerPackU2_t U2[], 
    TuckerPackS_t S[],
    const TuckerParams<Data_t>& params) {
    
    // 使用更小的初始化范围
    constexpr float scale = 0.3f;  // 降低初始值范围
    
    // 计算每个矩阵的缩放因子
    const float u1_scale = scale / sqrt(params.input_dim);
    const float u2_scale = scale / sqrt(params.rank_2);
    const float s_scale = scale;

    // Initialize U1 [R1 x I]
    for(unsigned i = 0; i < params.u1_size_memory(); ++i) {
        #pragma HLS PIPELINE II=1
        TuckerPackU1_t pack;
        for(unsigned j = 0; j < kTuckerMemoryWidthU1; ++j) {
            if(i * kTuckerMemoryWidthU1 + j < params.u1_size()) {
                // 直接使用较小的初始化范围，不做归一化
                float rand_val = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * u1_scale;
                pack[j] = static_cast<Data_t>(rand_val);
            } else {
                pack[j] = 0;
            }
        }
        U1[i] = pack;
    }

    // Initialize U2 [O x R2]
    for(unsigned i = 0; i < params.u2_size_memory(); ++i) {
        #pragma HLS PIPELINE II=1
        TuckerPackU2_t pack;
        for(unsigned j = 0; j < kTuckerMemoryWidthU2; ++j) {
            if(i * kTuckerMemoryWidthU2 + j < params.u2_size()) {
                float rand_val = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * u2_scale;
                pack[j] = static_cast<Data_t>(rand_val);
            } else {
                pack[j] = 0;
            }
        }
        U2[i] = pack;
    }

    // Initialize S [R2 x R1]
    for(unsigned i = 0; i < params.s_size_memory(); ++i) {
        #pragma HLS PIPELINE II=1
        TuckerPackS_t pack;
        for(unsigned j = 0; j < kTuckerMemoryWidthS; ++j) {
            if(i * kTuckerMemoryWidthS + j < params.s_size()) {
                float rand_val = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * s_scale;
                pack[j] = static_cast<Data_t>(rand_val);
            } else {
                pack[j] = 0;
            }
        }
        S[i] = pack;
    }

    // 打印一些统计信息
    std::cout << "\nInitialized weights statistics:\n";
    std::cout << "Scale factors - U1: " << u1_scale 
              << ", U2: " << u2_scale 
              << ", S: " << s_scale << "\n";
    std::cout << "U1 samples: " << U1[0][0] << ", " << U1[0][1] << "\n";
    std::cout << "U2 samples: " << U2[0][0] << ", " << U2[0][1] << "\n";
    std::cout << "S samples: " << S[0][0] << ", " << S[0][1] << "\n";
}
} // namespace tucker

// Top level kernel implementation
void TuckerLinearKernel(
    MemoryPackM_t const input[],
    MemoryPackM_t output[],
    const unsigned input_dim,
    const unsigned output_dim,
    const unsigned batch_size,
    const unsigned rank_1,
    const unsigned rank_2,
    const bool init) {

    #pragma HLS INTERFACE m_axi port=input offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem2
    #pragma HLS INTERFACE s_axilite port=input_dim bundle=control
    #pragma HLS INTERFACE s_axilite port=output_dim bundle=control
    #pragma HLS INTERFACE s_axilite port=batch_size bundle=control
    #pragma HLS INTERFACE s_axilite port=rank_1 bundle=control
    #pragma HLS INTERFACE s_axilite port=rank_2 bundle=control
    #pragma HLS INTERFACE s_axilite port=init bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    const tucker::TuckerParams<Data_t> params(
        input_dim, output_dim, 1, rank_1, rank_2
    );

    if (init) {
        tucker::initialize_tucker_weights(U1_global, U2_global, S_global, params);
    }
    std::cout << "tucker_linear_compute\n" << std::flush;
    // 执行三个阶段的矩阵乘法
    tucker_linear_compute(input, output, params);
}
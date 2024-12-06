#include "TTLinear.h"
#include <random>
#include <cassert>

namespace tt {

// 辅助函数:计算剩余维度的乘积 (I_{k+1}
namespace {
    // Clamp values to prevent overflow
    template<typename T>
    T clamp_value(T val, T min_val, T max_val) {
        return std::max(min_val, std::min(val, max_val));
    }
    
    // Compute safe scale factor for initialization
    float compute_scale_factor(unsigned in_features, unsigned out_features) {
        return std::sqrt(2.0f / (in_features + out_features));
    }
    template<typename T>
    T safe_multiply(T a, T b, T min_val = -10.0f, T max_val = 10.0f) {
        float result = static_cast<float>(a) * static_cast<float>(b);
        return static_cast<T>(std::max(min_val, std::min(max_val, result)));
    }

    template<typename T>
    T safe_add(T a, T b, T min_val = -10.0f, T max_val = 10.0f) {
        float result = static_cast<float>(a) + static_cast<float>(b);
        return static_cast<T>(std::max(min_val, std::min(max_val, result)));
    }

    void print_debug_info(const char* stage, Data_t* data, unsigned size) {
        float max_val = -std::numeric_limits<float>::infinity();
        float min_val = std::numeric_limits<float>::infinity();
        float sum = 0.0f;
        bool has_nan = false;
        bool has_inf = false;
        
        for(unsigned i = 0; i < size; i++) {
            float val = static_cast<float>(data[i]);
            if(std::isnan(val)) {
                has_nan = true;
                continue;
            }
            if(std::isinf(val)) {
                has_inf = true;
                continue;
            }
            max_val = std::max(max_val, val);
            min_val = std::min(min_val, val);
            sum += val;
        }
        
        std::cout << "\n=== " << stage << " ===\n"
                  << "Range: [" << min_val << ", " << max_val << "]\n"
                  << "Mean: " << (sum/size) << "\n"
                  << "Has NaN: " << (has_nan ? "YES" : "no") << "\n"
                  << "Has Inf: " << (has_inf ? "YES" : "no") << "\n\n";
    }
}
// TT计算的主函数实现
void tt_linear_compute(
    MemoryPackM_t const input[],
    MemoryPackM_t output[],
    const TTConfig& config,
    const TTCores& cores) {
    
    #pragma HLS DATAFLOW
    
    static Data_t buffer1[TT_MAX_MODE_SIZE * TT_MAX_MODE_SIZE * TT_MAX_RANK];
    static Data_t buffer2[TT_MAX_MODE_SIZE * TT_MAX_MODE_SIZE * TT_MAX_RANK];
    
    #pragma HLS ARRAY_PARTITION variable=buffer1 cyclic factor=8
    #pragma HLS ARRAY_PARTITION variable=buffer2 cyclic factor=8

    std::memset(buffer1, 0, sizeof(buffer1));
    std::memset(buffer2, 0, sizeof(buffer2));

    Data_t *curr_buffer = buffer1;
    Data_t *next_buffer = buffer2;

    // First GEMM
    unsigned m0, n0, k0;
    calculate_gemm_dims(config, 0, m0, n0, k0);
    std::cout << "\nFirst GEMM dimensions: " << m0 << "x" << n0 << "x" << k0 << "\n";
    
    MatrixMultiplicationKernel(
        reinterpret_cast<MemoryPackK_t const*>(cores.get_core(0)),
        input,
        reinterpret_cast<MemoryPackM_t*>(curr_buffer),
        m0, n0, k0
    );
    
    print_debug_info("After First GEMM", curr_buffer, m0 * k0);

    // Clamp values after first GEMM
    for(unsigned i = 0; i < m0 * k0; i++) {
        float val = static_cast<float>(curr_buffer[i]);
        val = std::max(-10.0f, std::min(10.0f, val));
        curr_buffer[i] = static_cast<Data_t>(val);
    }

    // Middle GEMMs with additional error checking
    for(unsigned i = 1; i < config.num_dims - 1; i++) {
        unsigned mi, ni, ki;
        calculate_gemm_dims(config, i, mi, ni, ki);
        std::cout << "\nMiddle GEMM " << i << " dimensions: " << mi << "x" << ni << "x" << ki << "\n";
        
        reshape_for_next_multiply(curr_buffer, next_buffer, config, i-1);
        
        // Check next_buffer before GEMM
        print_debug_info("Before Middle GEMM", next_buffer, mi * ni);
        
        MatrixMultiplicationKernel(
            reinterpret_cast<MemoryPackK_t const*>(cores.get_core(i)),
            reinterpret_cast<MemoryPackM_t*>(next_buffer),
            reinterpret_cast<MemoryPackM_t*>(curr_buffer),
            mi, ni, ki
        );
        
        print_debug_info("After Middle GEMM", curr_buffer, mi * ki);

        // Safe clamping after middle GEMM
        for(unsigned j = 0; j < mi * ki; j++) {
            float val = static_cast<float>(curr_buffer[j]);
            if(std::isnan(val) || std::isinf(val)) {
                curr_buffer[j] = 0;
            } else {
                val = std::max(-10.0f, std::min(10.0f, val));
                curr_buffer[j] = static_cast<Data_t>(val);
            }
        }

        Data_t* temp = curr_buffer;
        curr_buffer = next_buffer;
        next_buffer = temp;
    }
    // Final GEMM
    unsigned mf, nf, kf;
    calculate_gemm_dims(config, config.num_dims - 1, mf, nf, kf);
    reshape_for_next_multiply(curr_buffer, next_buffer, config, config.num_dims - 2);

    MatrixMultiplicationKernel(
        reinterpret_cast<MemoryPackK_t const*>(cores.get_core(config.num_dims - 1)),
        reinterpret_cast<MemoryPackM_t*>(next_buffer),
        output,
        mf, nf, kf
    );
}

// Top level kernel实现
extern "C" {

void TTLinearKernel(
    MemoryPackM_t const input[],
    MemoryPackM_t output[],
    const unsigned num_dims,
    const unsigned* input_modes,
    const unsigned* output_modes,
    const unsigned* ranks,
    const bool init) {

    #pragma HLS INTERFACE m_axi port=input offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi port=input_modes offset=slave bundle=gmem2
    #pragma HLS INTERFACE m_axi port=output_modes offset=slave bundle=gmem2
    #pragma HLS INTERFACE m_axi port=ranks offset=slave bundle=gmem2
    #pragma HLS INTERFACE s_axilite port=num_dims bundle=control
    #pragma HLS INTERFACE s_axilite port=init bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    // 构造配置
    std::vector<unsigned> in_modes(input_modes, input_modes + num_dims);
    std::vector<unsigned> out_modes(output_modes, output_modes + num_dims);
    std::vector<unsigned> tt_ranks(ranks, ranks + num_dims + 1);

    TTConfig config(num_dims, tt_ranks, in_modes, out_modes);
    
    // 创建并初始化TT cores
    static TTCores cores(config);
    
    if(init) {
        initialize_tt_weights(cores, config, true);
    }

    // 执行TT计算
    tt_linear_compute(input, output, config, cores);
}

} // extern "C" * ... * I_d)
unsigned compute_remaining_input_modes(
    const std::vector<unsigned>& modes,
    const unsigned start_idx) {
    
    unsigned result = 1;
    for(unsigned i = start_idx; i < modes.size(); i++) {
        result *= modes[i];
    }
    return result;
}

// 辅助函数:计算前缀输出维度乘积 (O_1 * ... * O_k)
unsigned compute_output_prefix(
    const std::vector<unsigned>& modes,
    const unsigned end_idx) {
    
    unsigned result = 1;
    for(unsigned i = 0; i < end_idx; i++) {
        result *= modes[i];
    }
    return result;
}

// 计算中间GEMM的混合维度
unsigned compute_mixed_dims(
    const TTConfig& config,
    const unsigned core_idx) {
    
    // 计算O_1 * ... * O_{i-1}部分
    unsigned prefix = compute_output_prefix(config.output_modes, core_idx);
    
    // 计算I_{i+1} * ... * I_d部分
    unsigned suffix = compute_remaining_input_modes(
        config.input_modes, core_idx + 1);
    
    return prefix * suffix;
}

void calculate_gemm_dims(
    const TTConfig& config,
    const unsigned core_idx,
    unsigned& m,
    unsigned& n,
    unsigned& k) {

    if (core_idx == 0) {
        // First GEMM: [I2I3...Id, I1] × [I1, r1O1]
        m = compute_remaining_input_modes(config.input_modes, 1);
        n = config.input_modes[0];
        k = config.ranks[1] * config.output_modes[0];
    }
    else if (core_idx == config.num_dims - 1) {
        // Final GEMM: [O1...Od-1, rd-1Id] × [rd-1Id, Od]
        m = compute_output_prefix(config.output_modes, config.num_dims - 1);
        n = config.ranks[core_idx] * config.input_modes[core_idx];
        k = config.output_modes[core_idx];
    }
    else {
        // Middle GEMMs: [O1...Oi-1Ii+1...Id, riIi] × [riIi, ri+1Oi]
        m = compute_mixed_dims(config, core_idx);
        n = config.ranks[core_idx] * config.input_modes[core_idx];
        k = config.ranks[core_idx + 1] * config.output_modes[core_idx];
    }

    #ifndef HLSLIB_SYNTHESIS
    // 验证维度计算的正确性
    assert(m > 0 && n > 0 && k > 0);
    std::cout << "GEMM " << core_idx << " dimensions: "
              << m << " x " << n << " x " << k << std::endl;
    #endif
}

// Reshape函数实现
void reshape_for_next_multiply(
    Data_t* input,
    Data_t* output,
    const TTConfig& config,
    const unsigned core_idx) {
    
    #pragma HLS INLINE
    unsigned rows, cols;
    
    if (core_idx == 0) {
        rows = compute_remaining_input_modes(config.input_modes, 2);
        cols = config.ranks[1] * config.output_modes[0];
    }
    else if (core_idx < config.num_dims - 1) {
        rows = compute_mixed_dims(config, core_idx + 1);
        cols = config.ranks[core_idx + 1] * config.output_modes[core_idx];
    }
    else {
        return; // No reshape needed for last core
    }
    
    print_debug_info("Before Reshape", input, rows * cols);
    
    // Safe copy with bounds checking
    ReshapeLoop:
    for(unsigned i = 0; i < rows * cols; i++) {
        #pragma HLS PIPELINE II=1
        float val = static_cast<float>(input[i]);
        // Detect and handle invalid values
        if(std::isnan(val) || std::isinf(val)) {
            output[i] = 0;
            continue;
        }
        // Clamp values to prevent overflow
        val = std::max(-10.0f, std::min(10.0f, val));
        output[i] = static_cast<Data_t>(val);
    }
    
    print_debug_info("After Reshape", output, rows * cols);
}

// Xavier初始化实现
void xavier_init(Data_t* data, unsigned fan_in, unsigned fan_out, unsigned size) {
    // 计算标准差
    float std = std::sqrt(2.0f / (fan_in + fan_out));
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0, std);

    for(unsigned i = 0; i < size; i++) {
        data[i] = static_cast<Data_t>(dist(gen) * std);
    }
}

// 初始化TT核权重
void initialize_tt_weights(TTCores& cores, const TTConfig& config, const bool use_xavier_init) {
    std::cout << "Initializing TT weights with improved scaling...\n";
    
    // Calculate overall scale factor based on TT-ranks
    float total_scale = 0.0f;
    for(unsigned i = 0; i < config.num_dims; i++) {
        total_scale += static_cast<float>(config.ranks[i] * config.ranks[i+1]);
    }
    float global_scale = 1.0f / std::sqrt(total_scale);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    
    for(unsigned i = 0; i < config.num_dims; i++) {
        unsigned core_size = config.core_size(i);
        Data_t* core_data = reinterpret_cast<Data_t*>(cores.get_core(i));
        
        // Calculate fan_in and fan_out for this core
        unsigned fan_in = config.ranks[i] * config.input_modes[i];
        unsigned fan_out = config.ranks[i+1] * config.output_modes[i];
        
        // Xavier initialization with improved scaling
        float std_dev = compute_scale_factor(fan_in, fan_out) * global_scale;
        std::normal_distribution<float> dist(0.0f, std_dev);
        
        // Initialize with bounds checking
        for(unsigned j = 0; j < core_size; j++) {
            float val = dist(gen);
            // Clip values to prevent extreme numbers
            val = clamp_value(val, -1.0f, 1.0f);
            core_data[j] = static_cast<Data_t>(val);
        }
        
        // Print statistics for this core
        float core_norm = 0.0f;
        for(unsigned j = 0; j < core_size; j++) {
            core_norm += static_cast<float>(core_data[j] * core_data[j]);
        }
        core_norm = std::sqrt(core_norm);
        std::cout << "Core " << i << " norm: " << core_norm << "\n";
    }
}

}
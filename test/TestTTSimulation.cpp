#include "TTLinear.h"
#include "TTTypes.h"
#include "Utility.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <type_traits>
#include <vector>

int main(int argc, char **argv) {
    // 解析命令行参数
    const unsigned num_dims = std::stoul(argv[1]);    // TT cores数量
    if (argc < 3 * num_dims + 3) {  // +3 for program name, num_dims, and ranks (need num_dims+1 ranks)
    std::cerr << "Usage: ./TestTTSimulation num_dims r0 r1...r{n+1} in_mode_1...in_mode_n out_mode_1...out_mode_n\n"
              << "  num_dims: Number of TT cores\n"
              << "  r0...r{n+1}: TT ranks (need num_dims+1 ranks, r0 and r{n+1} must be 1)\n"
              << "  in_mode_1...n: Size of each input mode\n"
              << "  out_mode_1...n: Size of each output mode\n";
    return 1;
}

// Parse ranks (need num_dims+1 ranks)
std::vector<unsigned> ranks(num_dims + 1);
for(unsigned i = 0; i <= num_dims; i++) {
    ranks[i] = std::stoul(argv[i + 2]);
    
    // Check rank constraints
    if(ranks[i] > TT_MAX_RANK) {
        std::cerr << "Rank " << ranks[i] << " at position " << i 
                  << " exceeds maximum allowed value (" << TT_MAX_RANK << ")" << std::endl;
        return 1;
    }
}

// Verify first and last ranks are 1
if(ranks[0] != 1 || ranks[num_dims] != 1) {
    std::cerr << "First and last ranks must be 1, got " 
              << ranks[0] << " and " << ranks[num_dims] << std::endl;
    return 1;
}

    // Parse input modes
std::vector<unsigned> input_modes(num_dims);
for(unsigned i = 0; i < num_dims; i++) {
    input_modes[i] = std::stoul(argv[i + 3]);
    if(input_modes[i] > TT_MAX_MODE_SIZE) {
        std::cerr << "Input mode size " << input_modes[i] << " at dimension " << i 
                  << " exceeds maximum allowed value (" << TT_MAX_MODE_SIZE << ")" << std::endl;
        return 1;
    }
}

std::vector<unsigned> output_modes(num_dims);
for(unsigned i = 0; i < num_dims; i++) {
    output_modes[i] = std::stoul(argv[i + num_dims + 3]);
    if(output_modes[i] > TT_MAX_MODE_SIZE) {
        std::cerr << "Output mode size " << output_modes[i] << " at dimension " << i 
                  << " exceeds maximum allowed value (" << TT_MAX_MODE_SIZE << ")" << std::endl;
        return 1;
    }
}
// Calculate total sizes
unsigned total_input_size = 1;
unsigned total_output_size = 1;
for (unsigned i = 0; i < num_dims; ++i) {
    total_input_size *= input_modes[i];
    total_output_size *= output_modes[i];
}
    // const unsigned num_dims = std::stoul(argv[1]);    // TT cores数量
    const unsigned rank = std::stoul(argv[2]);        // 统一的TT秩
    const unsigned mode_size_in = std::stoul(argv[3]); // 统一的in模式大小
    const unsigned mode_size_out = std::stoul(argv[4]);   // 统一的out模式大小
    constexpr unsigned batch_size = 1;                // 固定batch_size为1

    // 参数验证
    if (num_dims > TT_MAX_DIMS) {
        std::cerr << "Number of dimensions exceeds maximum allowed value ("
                  << TT_MAX_DIMS << ")" << std::endl;
        return 1;
    }
    if (rank > TT_MAX_RANK) {
        std::cerr << "Rank exceeds maximum allowed value ("
                  << TT_MAX_RANK << ")" << std::endl;
        return 1;
    }
    if (mode_size_in > TT_MAX_MODE_SIZE) {
        std::cerr << "Mode size exceeds maximum allowed value ("
                  << TT_MAX_MODE_SIZE << ")" << std::endl;
        return 1;
    }

    // 构建TT配置
    std::vector<unsigned> ranks(num_dims + 1, rank);
    ranks[0] = ranks[num_dims] = 1;  // 首尾秩设为1
    // std::vector<unsigned> input_modes(num_dims, mode_size_in);
    // std::vector<unsigned> output_modes(num_dims, mode_size_out);

    // 计算总的输入/输出维度
    // unsigned total_input_size = 1;
    // unsigned total_output_size = 1;
    for (unsigned i = 0; i < num_dims; ++i) {
        total_input_size *= input_modes[i];
        total_output_size *= output_modes[i];
    }

    // 初始化输入数据
    std::vector<Data_t> input(total_input_size);
    std::vector<Data_t> output(total_output_size, 0);
    std::vector<Data_t> reference_output(total_output_size, 0);

    // 使用随机数初始化输入
    std::default_random_engine rng(kSeed);
    std::uniform_real_distribution<double> dist(1.0, 10.0);
    std::for_each(input.begin(), input.end(),
                  [&dist, &rng](Data_t &val) { val = static_cast<Data_t>(dist(rng)); });

    // Pack数据用于kernel
    const auto input_kernel = Pack<kMemoryWidthM>(input);
    auto output_kernel = Pack<kMemoryWidthM>(output);

    std::cout << "Running TT Linear Layer test...\n" << std::flush;
    auto start = std::chrono::high_resolution_clock::now();

    // 执行TT计算
    TTLinearKernel(
        input_kernel.data(),
        output_kernel.data(),
        num_dims,
        input_modes.data(),
        output_modes.data(),
        ranks.data(),
        true  // 初始化权重
    );

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Unpack结果
    const auto result = Unpack<kMemoryWidthM>(output_kernel);

    // 打印性能统计
    std::cout << "\nPerformance Statistics:\n";
    std::cout << "Time taken: " << duration.count() << " microseconds\n";
    std::cout << "Input size: " << total_input_size << "\n";
    std::cout << "Output size: " << total_output_size << "\n";
    
    // 计算理论FLOPs
    unsigned long long total_flops = 0;
    for (unsigned i = 0; i < num_dims; ++i) {
        total_flops += 2ULL * ranks[i] * input_modes[i] * ranks[i+1] * output_modes[i];
    }
    
    double gflops = (static_cast<double>(total_flops) / duration.count()) * 1e-3;
    std::cout << "Theoretical GFLOPs: " << gflops << "\n";

    // 数值稳定性检查
    double max_val = -std::numeric_limits<double>::infinity();
    double min_val = std::numeric_limits<double>::infinity();
    double sum = 0.0;
    
    for (unsigned i = 0; i < total_output_size; ++i) {
        double val = static_cast<double>(result[i]);
        max_val = std::max(max_val, val);
        min_val = std::min(min_val, val);
        sum += val;
    }
    
    std::cout << "\nNumerical Stability Check:\n";
    std::cout << "Min value: " << min_val << "\n";
    std::cout << "Max value: " << max_val << "\n";
    std::cout << "Mean value: " << (sum / total_output_size) << "\n";

    // 检查无穷大或NaN值
    bool has_invalid = false;
    for (unsigned i = 0; i < total_output_size; ++i) {
        if (std::isinf(result[i]) || std::isnan(result[i])) {
            std::cout << "Invalid value detected at index " << i << ": " 
                      << result[i] << "\n";
            has_invalid = true;
        }
    }

    if (!has_invalid) {
        std::cout << "\nTest completed successfully!\n";
        return 0;
    } else {
        std::cout << "\nTest failed due to numerical instability!\n";
        return 1;
    }
}
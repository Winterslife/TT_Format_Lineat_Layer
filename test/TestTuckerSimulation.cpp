/// @author    Your Name
/// @copyright This software is copyrighted under the BSD 3-Clause License.

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <type_traits>
#include <vector>
#include "TuckerLinear.h"
#include "Utility.h"

// Simple direct matrix multiplication for reference
int main(int argc, char **argv) {
#ifdef MM_DYNAMIC_SIZES
  if (argc < 5 || argc > 5) {
    std::cerr << "Usage: ./TestTuckerSimulation I O R1 R2" << std::endl;
    std::cerr << "  I: Input dimension" << std::endl;
    std::cerr << "  O: Output dimension" << std::endl;
    std::cerr << "  R1: First Tucker rank" << std::endl;
    std::cerr << "  R2: Second Tucker rank" << std::endl;
    return 1;
  }
  const unsigned input_dim = std::stoul(argv[1]);   // 输入维度 I
  const unsigned output_dim = std::stoul(argv[2]);  // 输出维度 O
  const unsigned rank_1 = std::stoul(argv[3]);      // Tucker秩 R1
  const unsigned rank_2 = std::stoul(argv[4]);      // Tucker秩 R2
  constexpr unsigned batch_size = 1;                // 固定batch_size为1
#else
  constexpr auto input_dim = kInputDim;    // 需要在配置中定义
  constexpr auto output_dim = kOutputDim;  // 需要在配置中定义
  constexpr auto rank_1 = kRank1;          // Tucker秩 R1
  constexpr auto rank_2 = kRank2;          // Tucker秩 R2
  constexpr auto batch_size = 1;           // 固定batch_size为1
#endif

   bool has_mismatch = false;
  // Validation checks
  if (input_dim > TUCKER_MAX_INPUT_DIM) {
    std::cerr << "Input dimension exceeds maximum allowed value (" 
              << TUCKER_MAX_INPUT_DIM << ")" << std::endl;
    return 1;
  }
  if (output_dim > TUCKER_MAX_OUTPUT_DIM) {
    std::cerr << "Output dimension exceeds maximum allowed value (" 
              << TUCKER_MAX_OUTPUT_DIM << ")" << std::endl;
    return 1;
  }

  // 验证内存对齐
  if (rank_1 % kMemoryWidthK != 0) {
    std::cerr << "R1 must be divisible by memory width." << std::endl;
    return 1;
  }
  if (rank_2 % kMemoryWidthK != 0) {
    std::cerr << "R2 must be divisible by memory width." << std::endl;
    return 1;
  }
  // 1. 首先定义随机数生成器和分布
    std::default_random_engine rng(kSeed);
    std::normal_distribution<float> dist(0.0f, 0.3f);
    std::uniform_real_distribution<double> input_dist(1.0, 10.0);

    // 2. 声明并分配内存
    std::vector<Data_t> input(input_dim * batch_size);  // [I x 1] 输入
    std::vector<Data_t> tucker_output(output_dim * batch_size, 0);  // Tucker输出
    std::vector<Data_t> direct_output(output_dim * batch_size, 0);  // 直接矩阵乘法输出
    std::vector<Data_t> W(output_dim * input_dim);  // [O x I] 直接乘法的权重矩阵
    

    // 3. 初始化输入数据（使用均匀分布）
    for(auto& val : input) {
    val = static_cast<Data_t>(input_dist(rng));
  }
    for(auto& val : W) {
    val = static_cast<Data_t>(dist(rng));
  }

    // 5. Pack数据用于kernel
    const auto inputKernel = Pack<kMemoryWidthM>(input);
    auto tuckerOutputKernel = Pack<kMemoryWidthM>(tucker_output);
    auto directOutputKernel = Pack<kMemoryWidthM>(direct_output);
    const auto WKernel = Pack<kMemoryWidthK>(W);
       
    std::cout << "Running tests...\n" << std::flush;
    auto start_tucker = std::chrono::high_resolution_clock::now();

    // 7. 执行计算
    TuckerLinearKernel(
      inputKernel.data(),
      tuckerOutputKernel.data(),
      input_dim,   // 输入维度I
      output_dim,  // 输出维度O
      batch_size,  // 固定为1
      rank_1,      // R1
      rank_2,      // R2
      true        // 初始化权重
  );
    auto end_tucker = std::chrono::high_resolution_clock::now();
    // 8. 运行参考实现
    auto start_direct = std::chrono::high_resolution_clock::now();

  MatrixMultiplicationKernel(
      WKernel.data(),
      inputKernel.data(),
      directOutputKernel.data(), 
      output_dim,  // O
      input_dim,   // I 
      batch_size   // 1
  );

  auto end_direct = std::chrono::high_resolution_clock::now();

  // 计算运行时间
  auto time_tucker = std::chrono::duration_cast<std::chrono::microseconds>
                    (end_tucker - start_tucker).count();
  auto time_direct = std::chrono::duration_cast<std::chrono::microseconds>
                    (end_direct - start_direct).count();

    // 结果验证
  const auto tucker_result = Unpack<kMemoryWidthM>(tuckerOutputKernel);
  const auto direct_result = Unpack<kMemoryWidthM>(directOutputKernel);
  
  // 比较结果
  double max_diff = 0.0;
  double total_diff = 0.0;
  int mismatch_count = 0;

  for (unsigned i = 0; i < output_dim; ++i) {
    const auto val_tucker = make_signed<Data_t>(tucker_result[i]);
    const auto val_direct = make_signed<Data_t>(direct_result[i]);
    const double diff = std::abs(val_tucker - val_direct);
    max_diff = std::max(max_diff, diff);
    total_diff += diff;
    
    // 相对误差检查
    const double rel_error = diff / (std::abs(val_direct) + 1e-6);
    if (rel_error > 1e-3) { // 0.1%相对误差阈值
      mismatch_count++;
      if (mismatch_count <= 5) {  // 只打印前5个不匹配
        std::cout << "Mismatch at " << i << ": "
                 << "Tucker = " << val_tucker 
                 << ", Direct = " << val_direct
                 << ", Relative error = " << (rel_error * 100) << "%\n";
      }
    }
  }

  // 输出性能比较结果
  std::cout << "\n=== Performance Comparison ===\n";
  std::cout << "Tucker Implementation:\n"
            << "  Time: " << time_tucker << " us\n"
            << "  Ops: " << (2 * input_dim * output_dim * batch_size) << "\n"
            << "  Performance: " << (2 * input_dim * output_dim * batch_size / time_tucker) 
            << " MOps/s\n";
            
  std::cout << "\nDirect Matrix Multiplication:\n"
            << "  Time: " << time_direct << " us\n"
            << "  Ops: " << (2 * input_dim * output_dim * batch_size) << "\n"
            << "  Performance: " << (2 * input_dim * output_dim * batch_size / time_direct) 
            << " MOps/s\n";

  std::cout << "\n=== Accuracy Analysis ===\n"
            << "Max Absolute Difference: " << max_diff << "\n"
            << "Average Absolute Difference: " << (total_diff / output_dim) << "\n"
            << "Mismatches (>0.1% relative error): " << mismatch_count 
            << " of " << output_dim << " (" 
            << (100.0 * mismatch_count / output_dim) << "%)\n";

  return 0;
}

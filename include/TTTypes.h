/// @author    Your Name
/// @copyright This software is copyrighted under the BSD 3-Clause License.

#pragma once
#include "TTConfig.h"  // Must include TTConfig.h first
#include "MatrixMultiplication.h"  // For MemoryPackM_t

// #include "Config.h"
#include <vector>
#include <cassert>
#include <cmath>

namespace tt {

// TT配置结构
struct TTConfig {
    unsigned num_dims;                 // 总维度数(cores数量)
    std::vector<unsigned> ranks;       // TT秩 [r0=1, r1, ..., rd-1, rd=1]
    std::vector<unsigned> input_modes; // 输入维度分解 [I1,I2,...,Id]
    std::vector<unsigned> output_modes;// 输出维度分解 [O1,O2,...,Od]
    
    TTConfig(unsigned d,
            const std::vector<unsigned>& r,
            const std::vector<unsigned>& in_modes,
            const std::vector<unsigned>& out_modes)
        : num_dims(d), ranks(r), input_modes(in_modes), output_modes(out_modes) {
        
        validate();
    }

    void validate() const {
        #ifndef HLSLIB_SYNTHESIS
        // 基本维度检查
        assert(num_dims >= 2 && "Need at least 2 dimensions"); 
        assert(ranks.size() == num_dims + 1 && "Invalid ranks size");
        assert(input_modes.size() == num_dims && "Invalid input modes size");
        assert(output_modes.size() == num_dims && "Invalid output modes size");
        
        // 边界秩必须为1
        assert(ranks[0] == 1 && "First rank must be 1");
        assert(ranks[num_dims] == 1 && "Last rank must be 1");

        // 检查维度限制
        for(unsigned i = 0; i < num_dims; i++) {
            assert(input_modes[i] <= TT_MAX_MODE_SIZE && 
                   "Input mode exceeds maximum allowed size");
            assert(output_modes[i] <= TT_MAX_MODE_SIZE && 
                   "Output mode exceeds maximum allowed size");
            assert(ranks[i] <= TT_MAX_RANK && 
                   "Rank exceeds maximum allowed size");
        }
        #endif
    }

    // 辅助函数：计算输入维度总大小
    unsigned total_input_size() const {
        unsigned size = 1;
        for(auto mode : input_modes) {
            size *= mode;
        }
        return size;
    }

    // 计算输出维度总大小
    unsigned total_output_size() const {
        unsigned size = 1;
        for(auto mode : output_modes) {
            size *= mode;
        }
        return size;
    }

    // 计算特定core的大小(用于内存分配)
    unsigned core_size(unsigned idx) const {
        assert(idx < num_dims);
        return ranks[idx] * input_modes[idx] * ranks[idx+1] * output_modes[idx];
    }

    // 计算特定core的packed大小
    unsigned core_size_memory(unsigned idx) const {
        const unsigned size = core_size(idx);
        return (size + kMemoryWidthM - 1) / kMemoryWidthM;
    }

    // 计算所有cores的总参数数量
    unsigned total_parameters() const {
        unsigned total = 0;
        for(unsigned i = 0; i < num_dims; i++) {
            total += ranks[i] * input_modes[i] * ranks[i+1] * output_modes[i];
        }
        return total;
    }

    // 打印配置信息(用于调试)
    void print() const {
        std::cout << "\nTT Layer Configuration:\n";
        std::cout << "Number of dimensions: " << num_dims << "\n";
        std::cout << "Ranks: ";
        for(auto r : ranks) std::cout << r << " ";
        std::cout << "\nInput modes: ";
        for(auto m : input_modes) std::cout << m << " ";
        std::cout << "\nOutput modes: ";
        for(auto m : output_modes) std::cout << m << " ";
        std::cout << "\nTotal parameters: " << total_parameters() << "\n";
        std::cout << std::endl;
    }
};

// TT cores的内存管理
struct TTCores {
    std::vector<MemoryPackM_t*> cores;  // 存储每个TT core

    explicit TTCores(const TTConfig& config) {
        allocate(config);
    }

    ~TTCores() {
        for(auto& core : cores) {
            if(core) delete[] core;
        }
    }

    // 分配内存
    void allocate(const TTConfig& config) {
        cores.resize(config.num_dims);
        for(unsigned i = 0; i < config.num_dims; i++) {
            cores[i] = new MemoryPackM_t[config.core_size_memory(i)];
        }
    }

    // 获取特定core
    MemoryPackM_t* get_core(unsigned idx) {
        assert(idx < cores.size());
        return cores[idx];
    }

    const MemoryPackM_t* get_core(unsigned idx) const {
        assert(idx < cores.size());
        return cores[idx];
    }
};

} // namespace tt
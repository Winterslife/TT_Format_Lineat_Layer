/// @author    Your Name
/// @copyright This software is copyrighted under the BSD 3-Clause License.

#pragma once

#include "TuckerConfig.h"

namespace tucker {

// Tucker parameter struct with validation
template <typename T>
struct TuckerParams {
  const unsigned input_dim;  // Input dimension I
  const unsigned output_dim; // Output dimension O 
  const unsigned batch_size; // Batch size B
  const unsigned rank_1;     // Tucker rank R1
  const unsigned rank_2;     // Tucker rank R2

  TuckerParams(unsigned i, unsigned o, unsigned b, unsigned r1, unsigned r2)
    : input_dim(i), output_dim(o), batch_size(b), rank_1(r1), rank_2(r2) {
    // Validate dimensions
    #ifndef HLSLIB_SYNTHESIS
    if (input_dim > TUCKER_MAX_INPUT_DIM) {
      throw std::runtime_error("Input dimension exceeds maximum");
    }
    if (output_dim > TUCKER_MAX_OUTPUT_DIM) {
      throw std::runtime_error("Output dimension exceeds maximum");
    }
    if (batch_size != 1) {
            throw std::runtime_error("Batch size must be 1");
        }
    if (rank_1 > TUCKER_MAX_RANK_1) {
      throw std::runtime_error("Rank R1 exceeds maximum"); 
    }
    if (rank_2 > TUCKER_MAX_RANK_2) {
      throw std::runtime_error("Rank R2 exceeds maximum");
    }
    #endif
  }

  // Helper functions to compute sizes
  unsigned u1_size() const { return rank_1 * input_dim; }
  unsigned u2_size() const { return output_dim * rank_2; }
  unsigned s_size() const { return rank_2 * rank_1; }
  
  // Memory width related helpers
  unsigned u1_size_memory() const { 
    return (u1_size() + kTuckerMemoryWidthU1 - 1) / kTuckerMemoryWidthU1; 
  }
  
  unsigned u2_size_memory() const {
    return (u2_size() + kTuckerMemoryWidthU2 - 1) / kTuckerMemoryWidthU2;
  }
  
  unsigned s_size_memory() const {
    return (s_size() + kTuckerMemoryWidthS - 1) / kTuckerMemoryWidthS;
  }

  void print_dimensions() const {
        std::cout << "Tucker dimensions:\n"
                  << "  Input dim (I): " << input_dim << "\n"
                  << "  Output dim (O): " << output_dim << "\n"
                  << "  Batch size (B): " << batch_size << "\n"
                  << "  Rank1 (R1): " << rank_1 << "\n"
                  << "  Rank2 (R2): " << rank_2 << "\n";
    }

    size_t get_total_params() const {
        return (input_dim * rank_1 +  // U1
                rank_1 * rank_2 +     // S
                output_dim * rank_2); // U2
    }
};

// Memory buffer struct for Tucker matrices
template <typename T>
struct TuckerBuffers {
  T* U1;        // [R1 x I] matrix
  T* U2;        // [O x R2] matrix  
  T* S;         // [R2 x R1] matrix
  T* tmp1;      // [R1 x B] temporary buffer
  T* tmp2;      // [R2 x B] temporary buffer
  
  TuckerBuffers(
    T* u1, T* u2, T* s, 
    T* t1, T* t2)
    : U1(u1), U2(u2), S(s),
      tmp1(t1), tmp2(t2) {}
};

} // namespace tucker
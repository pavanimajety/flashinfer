/*
 * Copyright (c) 2024 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <flashinfer/pos_enc.cuh>

#include "pytorch_extension_utils.h"

using namespace flashinfer;

void apply_rope(at::Tensor q, at::Tensor k, at::Tensor q_rope, at::Tensor k_rope, at::Tensor indptr,
                at::Tensor offsets, unsigned int rotary_dim, bool interleave, float rope_scale,
                float rope_theta, int64_t cuda_stream) {
  CHECK_LAST_DIM_CONTIGUOUS(q);
  CHECK_LAST_DIM_CONTIGUOUS(k);
  CHECK_INPUT(indptr);
  CHECK_INPUT(offsets);

  auto device = q.device();
  CHECK_EQ(k.device(), device);
  CHECK_DIM(3, q);        // q: (nnz, H_Q, D)
  CHECK_DIM(3, k);        // k: (nnz, H_K, D)
  CHECK_DIM(1, indptr);   // indptr: (B + 1)
  CHECK_DIM(1, offsets);  // offsets: (B)
  CHECK_EQ(q.size(0), k.size(0));
  CHECK_EQ(q.size(2), k.size(2));
  unsigned int num_qo_heads = q.size(1);
  unsigned int num_kv_heads = k.size(1);
  unsigned int head_dim = q.size(2);
  unsigned int batch_size = offsets.size(0);
  CHECK_EQ(indptr.size(0), batch_size + 1);
  size_t q_stride_n = q.stride(0);
  size_t q_stride_h = q.stride(1);
  size_t k_stride_n = k.stride(0);
  size_t k_stride_h = k.stride(1);
  size_t q_rope_stride_n = q_rope.stride(0);
  size_t q_rope_stride_h = q_rope.stride(1);
  size_t k_rope_stride_n = k_rope.stride(0);
  size_t k_rope_stride_h = k_rope.stride(1);

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(q.scalar_type(), c_type, [&] {
    cudaError_t status = BatchQKApplyRotary(
        static_cast<c_type*>(q.data_ptr()), static_cast<c_type*>(k.data_ptr()),
        static_cast<c_type*>(q_rope.data_ptr()), static_cast<c_type*>(k_rope.data_ptr()),
        static_cast<int32_t*>(indptr.data_ptr()), static_cast<int32_t*>(offsets.data_ptr()),
        batch_size, num_qo_heads, num_kv_heads, rotary_dim, head_dim, q_stride_n, q_stride_h,
        k_stride_n, k_stride_h, q_rope_stride_n, q_rope_stride_h, k_rope_stride_n, k_rope_stride_h,
        interleave, rope_scale, rope_theta, stream);
    TORCH_CHECK(status == cudaSuccess, "BatchQKApplyRotary failed with error code " +
                                           std::string(cudaGetErrorString(status)));
    return true;
  });
}

void apply_rope_pos_ids(at::Tensor q, at::Tensor k, at::Tensor q_rope, at::Tensor k_rope,
                        at::Tensor pos_ids, unsigned int rotary_dim, bool interleave,
                        float rope_scale, float rope_theta, int64_t cuda_stream) {
  CHECK_LAST_DIM_CONTIGUOUS(q);
  CHECK_LAST_DIM_CONTIGUOUS(k);
  CHECK_INPUT(pos_ids);

  auto device = q.device();
  CHECK_EQ(k.device(), device);
  CHECK_DIM(3, q);  // q: (nnz, H_Q, D)
  CHECK_DIM(3, k);  // k: (nnz, H_K, D)
  CHECK_EQ(q.size(0), k.size(0));
  CHECK_EQ(q.size(2), k.size(2));
  unsigned int num_qo_heads = q.size(1);
  unsigned int num_kv_heads = k.size(1);
  unsigned int head_dim = q.size(2);
  unsigned int nnz = q.size(0);
  size_t q_stride_n = q.stride(0);
  size_t q_stride_h = q.stride(1);
  size_t k_stride_n = k.stride(0);
  size_t k_stride_h = k.stride(1);
  size_t q_rope_stride_n = q_rope.stride(0);
  size_t q_rope_stride_h = q_rope.stride(1);
  size_t k_rope_stride_n = k_rope.stride(0);
  size_t k_rope_stride_h = k_rope.stride(1);

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(q.scalar_type(), c_type, [&] {
    cudaError_t status = BatchQKApplyRotaryPosIds(
        static_cast<c_type*>(q.data_ptr()), static_cast<c_type*>(k.data_ptr()),
        static_cast<c_type*>(q_rope.data_ptr()), static_cast<c_type*>(k_rope.data_ptr()),
        static_cast<int32_t*>(pos_ids.data_ptr()), nnz, num_qo_heads, num_kv_heads, rotary_dim,
        head_dim, q_stride_n, q_stride_h, k_stride_n, k_stride_h, q_rope_stride_n, q_rope_stride_h,
        k_rope_stride_n, k_rope_stride_h, interleave, rope_scale, rope_theta, stream);
    TORCH_CHECK(status == cudaSuccess, "BatchQKApplyRotaryPosIds failed with error code " +
                                           std::string(cudaGetErrorString(status)));
    return true;
  });
}

void apply_rope_pos_ids_cos_sin_cache(at::Tensor q, at::Tensor k, at::Tensor q_rope,
                                      at::Tensor k_rope, at::Tensor cos_cache, at::Tensor sin_cache,
                                      at::Tensor pos_ids, bool interleave, int64_t cuda_stream) {
  CHECK_LAST_DIM_CONTIGUOUS(q);
  CHECK_LAST_DIM_CONTIGUOUS(k);
  CHECK_INPUT(cos_cache);
  CHECK_INPUT(sin_cache);
  CHECK_INPUT(pos_ids);
  auto device = q.device();
  CHECK_EQ(k.device(), device);
  CHECK_EQ(cos_cache.device(), device);
  CHECK_EQ(sin_cache.device(), device);
  CHECK_EQ(pos_ids.device(), device);
  CHECK_DIM(3, q);          // q: (nnz, H_Q, D)
  CHECK_DIM(3, k);          // k: (nnz, H_K, D)
  CHECK_DIM(2, cos_cache);  // cos_cache: (max_seq_len, D)
  CHECK_DIM(2, sin_cache);  // sin_cache: (max_seq_len, D)
  CHECK_EQ(q.size(0), k.size(0));
  CHECK_EQ(q.size(2), k.size(2));
  unsigned int rotary_dim = cos_cache.size(1);
  CHECK_EQ(sin_cache.size(1), rotary_dim);
  unsigned int num_qo_heads = q.size(1);
  unsigned int num_kv_heads = k.size(1);
  unsigned int head_dim = q.size(2);
  unsigned int nnz = q.size(0);
  size_t q_stride_n = q.stride(0);
  size_t q_stride_h = q.stride(1);
  size_t k_stride_n = k.stride(0);
  size_t k_stride_h = k.stride(1);
  size_t q_rope_stride_n = q_rope.stride(0);
  size_t q_rope_stride_h = q_rope.stride(1);
  size_t k_rope_stride_n = k_rope.stride(0);
  size_t k_rope_stride_h = k_rope.stride(1);

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(q.scalar_type(), c_type, [&] {
    cudaError_t status = BatchQKApplyRotaryPosIdsCosSinCache(
        static_cast<c_type*>(q.data_ptr()), static_cast<c_type*>(k.data_ptr()),
        static_cast<c_type*>(q_rope.data_ptr()), static_cast<c_type*>(k_rope.data_ptr()),
        static_cast<float*>(cos_cache.data_ptr()), static_cast<float*>(sin_cache.data_ptr()),
        static_cast<int32_t*>(pos_ids.data_ptr()), nnz, num_qo_heads, num_kv_heads, rotary_dim,
        head_dim, q_stride_n, q_stride_h, k_stride_n, k_stride_h, q_rope_stride_n, q_rope_stride_h,
        k_rope_stride_n, k_rope_stride_h, interleave, stream);
    TORCH_CHECK(status == cudaSuccess,
                "BatchQKApplyRotaryPosIdsCosSinCache failed with error code " +
                    std::string(cudaGetErrorString(status)));
    return true;
  });
}

void apply_llama31_rope(at::Tensor q, at::Tensor k, at::Tensor q_rope, at::Tensor k_rope,
                        at::Tensor indptr, at::Tensor offsets, unsigned int rotary_dim,
                        bool interleave, float rope_scale, float rope_theta, float low_freq_factor,
                        float high_freq_factor, float old_context_length, int64_t cuda_stream) {
  CHECK_CUDA(q);  // not necessarily contiguous
  CHECK_CUDA(k);  // not necessarily contiguous
  CHECK_INPUT(indptr);
  CHECK_INPUT(offsets);

  auto device = q.device();
  CHECK_EQ(k.device(), device);
  CHECK_DIM(3, q);        // q: (nnz, H_Q, D)
  CHECK_DIM(3, k);        // k: (nnz, H_K, D)
  CHECK_DIM(1, indptr);   // indptr: (B + 1)
  CHECK_DIM(1, offsets);  // offsets: (B)
  CHECK_EQ(q.size(0), k.size(0));
  CHECK_EQ(q.size(2), k.size(2));
  unsigned int num_qo_heads = q.size(1);
  unsigned int num_kv_heads = k.size(1);
  unsigned int head_dim = q.size(2);
  unsigned int batch_size = offsets.size(0);
  CHECK_EQ(indptr.size(0), batch_size + 1);
  size_t q_stride_n = q.stride(0);
  size_t q_stride_h = q.stride(1);
  size_t k_stride_n = k.stride(0);
  size_t k_stride_h = k.stride(1);
  size_t q_rope_stride_n = q_rope.stride(0);
  size_t q_rope_stride_h = q_rope.stride(1);
  size_t k_rope_stride_n = k_rope.stride(0);
  size_t k_rope_stride_h = k_rope.stride(1);

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(q.scalar_type(), c_type, [&] {
    cudaError_t status = BatchQKApplyLlama31Rotary(
        static_cast<c_type*>(q.data_ptr()), static_cast<c_type*>(k.data_ptr()),
        static_cast<c_type*>(q_rope.data_ptr()), static_cast<c_type*>(k_rope.data_ptr()),
        static_cast<int32_t*>(indptr.data_ptr()), static_cast<int32_t*>(offsets.data_ptr()),
        batch_size, num_qo_heads, num_kv_heads, rotary_dim, head_dim, q_stride_n, q_stride_h,
        k_stride_n, k_stride_h, q_rope_stride_n, q_rope_stride_h, k_rope_stride_n, k_rope_stride_h,
        interleave, rope_scale, rope_theta, low_freq_factor, high_freq_factor, old_context_length,
        stream);
    TORCH_CHECK(status == cudaSuccess, "BatchQKApplyLlama31Rotary failed with error code " +
                                           std::string(cudaGetErrorString(status)));
    return true;
  });
}

void apply_llama31_rope_pos_ids(at::Tensor q, at::Tensor k, at::Tensor q_rope, at::Tensor k_rope,
                                at::Tensor pos_ids, unsigned int rotary_dim, bool interleave,
                                float rope_scale, float rope_theta, float low_freq_factor,
                                float high_freq_factor, float old_context_length,
                                int64_t cuda_stream) {
  CHECK_CUDA(q);  // not necessarily contiguous
  CHECK_CUDA(k);  // not necessarily contiguous
  CHECK_INPUT(pos_ids);

  auto device = q.device();
  CHECK_EQ(k.device(), device);
  CHECK_DIM(3, q);  // q: (nnz, H_Q, D)
  CHECK_DIM(3, k);  // k: (nnz, H_K, D)
  CHECK_EQ(q.size(0), k.size(0));
  CHECK_EQ(q.size(2), k.size(2));
  unsigned int num_qo_heads = q.size(1);
  unsigned int num_kv_heads = k.size(1);
  unsigned int head_dim = q.size(2);
  unsigned int nnz = q.size(0);
  size_t q_stride_n = q.stride(0);
  size_t q_stride_h = q.stride(1);
  size_t k_stride_n = k.stride(0);
  size_t k_stride_h = k.stride(1);
  size_t q_rope_stride_n = q_rope.stride(0);
  size_t q_rope_stride_h = q_rope.stride(1);
  size_t k_rope_stride_n = k_rope.stride(0);
  size_t k_rope_stride_h = k_rope.stride(1);

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(q.scalar_type(), c_type, [&] {
    cudaError_t status = BatchQKApplyLlama31RotaryPosIds(
        static_cast<c_type*>(q.data_ptr()), static_cast<c_type*>(k.data_ptr()),
        static_cast<c_type*>(q_rope.data_ptr()), static_cast<c_type*>(k_rope.data_ptr()),
        static_cast<int32_t*>(pos_ids.data_ptr()), nnz, num_qo_heads, num_kv_heads, rotary_dim,
        head_dim, q_stride_n, q_stride_h, k_stride_n, k_stride_h, q_rope_stride_n, q_rope_stride_h,
        k_rope_stride_n, k_rope_stride_h, interleave, rope_scale, rope_theta, low_freq_factor,
        high_freq_factor, old_context_length, stream);
    TORCH_CHECK(status == cudaSuccess, "BatchQKApplyLlama31RotaryPosIds failed with error code " +
                                           std::string(cudaGetErrorString(status)));
    return true;
  });
}
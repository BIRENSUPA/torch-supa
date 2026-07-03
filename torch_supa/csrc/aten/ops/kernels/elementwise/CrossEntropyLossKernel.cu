#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

#include <ATen/core/Reduction.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/DeviceUtils.cuh>
#include <ATen/ops/empty.h>
#include <c10/macros/Macros.h>
#include <limits>
#include <supa_runtime.h>

#include "torch_supa/csrc/core/supa/SUPAException.h"
#include "torch_supa/csrc/core/supa/SUPAMacros.h"
#include "torch_supa/csrc/core/supa/SUPAStream.h"

namespace at::native {
namespace {

constexpr int kThreads = 256;
constexpr int kSmallClassThreads = 32;
constexpr int kHalfClassThreads = 16;
constexpr int kQuarterClassThreads = 8;
constexpr int kSmallRowsPerBlock = 256;
constexpr int kWarpRowsPerBlock = kThreads / kSmallClassThreads;
constexpr int kHalfWarpRowsPerBlock = kThreads / kHalfClassThreads;
constexpr int kQuarterWarpRowsPerBlock = kThreads / kQuarterClassThreads;

__device__ float block_reduce_sum(float value) {
  __shared__ float scratch[kThreads];
  scratch[threadIdx.x] = value;
  __syncthreads();

  for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      scratch[threadIdx.x] += scratch[threadIdx.x + stride];
    }
    __syncthreads();
  }
  return scratch[0];
}

__device__ float block_reduce_max(float value) {
  __shared__ float scratch[kThreads];
  scratch[threadIdx.x] = value;
  __syncthreads();

  for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      scratch[threadIdx.x] =
          scratch[threadIdx.x] > scratch[threadIdx.x + stride]
          ? scratch[threadIdx.x]
          : scratch[threadIdx.x + stride];
    }
    __syncthreads();
  }
  return scratch[0];
}

__global__ void cross_entropy_zero_kernel(float* output, float* count) {
  if (threadIdx.x == 0) {
    output[0] = 0.0f;
    count[0] = 0.0f;
  }
}

__global__ void cross_entropy_finalize_mean_kernel(float* output, const float* count) {
  if (threadIdx.x == 0) {
    output[0] = count[0] > 0.0f ? output[0] / count[0] : __builtin_nanf("");
  }
}

__global__ void cross_entropy_index_kernel(
    const float* __restrict__ input,
    const int64_t* __restrict__ target,
    float* __restrict__ output,
    float* __restrict__ count,
    int64_t rows,
    int64_t classes,
    int64_t ignore_index,
    int64_t reduction) {
  const int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  const int64_t cur_target = target[row];
  if (cur_target == ignore_index) {
    if (reduction == at::Reduction::None && threadIdx.x == 0) {
      output[row] = 0.0f;
    }
    return;
  }
  CUDA_KERNEL_ASSERT(cur_target >= 0 && cur_target < classes);

  const float* row_input = input + row * classes;
  float local_max = -CUDART_INF_F;
  for (int64_t col = threadIdx.x; col < classes; col += blockDim.x) {
    const float value = row_input[col];
    local_max = value > local_max ? value : local_max;
  }
  const float row_max = block_reduce_max(local_max);

  float local_sum = 0.0f;
  for (int64_t col = threadIdx.x; col < classes; col += blockDim.x) {
    local_sum += expf(row_input[col] - row_max);
  }
  const float row_sum = block_reduce_sum(local_sum);

  if (threadIdx.x == 0) {
    const float loss = logf(row_sum) + row_max - row_input[cur_target];
    if (reduction == at::Reduction::None) {
      output[row] = loss;
    } else {
      atomicAdd(output, loss);
      if (reduction == at::Reduction::Mean) {
        atomicAdd(count, 1.0f);
      }
    }
  }
}

__global__ void cross_entropy_index_mean_kernel(
    const float* __restrict__ input,
    const int64_t* __restrict__ target,
    float* __restrict__ losses,
    float* __restrict__ counts,
    int64_t rows,
    int64_t classes,
    int64_t ignore_index) {
  const int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  const int64_t cur_target = target[row];
  if (cur_target == ignore_index) {
    if (threadIdx.x == 0) {
      losses[row] = 0.0f;
      counts[row] = 0.0f;
    }
    return;
  }
  CUDA_KERNEL_ASSERT(cur_target >= 0 && cur_target < classes);

  const float* row_input = input + row * classes;
  float local_max = -CUDART_INF_F;
  for (int64_t col = threadIdx.x; col < classes; col += blockDim.x) {
    const float value = row_input[col];
    local_max = value > local_max ? value : local_max;
  }
  const float row_max = block_reduce_max(local_max);

  float local_sum = 0.0f;
  for (int64_t col = threadIdx.x; col < classes; col += blockDim.x) {
    local_sum += expf(row_input[col] - row_max);
  }
  const float row_sum = block_reduce_sum(local_sum);

  if (threadIdx.x == 0) {
    losses[row] = logf(row_sum) + row_max - row_input[cur_target];
    counts[row] = 1.0f;
  }
}

__global__ void cross_entropy_finalize_mean_buffer_kernel(
    const float* __restrict__ losses,
    const float* __restrict__ counts,
    float* __restrict__ output,
    int64_t rows) {
  float local_loss = 0.0f;
  float local_count = 0.0f;
  for (int64_t row = threadIdx.x; row < rows; row += blockDim.x) {
    local_loss += losses[row];
    local_count += counts[row];
  }

  __shared__ float loss_scratch[kThreads];
  __shared__ float count_scratch[kThreads];
  loss_scratch[threadIdx.x] = local_loss;
  count_scratch[threadIdx.x] = local_count;
  __syncthreads();

  for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      loss_scratch[threadIdx.x] += loss_scratch[threadIdx.x + stride];
      count_scratch[threadIdx.x] += count_scratch[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    output[0] = count_scratch[0] > 0.0f
        ? loss_scratch[0] / count_scratch[0]
        : __builtin_nanf("");
  }
}

__device__ float small_block_reduce_sum(float value) {
  __shared__ float scratch[kSmallClassThreads];
  scratch[threadIdx.x] = value;
  __syncthreads();

  for (int stride = kSmallClassThreads / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      scratch[threadIdx.x] += scratch[threadIdx.x + stride];
    }
    __syncthreads();
  }
  return scratch[0];
}

__device__ float small_block_reduce_max(float value) {
  __shared__ float scratch[kSmallClassThreads];
  scratch[threadIdx.x] = value;
  __syncthreads();

  for (int stride = kSmallClassThreads / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      scratch[threadIdx.x] =
          scratch[threadIdx.x] > scratch[threadIdx.x + stride]
          ? scratch[threadIdx.x]
          : scratch[threadIdx.x + stride];
    }
    __syncthreads();
  }
  return scratch[0];
}

__global__ void cross_entropy_index_small_none_kernel(
    const float* __restrict__ input,
    const int64_t* __restrict__ target,
    float* __restrict__ output,
    int64_t rows,
    int64_t classes,
    int64_t ignore_index) {
  const int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  const int64_t cur_target = target[row];
  if (cur_target == ignore_index) {
    if (threadIdx.x == 0) {
      output[row] = 0.0f;
    }
    return;
  }
  CUDA_KERNEL_ASSERT(cur_target >= 0 && cur_target < classes);

  const float* row_input = input + row * classes;
  float local_max = -CUDART_INF_F;
  for (int64_t col = threadIdx.x; col < classes; col += blockDim.x) {
    const float value = row_input[col];
    local_max = value > local_max ? value : local_max;
  }
  const float row_max = small_block_reduce_max(local_max);

  float local_sum = 0.0f;
  for (int64_t col = threadIdx.x; col < classes; col += blockDim.x) {
    local_sum += expf(row_input[col] - row_max);
  }
  const float row_sum = small_block_reduce_sum(local_sum);

  if (threadIdx.x == 0) {
    output[row] = logf(row_sum) + row_max - row_input[cur_target];
  }
}

__global__ void cross_entropy_index_small_none_row_kernel(
    const float* __restrict__ input,
    const int64_t* __restrict__ target,
    float* __restrict__ output,
    int64_t rows,
    int64_t classes,
    int64_t ignore_index) {
  const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= rows) {
    return;
  }

  const int64_t cur_target = target[row];
  if (cur_target == ignore_index) {
    output[row] = 0.0f;
    return;
  }
  CUDA_KERNEL_ASSERT(cur_target >= 0 && cur_target < classes);

  const float* row_input = input + row * classes;
  float row_max = -CUDART_INF_F;
  for (int64_t col = 0; col < classes; ++col) {
    const float value = row_input[col];
    row_max = value > row_max ? value : row_max;
  }

  float row_sum = 0.0f;
  for (int64_t col = 0; col < classes; ++col) {
    row_sum += expf(row_input[col] - row_max);
  }
  output[row] = logf(row_sum) + row_max - row_input[cur_target];
}

__global__ void cross_entropy_index_small32_none_warp_kernel(
    const float* __restrict__ input,
    const int64_t* __restrict__ target,
    float* __restrict__ output,
    int64_t rows,
    int64_t classes,
    int64_t ignore_index) {
  const int lane = threadIdx.x % kSmallClassThreads;
  const int group = threadIdx.x / kSmallClassThreads;
  const int64_t row =
      static_cast<int64_t>(blockIdx.x) * kWarpRowsPerBlock + group;

  float value = -CUDART_INF_F;
  int64_t cur_target = -1;
  if (row < rows) {
    cur_target = target[row];
    if (cur_target == ignore_index) {
      if (lane == 0) {
        output[row] = 0.0f;
      }
    } else {
      CUDA_KERNEL_ASSERT(cur_target >= 0 && cur_target < classes);
      if (lane < classes) {
        value = input[row * classes + lane];
      }
    }
  }

  for (int stride = kSmallClassThreads / 2; stride > 0; stride >>= 1) {
    const float rhs = WARP_SHFL_XOR(value, stride, kSmallClassThreads);
    value = value > rhs ? value : rhs;
  }
  const float row_max = value;

  float exp_value = 0.0f;
  if (row < rows && cur_target != ignore_index && lane < classes) {
    exp_value = expf(input[row * classes + lane] - row_max);
  }
  for (int stride = kSmallClassThreads / 2; stride > 0; stride >>= 1) {
    exp_value += WARP_SHFL_XOR(exp_value, stride, kSmallClassThreads);
  }

  if (row < rows && cur_target != ignore_index && lane == 0) {
    output[row] =
        logf(exp_value) + row_max -
        input[row * classes + cur_target];
  }
}

__global__ void cross_entropy_index_32_none_warp_kernel(
    const float* __restrict__ input,
    const int64_t* __restrict__ target,
    float* __restrict__ output,
    int64_t rows,
    int64_t ignore_index) {
  const int lane = threadIdx.x & (kSmallClassThreads - 1);
  const int group = threadIdx.x >> 5;
  const int64_t row =
      static_cast<int64_t>(blockIdx.x) * kWarpRowsPerBlock + group;

  if (row >= rows) {
    return;
  }

  const int64_t cur_target = target[row];
  if (cur_target == ignore_index) {
    if (lane == 0) {
      output[row] = 0.0f;
    }
    return;
  }
  CUDA_KERNEL_ASSERT(cur_target >= 0 && cur_target < kSmallClassThreads);

  const int64_t base = row * kSmallClassThreads;
  const float lane_value = input[base + lane];
  float value = lane_value;
  for (int stride = kSmallClassThreads / 2; stride > 0; stride >>= 1) {
    const float rhs = WARP_SHFL_XOR(value, stride, kSmallClassThreads);
    value = value > rhs ? value : rhs;
  }
  const float row_max = value;

  float exp_value = expf(lane_value - row_max);
  for (int stride = kSmallClassThreads / 2; stride > 0; stride >>= 1) {
    exp_value += WARP_SHFL_XOR(exp_value, stride, kSmallClassThreads);
  }

  const float target_value =
      WARP_SHFL(lane_value, static_cast<int>(cur_target), kSmallClassThreads);
  if (lane == 0) {
    output[row] = logf(exp_value) + row_max - target_value;
  }
}

__global__ void cross_entropy_index_32_noignore_warp_kernel(
    const float* __restrict__ input,
    const int64_t* __restrict__ target,
    float* __restrict__ output,
    int rows) {
  const int lane = threadIdx.x & (kSmallClassThreads - 1);
  const int group = threadIdx.x >> 5;
  const int row = static_cast<int>(blockIdx.x) * kWarpRowsPerBlock + group;

  if (row >= rows) {
    return;
  }

  const int cur_target = static_cast<int>(target[row]);
  CUDA_KERNEL_ASSERT(cur_target >= 0 && cur_target < kSmallClassThreads);

  const int base = row << 5;
  const float lane_value = input[base + lane];
  float value = lane_value;
  for (int stride = kSmallClassThreads / 2; stride > 0; stride >>= 1) {
    const float rhs = WARP_SHFL_XOR(value, stride, kSmallClassThreads);
    value = value > rhs ? value : rhs;
  }
  const float row_max = value;

  float exp_value = expf(lane_value - row_max);
  for (int stride = kSmallClassThreads / 2; stride > 0; stride >>= 1) {
    exp_value += WARP_SHFL_XOR(exp_value, stride, kSmallClassThreads);
  }

  const float target_value =
      WARP_SHFL(lane_value, cur_target, kSmallClassThreads);
  if (lane == 0) {
    output[row] = logf(exp_value) + row_max - target_value;
  }
}

__global__ void cross_entropy_index_32_noignore_halfwarp_kernel(
    const float* __restrict__ input,
    const int64_t* __restrict__ target,
    float* __restrict__ output,
    int rows) {
  const int lane = threadIdx.x & (kHalfClassThreads - 1);
  const int group = threadIdx.x >> 4;
  const int row = static_cast<int>(blockIdx.x) * kHalfWarpRowsPerBlock + group;

  if (row >= rows) {
    return;
  }

  const int cur_target = static_cast<int>(target[row]);
  CUDA_KERNEL_ASSERT(cur_target >= 0 && cur_target < kSmallClassThreads);

  const int base = row << 5;
  const float value0 = input[base + lane];
  const float value1 = input[base + lane + kHalfClassThreads];
  float value = value0 > value1 ? value0 : value1;
  for (int stride = kHalfClassThreads / 2; stride > 0; stride >>= 1) {
    const float rhs = WARP_SHFL_XOR(value, stride, kHalfClassThreads);
    value = value > rhs ? value : rhs;
  }
  const float row_max = value;

  float exp_value = expf(value0 - row_max) + expf(value1 - row_max);
  for (int stride = kHalfClassThreads / 2; stride > 0; stride >>= 1) {
    exp_value += WARP_SHFL_XOR(exp_value, stride, kHalfClassThreads);
  }

  float target_value = 0.0f;
  if (cur_target == lane) {
    target_value = value0;
  } else if (cur_target == lane + kHalfClassThreads) {
    target_value = value1;
  }
  for (int stride = kHalfClassThreads / 2; stride > 0; stride >>= 1) {
    target_value += WARP_SHFL_XOR(target_value, stride, kHalfClassThreads);
  }
  if (lane == 0) {
    output[row] = logf(exp_value) + row_max - target_value;
  }
}

__global__ void cross_entropy_index_32_noignore_quarterwarp_kernel(
    const float* __restrict__ input,
    const int64_t* __restrict__ target,
    float* __restrict__ output,
    int rows) {
  const int lane = threadIdx.x & (kQuarterClassThreads - 1);
  const int group = threadIdx.x >> 3;
  const int row = static_cast<int>(blockIdx.x) * kQuarterWarpRowsPerBlock + group;

  if (row >= rows) {
    return;
  }

  const int cur_target = static_cast<int>(target[row]);
  CUDA_KERNEL_ASSERT(cur_target >= 0 && cur_target < kSmallClassThreads);

  const int base = row << 5;
  const float value0 = input[base + lane];
  const float value1 = input[base + lane + kQuarterClassThreads];
  const float value2 = input[base + lane + kQuarterClassThreads * 2];
  const float value3 = input[base + lane + kQuarterClassThreads * 3];
  float value = value0 > value1 ? value0 : value1;
  value = value > value2 ? value : value2;
  value = value > value3 ? value : value3;
  for (int stride = kQuarterClassThreads / 2; stride > 0; stride >>= 1) {
    const float rhs = WARP_SHFL_XOR(value, stride, kQuarterClassThreads);
    value = value > rhs ? value : rhs;
  }
  const float row_max = value;

  float exp_value =
      expf(value0 - row_max) + expf(value1 - row_max) +
      expf(value2 - row_max) + expf(value3 - row_max);
  for (int stride = kQuarterClassThreads / 2; stride > 0; stride >>= 1) {
    exp_value += WARP_SHFL_XOR(exp_value, stride, kQuarterClassThreads);
  }

  const int target_group = cur_target >> 3;
  float selected_target_value = value0;
  if (target_group == 1) {
    selected_target_value = value1;
  } else if (target_group == 2) {
    selected_target_value = value2;
  } else if (target_group == 3) {
    selected_target_value = value3;
  }
  const float target_value = WARP_SHFL(
      selected_target_value,
      cur_target & (kQuarterClassThreads - 1),
      kQuarterClassThreads);
  if (lane == 0) {
    output[row] = logf(exp_value) + row_max - target_value;
  }
}

__global__ void cross_entropy_prob_kernel(
    const float* __restrict__ input,
    const float* __restrict__ target,
    float* __restrict__ output,
    float* __restrict__ count,
    int64_t rows,
    int64_t classes,
    int64_t reduction) {
  const int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  const float* row_input = input + row * classes;
  const float* row_target = target + row * classes;
  float local_max = -CUDART_INF_F;
  for (int64_t col = threadIdx.x; col < classes; col += blockDim.x) {
    const float value = row_input[col];
    local_max = value > local_max ? value : local_max;
  }
  const float row_max = block_reduce_max(local_max);

  float local_exp_sum = 0.0f;
  for (int64_t col = threadIdx.x; col < classes; col += blockDim.x) {
    local_exp_sum += expf(row_input[col] - row_max);
  }
  const float log_denom = logf(block_reduce_sum(local_exp_sum)) + row_max;

  float local_loss = 0.0f;
  for (int64_t col = threadIdx.x; col < classes; col += blockDim.x) {
    local_loss += row_target[col] * (log_denom - row_input[col]);
  }
  const float row_loss = block_reduce_sum(local_loss);

  if (threadIdx.x == 0) {
    if (reduction == at::Reduction::None) {
      output[row] = row_loss;
    } else {
      atomicAdd(output, row_loss);
      if (reduction == at::Reduction::Mean) {
        atomicAdd(count, 1.0f);
      }
    }
  }
}

} // namespace

TORCH_SUPA_API Tensor cross_entropy_loss_2d_kernel_supa(
    const Tensor& self,
    const Tensor& target,
    int64_t reduction,
    int64_t ignore_index) {
  const auto rows = self.size(0);
  const auto classes = self.size(1);
  TORCH_CHECK(
      rows <= std::numeric_limits<int>::max(),
      "cross_entropy_loss SUPA kernel only supports rows <= ",
      std::numeric_limits<int>::max(),
      ", got ",
      rows);
  Tensor output = reduction == at::Reduction::None
      ? at::empty({rows}, self.options())
      : at::empty({}, self.options());
  Tensor count;
  Tensor mean_losses;
  Tensor mean_counts;
  const bool use_index_mean_buffer =
      target.scalar_type() == at::kLong && reduction == at::Reduction::Mean;

  const auto stream = c10::supa::getCurrentSUPAStream();
  if (use_index_mean_buffer) {
    mean_losses = at::empty({rows}, self.options());
    mean_counts = at::empty({rows}, self.options());
  } else if (reduction != at::Reduction::None) {
    count = at::empty({}, self.options());
    cross_entropy_zero_kernel<<<1, 1, 0, stream>>>(
        output.mutable_data_ptr<float>(),
        count.mutable_data_ptr<float>());
  }

  if (target.scalar_type() == at::kLong &&
      reduction == at::Reduction::None &&
      classes <= kSmallClassThreads * 2) {
    if (classes <= kSmallClassThreads && rows >= kWarpRowsPerBlock) {
      if (classes == kSmallClassThreads && ignore_index < 0 && rows <= 2147483647) {
        if (rows < kSmallRowsPerBlock) {
          // Tiny C=32 batches are CP-duration sensitive. The half-warp mapping
          // was fastest in PFC for rows=128 because it avoids the extra per-lane
          // arithmetic of the quarter-warp variant.
          cross_entropy_index_32_noignore_halfwarp_kernel<<<
              static_cast<unsigned int>((rows + kHalfWarpRowsPerBlock - 1) / kHalfWarpRowsPerBlock),
              kThreads,
              0,
              stream>>>(
              self.const_data_ptr<float>(),
              target.const_data_ptr<int64_t>(),
              output.mutable_data_ptr<float>(),
              static_cast<int>(rows));
        } else {
          // Larger C=32 none workloads benefit from fewer row groups and a
          // narrower shuffle; each lane owns four classes.
          cross_entropy_index_32_noignore_quarterwarp_kernel<<<
              static_cast<unsigned int>((rows + kQuarterWarpRowsPerBlock - 1) / kQuarterWarpRowsPerBlock),
              kThreads,
              0,
              stream>>>(
              self.const_data_ptr<float>(),
              target.const_data_ptr<int64_t>(),
              output.mutable_data_ptr<float>(),
              static_cast<int>(rows));
        }
      } else if (classes == kSmallClassThreads) {
        cross_entropy_index_32_none_warp_kernel<<<
            static_cast<unsigned int>((rows + kWarpRowsPerBlock - 1) / kWarpRowsPerBlock),
            kThreads,
            0,
            stream>>>(
            self.const_data_ptr<float>(),
            target.const_data_ptr<int64_t>(),
            output.mutable_data_ptr<float>(),
            rows,
            ignore_index);
      } else {
        cross_entropy_index_small32_none_warp_kernel<<<
            static_cast<unsigned int>((rows + kWarpRowsPerBlock - 1) / kWarpRowsPerBlock),
            kThreads,
            0,
            stream>>>(
            self.const_data_ptr<float>(),
            target.const_data_ptr<int64_t>(),
            output.mutable_data_ptr<float>(),
            rows,
            classes,
            ignore_index);
      }
    } else if (rows >= kSmallRowsPerBlock) {
      cross_entropy_index_small_none_row_kernel<<<
          static_cast<unsigned int>((rows + kSmallRowsPerBlock - 1) / kSmallRowsPerBlock),
          kSmallRowsPerBlock,
          0,
          stream>>>(
          self.const_data_ptr<float>(),
          target.const_data_ptr<int64_t>(),
          output.mutable_data_ptr<float>(),
          rows,
          classes,
          ignore_index);
    } else {
      cross_entropy_index_small_none_kernel<<<
          static_cast<unsigned int>(rows),
          kSmallClassThreads,
          0,
          stream>>>(
          self.const_data_ptr<float>(),
          target.const_data_ptr<int64_t>(),
          output.mutable_data_ptr<float>(),
          rows,
          classes,
          ignore_index);
    }
  } else if (use_index_mean_buffer) {
    cross_entropy_index_mean_kernel<<<
        static_cast<unsigned int>(rows),
        kThreads,
        0,
        stream>>>(
        self.const_data_ptr<float>(),
        target.const_data_ptr<int64_t>(),
        mean_losses.mutable_data_ptr<float>(),
        mean_counts.mutable_data_ptr<float>(),
        rows,
        classes,
        ignore_index);
  } else if (target.scalar_type() == at::kLong) {
    cross_entropy_index_kernel<<<
        static_cast<unsigned int>(rows),
        kThreads,
        0,
        stream>>>(
        self.const_data_ptr<float>(),
        target.const_data_ptr<int64_t>(),
        output.mutable_data_ptr<float>(),
        reduction == at::Reduction::Mean ? count.mutable_data_ptr<float>() : nullptr,
        rows,
        classes,
        ignore_index,
        reduction);
  } else {
    cross_entropy_prob_kernel<<<
        static_cast<unsigned int>(rows),
        kThreads,
        0,
        stream>>>(
        self.const_data_ptr<float>(),
        target.const_data_ptr<float>(),
        output.mutable_data_ptr<float>(),
        reduction == at::Reduction::Mean ? count.mutable_data_ptr<float>() : nullptr,
        rows,
        classes,
        reduction);
  }

  if (use_index_mean_buffer) {
    cross_entropy_finalize_mean_buffer_kernel<<<1, kThreads, 0, stream>>>(
        mean_losses.const_data_ptr<float>(),
        mean_counts.const_data_ptr<float>(),
        output.mutable_data_ptr<float>(),
        rows);
  } else if (reduction == at::Reduction::Mean) {
    cross_entropy_finalize_mean_kernel<<<1, 1, 0, stream>>>(
        output.mutable_data_ptr<float>(),
        count.const_data_ptr<float>());
  }

  C10_SUPA_KERNEL_LAUNCH_CHECK();
  return output;
}

} // namespace at::native

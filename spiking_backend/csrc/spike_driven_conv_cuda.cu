#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include <cmath>
#include <limits>
#include <tuple>

namespace {

constexpr float kInf = std::numeric_limits<float>::infinity();
constexpr int kWarpSize = 32;
constexpr int kGatherMaxBlock = 256;
constexpr int kGatherSmemBudgetBytes = 32 * 1024;
// Channels staged into shared memory per tile, bounding the input scratch footprint.
constexpr int kChannelChunk = 16;

// Stage each channel tile of the receptive field into shared memory, then bin per-filter.
// Every thread must reach both __syncthreads(), so threads with f >= F still load smem.
__device__ __forceinline__ void gather_bins_tiled(
    const float* __restrict__ input_times,
    const float* __restrict__ weights,
    int b, int C, int H, int W, int f, int F, int kH, int kW,
    int oh, int ow, int stride, int pad, int num_bins,
    float* smem_inputs, float* my_bins
) {
  const int tid = threadIdx.x;
  const int CkHkW = C * kH * kW;
  for (int i = 0; i < num_bins; ++i) my_bins[i] = 0.0f;

  for (int c_start = 0; c_start < C; c_start += kChannelChunk) {
    const int chunk_C = (C - c_start) < kChannelChunk ? (C - c_start) : kChannelChunk;
    const int chunk_size = chunk_C * kH * kW;

    for (int i = tid; i < chunk_size; i += blockDim.x) {
      int c = c_start + i / (kH * kW);
      int ky = (i / kW) % kH;
      int kx = i % kW;
      int y = oh * stride + ky - pad;
      int x = ow * stride + kx - pad;
      smem_inputs[i] = (y >= 0 && y < H && x >= 0 && x < W)
          ? input_times[(((size_t)b * C + c) * H + y) * W + x]
          : kInf;
    }
    __syncthreads();

    if (f < F) {
      const float* W_f = weights + (size_t)f * CkHkW + (size_t)c_start * kH * kW;
      for (int i = 0; i < chunk_size; ++i) {
        float t = smem_inputs[i];
        if (!isfinite(t)) continue;
        int bin = (int)(t * num_bins);
        bin = bin < 0 ? 0 : (bin >= num_bins ? num_bins - 1 : bin);
        my_bins[bin] += W_f[i];
      }
    }
    __syncthreads();
  }
}

template <bool WITH_TOTAL>
__device__ __forceinline__ float scan_bins(
    const float* my_bins, int num_bins, float th, float* total_out
) {
  float cum = 0.0f, spike_t = kInf;
  if constexpr (WITH_TOTAL) {
    for (int bin = 0; bin < num_bins; ++bin) {
      cum += my_bins[bin];
      if (spike_t == kInf && cum >= th) spike_t = (float)bin / (float)num_bins;
    }
    *total_out = cum;
    return spike_t;
  } else {
    for (int bin = 0; bin < num_bins; ++bin) {
      cum += my_bins[bin];
      if (cum >= th) return (float)bin / (float)num_bins;
    }
    return kInf;
  }
}

template <bool WITH_POT>
__global__ void gather_first_spike_kernel_t(
    const float* __restrict__ input_times,
    const float* __restrict__ weights,
    const float* __restrict__ thresholds,
    int B, int C, int H, int W, int F, int kH, int kW,
    int stride, int pad, int oH, int oW, int num_bins,
    float* __restrict__ spike_times,
    float* __restrict__ cum_potential
) {
  extern __shared__ float smem[];
  const int spatial_idx = blockIdx.x;
  const int f_block = blockIdx.y;
  const int chunk_C = C < kChannelChunk ? C : kChannelChunk;

  float* smem_inputs = smem;
  float* my_bins = smem + chunk_C * kH * kW + (size_t)threadIdx.x * (num_bins + 1);

  const int ow = spatial_idx % oW;
  const int oh = (spatial_idx / oW) % oH;
  const int b = spatial_idx / (oH * oW);
  const int f = f_block * blockDim.x + threadIdx.x;

  gather_bins_tiled(input_times, weights, b, C, H, W, f, F, kH, kW,
                    oh, ow, stride, pad, num_bins, smem_inputs, my_bins);

  if (f >= F) return;

  float total;
  float st = scan_bins<WITH_POT>(my_bins, num_bins, thresholds[f], &total);

  size_t out_idx = (((size_t)b * F + f) * oH + oh) * oW + ow;
  spike_times[out_idx] = st;
  if constexpr (WITH_POT) {
    cum_potential[out_idx] = total;
  }
}

template <bool WITH_POT>
__global__ void gather_first_spike_multi_kernel_t(
    const float* __restrict__ input_times,
    const float* __restrict__ weights,
    const float* __restrict__ thresholds_2d,
    int B, int C, int H, int W, int F, int kH, int kW,
    int stride, int pad, int oH, int oW, int num_bins, int K,
    float* __restrict__ spike_times_K,
    float* __restrict__ cum_potential
) {
  extern __shared__ float smem[];
  const int spatial_idx = blockIdx.x;
  const int f_block = blockIdx.y;
  const int chunk_C = C < kChannelChunk ? C : kChannelChunk;

  float* smem_inputs = smem;
  float* my_bins = smem + chunk_C * kH * kW + (size_t)threadIdx.x * (num_bins + 1);

  const int ow = spatial_idx % oW;
  const int oh = (spatial_idx / oW) % oH;
  const int b = spatial_idx / (oH * oW);
  const int f = f_block * blockDim.x + threadIdx.x;

  gather_bins_tiled(input_times, weights, b, C, H, W, f, F, kH, kW,
                    oh, ow, stride, pad, num_bins, smem_inputs, my_bins);

  if (f >= F) return;

  if constexpr (WITH_POT) {
    float total = 0.0f;
    for (int bin = 0; bin < num_bins; ++bin) total += my_bins[bin];
    size_t out_idx = (((size_t)b * F + f) * oH + oh) * oW + ow;
    cum_potential[out_idx] = total;
  }

  const size_t FoHoW = (size_t)F * oH * oW;
  for (int k = 0; k < K; ++k) {
    float st = scan_bins<false>(my_bins, num_bins, thresholds_2d[(size_t)k * F + f], nullptr);
    size_t out_idx = ((size_t)k * B + b) * FoHoW + (size_t)f * oH * oW + oh * oW + ow;
    spike_times_K[out_idx] = st;
  }
}

static int pick_gather_block(int num_bins) {
  int t = kGatherSmemBudgetBytes / ((num_bins + 1) * (int)sizeof(float));
  if (t > kGatherMaxBlock) t = kGatherMaxBlock;
  t = (t / kWarpSize) * kWarpSize;
  return t < kWarpSize ? kWarpSize : t;
}

struct GatherLaunchConfig {
  int block;
  dim3 grid;
  size_t smem;
};

static GatherLaunchConfig make_gather_launch(
    int total_spatial, int F, int C, int kH, int kW, int num_bins
) {
  int block = pick_gather_block(num_bins);
  if (block > F) block = ((F + kWarpSize - 1) / kWarpSize) * kWarpSize;
  dim3 grid(total_spatial, (F + block - 1) / block);
  const int chunk_C = C < kChannelChunk ? C : kChannelChunk;
  const size_t smem = (size_t)(chunk_C * kH * kW) * sizeof(float)
                    + (size_t)block * (size_t)(num_bins + 1) * sizeof(float);
  return {block, grid, smem};
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> first_spike_times_cuda(
    at::Tensor input_times, at::Tensor weights_4d, at::Tensor thresholds,
    int64_t num_bins, int64_t stride, int64_t padding, bool compute_cum_potential
) {
  TORCH_CHECK(input_times.is_cuda() && weights_4d.is_cuda() && thresholds.is_cuda(),
              "tensors must be on CUDA");
  TORCH_CHECK(input_times.scalar_type() == at::kFloat, "fp32 only");
  TORCH_CHECK(num_bins > 0, "num_bins > 0");

  const at::cuda::OptionalCUDAGuard guard(input_times.device());
  auto in_c = input_times.contiguous();
  auto W_c = weights_4d.contiguous();
  auto th_c = thresholds.contiguous();

  const int B = in_c.size(0), C = in_c.size(1), H = in_c.size(2), W = in_c.size(3);
  const int F = W_c.size(0), kH = W_c.size(2), kW = W_c.size(3);
  const int oH = (H + 2 * padding - kH) / stride + 1;
  const int oW = (W + 2 * padding - kW) / stride + 1;
  auto opts = in_c.options();
  auto spike_times = at::full({B, F, oH, oW}, kInf, opts);
  auto cum_potential = compute_cum_potential
      ? at::zeros({B, F, oH, oW}, opts) : at::empty({0}, opts);

  const int total_spatial = B * oH * oW;
  if (total_spatial == 0) return {spike_times, cum_potential};

  auto cfg = make_gather_launch(total_spatial, F, C, kH, kW, (int)num_bins);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  auto launch = [&](auto kernel, float* pot_p) {
    kernel<<<cfg.grid, cfg.block, cfg.smem, stream>>>(
        in_c.data_ptr<float>(), W_c.data_ptr<float>(), th_c.data_ptr<float>(),
        B, C, H, W, F, kH, kW, (int)stride, (int)padding, oH, oW, (int)num_bins,
        spike_times.data_ptr<float>(), pot_p);
  };
  if (compute_cum_potential) launch(gather_first_spike_kernel_t<true>,
                                    cum_potential.data_ptr<float>());
  else                       launch(gather_first_spike_kernel_t<false>, nullptr);
  return {spike_times, cum_potential};
}

std::tuple<at::Tensor, at::Tensor> first_spike_times_multi_threshold_cuda(
    at::Tensor input_times, at::Tensor weights_4d, at::Tensor thresholds_2d,
    int64_t num_bins, int64_t stride, int64_t padding, bool compute_cum_potential
) {
  TORCH_CHECK(input_times.is_cuda() && weights_4d.is_cuda() && thresholds_2d.is_cuda(),
              "tensors must be on CUDA");
  TORCH_CHECK(input_times.scalar_type() == at::kFloat, "fp32 only");
  TORCH_CHECK(thresholds_2d.dim() == 2 && num_bins > 0, "(K,F), num_bins > 0");

  const at::cuda::OptionalCUDAGuard guard(input_times.device());
  auto in_c = input_times.contiguous();
  auto W_c = weights_4d.contiguous();
  auto th_c = thresholds_2d.contiguous();

  const int B = in_c.size(0), C = in_c.size(1), H = in_c.size(2), W = in_c.size(3);
  const int F = W_c.size(0), kH = W_c.size(2), kW = W_c.size(3), K = th_c.size(0);
  const int oH = (H + 2 * padding - kH) / stride + 1;
  const int oW = (W + 2 * padding - kW) / stride + 1;
  auto opts = in_c.options();
  auto spike_times = at::full({K, B, F, oH, oW}, kInf, opts);
  auto cum_potential = compute_cum_potential
      ? at::zeros({B, F, oH, oW}, opts) : at::empty({0}, opts);

  const int total_spatial = B * oH * oW;
  if (total_spatial == 0) return {spike_times, cum_potential};

  auto cfg = make_gather_launch(total_spatial, F, C, kH, kW, (int)num_bins);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  auto launch = [&](auto kernel, float* pot_p) {
    kernel<<<cfg.grid, cfg.block, cfg.smem, stream>>>(
        in_c.data_ptr<float>(), W_c.data_ptr<float>(), th_c.data_ptr<float>(),
        B, C, H, W, F, kH, kW, (int)stride, (int)padding,
        oH, oW, (int)num_bins, K,
        spike_times.data_ptr<float>(), pot_p);
  };
  if (compute_cum_potential) launch(gather_first_spike_multi_kernel_t<true>,
                                    cum_potential.data_ptr<float>());
  else                       launch(gather_first_spike_multi_kernel_t<false>, nullptr);
  return {spike_times, cum_potential};
}

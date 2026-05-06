// CUDA sparse-event spike-driven conv accumulator.
//
// Algorithm:
//   For each unique input time t (ascending), one kernel launch processes
//   *all* events firing at exactly t. Each event spawns a CUDA block; threads
//   within a block fan out across (filter, ky, kx). Per-thread inner loop
//   updates the cumulative potential at the affected output position via
//   atomicAdd, and records the first-crossing spike time in the only
//   ordering where atomicAdd makes the value cross the threshold.
//
// Why one launch per time: events at different times have different t values
// for the spike-time write, and within a single launch the atomicAdd order is
// non-deterministic. Sequencing by time guarantees that whatever ordering
// CUDA picks within one t, the spike time recorded is t.

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include <cmath>
#include <limits>
#include <tuple>

namespace {

constexpr float kInf = std::numeric_limits<float>::infinity();

__device__ inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

__device__ inline int affected_lo(int pos_padded, int kernel, int stride) {
  int n = pos_padded - kernel + 1;
  if (n <= 0) return 0;
  return (n + stride - 1) / stride;
}

__device__ inline int affected_hi(int pos_padded, int stride, int out_size) {
  int hi = pos_padded / stride + 1;
  return hi < out_size ? hi : out_size;
}

// One block per event at this time t. blockDim.x = F (one thread per filter).
__global__ void event_scatter_single(
    const int4* __restrict__ events,  // (N_events,) packed b,c,y,x
    int N_events,
    float t,
    const float* __restrict__ W,       // (F, C, kH, kW) row-major
    const float* __restrict__ th,      // (F,)
    int B, int C, int H, int W_W,
    int F, int kH, int kW,
    int stride, int pad,
    int oH, int oW,
    float* __restrict__ spike_times,    // (B, F, oH, oW)
    float* __restrict__ cum_potential   // (B, F, oH, oW)
) {
  const int eid = blockIdx.x;
  if (eid >= N_events) return;
  const int4 e = events[eid];
  const int b = e.x;
  const int c = e.y;
  const int y = e.z;
  const int x = e.w;
  const int yp = y + pad;
  const int xp = x + pad;
  const int oh_lo = affected_lo(yp, kH, stride);
  const int oh_hi = affected_hi(yp, stride, oH);
  const int ow_lo = affected_lo(xp, kW, stride);
  const int ow_hi = affected_hi(xp, stride, oW);
  if (oh_lo >= oh_hi || ow_lo >= ow_hi) return;

  const int f = threadIdx.x;
  if (f >= F) return;
  const float th_f = th[f];

  const int oHoW = oH * oW;
  const int FoHoW = F * oHoW;
  // weights row offset for this (filter, channel)
  const float* W_fc = W + ((f * C) + c) * kH * kW;
  float* pot_b_f = cum_potential + (b * F + f) * oHoW;
  float* st_b_f = spike_times + (b * F + f) * oHoW;
  (void)FoHoW;

  for (int oh = oh_lo; oh < oh_hi; ++oh) {
    const int ky = yp - oh * stride;
    for (int ow = ow_lo; ow < ow_hi; ++ow) {
      const int kx = xp - ow * stride;
      const float w = W_fc[ky * kW + kx];
      const int p_off = oh * oW + ow;
      float old_pot = atomicAdd(pot_b_f + p_off, w);
      float new_pot = old_pot + w;
      if (old_pot < th_f && new_pot >= th_f) {
        // Race-safe: only the atomicAdd that crossed wins this branch.
        st_b_f[p_off] = t;
      }
    }
  }
}

__global__ void event_scatter_multi(
    const int4* __restrict__ events,
    int N_events,
    float t,
    const float* __restrict__ W,
    const float* __restrict__ th_2d,   // (K, F)
    int B, int C, int H, int W_W,
    int F, int kH, int kW,
    int stride, int pad,
    int oH, int oW, int K,
    float* __restrict__ spike_times_K,  // (K, B, F, oH, oW)
    float* __restrict__ cum_potential   // (B, F, oH, oW)
) {
  const int eid = blockIdx.x;
  if (eid >= N_events) return;
  const int4 e = events[eid];
  const int b = e.x;
  const int c = e.y;
  const int y = e.z;
  const int x = e.w;
  const int yp = y + pad;
  const int xp = x + pad;
  const int oh_lo = affected_lo(yp, kH, stride);
  const int oh_hi = affected_hi(yp, stride, oH);
  const int ow_lo = affected_lo(xp, kW, stride);
  const int ow_hi = affected_hi(xp, stride, oW);
  if (oh_lo >= oh_hi || ow_lo >= ow_hi) return;

  const int f = threadIdx.x;
  if (f >= F) return;

  const int oHoW = oH * oW;
  const int FoHoW = F * oHoW;
  const float* W_fc = W + ((f * C) + c) * kH * kW;
  float* pot_b_f = cum_potential + (b * F + f) * oHoW;

  for (int oh = oh_lo; oh < oh_hi; ++oh) {
    const int ky = yp - oh * stride;
    for (int ow = ow_lo; ow < ow_hi; ++ow) {
      const int kx = xp - ow * stride;
      const float w = W_fc[ky * kW + kx];
      const int p_off = oh * oW + ow;
      float old_pot = atomicAdd(pot_b_f + p_off, w);
      float new_pot = old_pot + w;
      // Test all K threshold sets — only the crosser triggers the write.
      for (int k = 0; k < K; ++k) {
        float th_kf = th_2d[k * F + f];
        if (old_pot < th_kf && new_pot >= th_kf) {
          float* st_kbf = spike_times_K + (((int64_t)k * B + b) * F + f) * oHoW;
          st_kbf[p_off] = t;
        }
      }
    }
  }
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> spike_driven_conv_accumulate_cuda(
    at::Tensor input_times,
    at::Tensor weights_4d,
    at::Tensor thresholds,
    int64_t stride,
    int64_t padding) {
  TORCH_CHECK(input_times.is_cuda(), "input_times must be on CUDA");
  TORCH_CHECK(weights_4d.is_cuda(), "weights_4d must be on CUDA");
  TORCH_CHECK(thresholds.is_cuda(), "thresholds must be on CUDA");
  TORCH_CHECK(input_times.scalar_type() == at::kFloat, "fp32 only");
  TORCH_CHECK(weights_4d.scalar_type() == at::kFloat, "fp32 only");
  TORCH_CHECK(thresholds.scalar_type() == at::kFloat, "fp32 only");
  TORCH_CHECK(input_times.dim() == 4, "input_times must be (B,C,H,W)");
  TORCH_CHECK(weights_4d.dim() == 4, "weights_4d must be (F,C,kH,kW)");
  TORCH_CHECK(thresholds.dim() == 1, "thresholds must be (F,)");

  const at::cuda::OptionalCUDAGuard guard(input_times.device());

  auto in_c = input_times.contiguous();
  auto W_c = weights_4d.contiguous();
  auto th_c = thresholds.contiguous();

  const int B = in_c.size(0);
  const int C = in_c.size(1);
  const int H = in_c.size(2);
  const int W = in_c.size(3);
  const int F = W_c.size(0);
  TORCH_CHECK(W_c.size(1) == C, "weights C must match input C");
  const int kH = W_c.size(2);
  const int kW = W_c.size(3);
  TORCH_CHECK(th_c.size(0) == F, "thresholds size must equal F");
  const int oH = (H + 2 * padding - kH) / stride + 1;
  const int oW = (W + 2 * padding - kW) / stride + 1;
  TORCH_CHECK(oH > 0 && oW > 0, "non-positive output");

  auto opts = in_c.options();
  auto spike_times = at::full({B, F, oH, oW}, kInf, opts);
  auto cum_potential = at::zeros({B, F, oH, oW}, opts);

  // Compute the global set of unique input times across the batch.
  auto finite_mask = at::isfinite(in_c);
  if (!finite_mask.any().item<bool>()) {
    return {spike_times, cum_potential};
  }
  auto finite_times = in_c.masked_select(finite_mask);
  // torch::unique returns sorted ascending unique values
  auto unique_t = std::get<0>(at::_unique(finite_times, /*sorted=*/true));
  const int n_times = unique_t.size(0);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  for (int ti = 0; ti < n_times; ++ti) {
    float t = unique_t[ti].item<float>();
    auto mask_t = (in_c == t);
    auto event_idx = at::nonzero(mask_t).to(at::kInt);  // (N_events, 4)
    int N_events = event_idx.size(0);
    if (N_events == 0) continue;

    // Pack into int4 for the kernel.
    auto packed = event_idx.contiguous();
    const int4* events_p =
        reinterpret_cast<const int4*>(packed.data_ptr<int>());

    const int threads = F;  // one thread per filter
    const int blocks = N_events;
    event_scatter_single<<<blocks, threads, 0, stream>>>(
        events_p, N_events, t,
        W_c.data_ptr<float>(),
        th_c.data_ptr<float>(),
        B, C, H, W, F, kH, kW,
        (int)stride, (int)padding,
        oH, oW,
        spike_times.data_ptr<float>(),
        cum_potential.data_ptr<float>());
  }

  return {spike_times, cum_potential};
}

at::Tensor spike_driven_conv_accumulate_multi_threshold_cuda(
    at::Tensor input_times,
    at::Tensor weights_4d,
    at::Tensor thresholds_2d,
    int64_t stride,
    int64_t padding) {
  TORCH_CHECK(input_times.is_cuda(), "input_times must be on CUDA");
  TORCH_CHECK(input_times.scalar_type() == at::kFloat, "fp32 only");
  TORCH_CHECK(weights_4d.scalar_type() == at::kFloat, "fp32 only");
  TORCH_CHECK(thresholds_2d.scalar_type() == at::kFloat, "fp32 only");
  TORCH_CHECK(input_times.dim() == 4, "input_times must be (B,C,H,W)");
  TORCH_CHECK(weights_4d.dim() == 4, "weights_4d must be (F,C,kH,kW)");
  TORCH_CHECK(thresholds_2d.dim() == 2, "thresholds must be (K,F)");

  const at::cuda::OptionalCUDAGuard guard(input_times.device());

  auto in_c = input_times.contiguous();
  auto W_c = weights_4d.contiguous();
  auto th_c = thresholds_2d.contiguous();

  const int B = in_c.size(0);
  const int C = in_c.size(1);
  const int H = in_c.size(2);
  const int W = in_c.size(3);
  const int F = W_c.size(0);
  const int kH = W_c.size(2);
  const int kW = W_c.size(3);
  const int K = th_c.size(0);
  TORCH_CHECK(W_c.size(1) == C, "weights C must match input C");
  TORCH_CHECK(th_c.size(1) == F, "threshold last dim must equal F");
  const int oH = (H + 2 * padding - kH) / stride + 1;
  const int oW = (W + 2 * padding - kW) / stride + 1;

  auto opts = in_c.options();
  auto spike_times = at::full({K, B, F, oH, oW}, kInf, opts);
  auto cum_potential = at::zeros({B, F, oH, oW}, opts);

  auto finite_mask = at::isfinite(in_c);
  if (!finite_mask.any().item<bool>()) return spike_times;
  auto finite_times = in_c.masked_select(finite_mask);
  auto unique_t = std::get<0>(at::_unique(finite_times, /*sorted=*/true));
  const int n_times = unique_t.size(0);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  for (int ti = 0; ti < n_times; ++ti) {
    float t = unique_t[ti].item<float>();
    auto mask_t = (in_c == t);
    auto event_idx = at::nonzero(mask_t).to(at::kInt);
    int N_events = event_idx.size(0);
    if (N_events == 0) continue;

    auto packed = event_idx.contiguous();
    const int4* events_p =
        reinterpret_cast<const int4*>(packed.data_ptr<int>());

    const int threads = F;
    const int blocks = N_events;
    event_scatter_multi<<<blocks, threads, 0, stream>>>(
        events_p, N_events, t,
        W_c.data_ptr<float>(),
        th_c.data_ptr<float>(),
        B, C, H, W, F, kH, kW,
        (int)stride, (int)padding,
        oH, oW, K,
        spike_times.data_ptr<float>(),
        cum_potential.data_ptr<float>());
  }

  return spike_times;
}

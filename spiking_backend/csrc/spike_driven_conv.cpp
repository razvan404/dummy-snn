// CPU sparse-event spike-driven conv accumulator.
//
// Drop-in replacement for ConvIntegrateAndFireLayer._conv2d_accumulate and
// applications.threshold_research.conv_neuron_perturbation.multi_threshold_conv_accumulate.
//
// The key observation: latency-encoded inputs are sparse (~50% finite slots,
// distributed over many discretised timesteps so per-step density is ~1-2%).
// Dense F.conv2d burns ~99% of FLOPs on zero contributions. We instead loop
// over the actual events and push contributions into the small kH*kW output
// receptive field per event, matching the dense "accumulate everything for a
// given time, then check threshold" semantics.

#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <tuple>
#include <vector>

namespace {

constexpr float kInf = std::numeric_limits<float>::infinity();

// Half-open output index range whose receptive field covers a padded
// input position. Stride and kernel are positive ints.
inline std::pair<int, int> affected_out_range(
    int pos_padded, int kernel, int stride, int out_size) {
  int lo_num = pos_padded - kernel + 1;
  int lo = lo_num <= 0 ? 0 : (lo_num + stride - 1) / stride;
  int hi = std::min(out_size, pos_padded / stride + 1);
  if (lo < 0) lo = 0;
  if (hi < 0) hi = 0;
  return {lo, hi};
}

// Per-sample worker for the single-threshold case.
//
// Layout assumptions: contiguous float32 tensors. We reach in via raw
// pointers to avoid TensorAccessor overhead in the inner loops.
void run_single(
    int b, int C, int H, int W, int F, int kH, int kW,
    int stride, int padding, int oH, int oW,
    const float* in_b,           // (C, H, W)
    const float* W_base,         // (F, C, kH, kW)
    const float* th,             // (F,)
    float* st_b,                 // (F, oH, oW), pre-filled with inf
    float* pot_b                 // (F, oH, oW), pre-filled with 0
) {
  const int CHW = C * H * W;
  const int FoHoW = F * oH * oW;
  const int oHoW = oH * oW;
  const int kHkW = kH * kW;

  // Collect unique times present in this sample.
  // A small set in practice (<= 64 for our num_bins=64 encoding); a flat
  // vector with linear-scan insertion is faster than std::unordered_set
  // for these sizes and keeps memory footprint trivial.
  std::vector<float> uniq;
  uniq.reserve(64);
  for (int i = 0; i < CHW; ++i) {
    float v = in_b[i];
    if (std::isfinite(v)) {
      bool seen = false;
      for (float u : uniq)
        if (u == v) {
          seen = true;
          break;
        }
      if (!seen) uniq.push_back(v);
    }
  }
  std::sort(uniq.begin(), uniq.end());
  if (uniq.empty()) return;

  // We deliberately do NOT early-exit when every position has spiked. The
  // dense F.conv2d path keeps accumulating into ``cum_potential`` for spiked
  // positions through all subsequent timesteps; matching that semantic on
  // ``cum_potential`` requires us to do the same. If a future caller only
  // needs ``spike_times``, we can add a "no_potential" fast path that DOES
  // early-exit.
  for (float t : uniq) {
    // Phase 1: accumulate every event firing at this t.
    for (int c = 0; c < C; ++c) {
      const float* in_c = in_b + c * H * W;
      const float* W_fc_base = W_base + c * kH * kW;  // (F filters, this channel)
      for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
          if (in_c[y * W + x] != t) continue;
          const int yp = y + padding;
          const int xp = x + padding;
          auto [oh_lo, oh_hi] = affected_out_range(yp, kH, stride, oH);
          auto [ow_lo, ow_hi] = affected_out_range(xp, kW, stride, oW);
          if (oh_lo >= oh_hi || ow_lo >= ow_hi) continue;
          for (int oh = oh_lo; oh < oh_hi; ++oh) {
            const int ky = yp - oh * stride;
            for (int ow = ow_lo; ow < ow_hi; ++ow) {
              const int kx = xp - ow * stride;
              const int wt_offset = ky * kW + kx;  // within (kH*kW)
              float* pot_pos = pot_b + oh * oW + ow;
              const float* W_pos = W_fc_base + wt_offset;
              for (int f = 0; f < F; ++f) {
                pot_pos[f * oHoW] += W_pos[f * (C * kH * kW)];
              }
            }
          }
        }
      }
    }

    // Phase 2: check first-crossing only for positions that haven't spiked.
    for (int f = 0; f < F; ++f) {
      const float th_f = th[f];
      const int f_off = f * oHoW;
      for (int p = 0; p < oHoW; ++p) {
        const int idx = f_off + p;
        if (std::isinf(st_b[idx]) && pot_b[idx] >= th_f) {
          st_b[idx] = t;
        }
      }
    }
  }
  (void)FoHoW;
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> spike_driven_conv_accumulate_cpu(
    at::Tensor input_times,
    at::Tensor weights_4d,
    at::Tensor thresholds,
    int64_t stride,
    int64_t padding) {
  TORCH_CHECK(input_times.dim() == 4, "input_times must be (B,C,H,W)");
  TORCH_CHECK(weights_4d.dim() == 4, "weights_4d must be (F,C,kH,kW)");
  TORCH_CHECK(thresholds.dim() == 1, "thresholds must be (F,)");
  TORCH_CHECK(input_times.scalar_type() == at::kFloat, "fp32 only");
  TORCH_CHECK(weights_4d.scalar_type() == at::kFloat, "fp32 only");
  TORCH_CHECK(thresholds.scalar_type() == at::kFloat, "fp32 only");
  TORCH_CHECK(input_times.device().is_cpu(), "CPU kernel: input on CPU");
  TORCH_CHECK(weights_4d.device().is_cpu(), "CPU kernel: weights on CPU");
  TORCH_CHECK(thresholds.device().is_cpu(), "CPU kernel: thresholds on CPU");

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
  TORCH_CHECK(oH > 0 && oW > 0, "non-positive output size");

  auto opts = in_c.options();
  auto spike_times =
      at::full({B, F, oH, oW}, kInf, opts);
  auto cum_potential = at::zeros({B, F, oH, oW}, opts);

  const float* in_p = in_c.data_ptr<float>();
  const float* W_p = W_c.data_ptr<float>();
  const float* th_p = th_c.data_ptr<float>();
  float* st_p = spike_times.data_ptr<float>();
  float* pot_p = cum_potential.data_ptr<float>();

  const int CHW = C * H * W;
  const int FoHoW = F * oH * oW;

#pragma omp parallel for if (B > 1)
  for (int b = 0; b < B; ++b) {
    run_single(b, C, H, W, F, kH, kW, stride, padding, oH, oW,
               in_p + b * CHW, W_p, th_p,
               st_p + b * FoHoW, pot_p + b * FoHoW);
  }

  return {spike_times, cum_potential};
}

at::Tensor spike_driven_conv_accumulate_multi_threshold_cpu(
    at::Tensor input_times,
    at::Tensor weights_4d,
    at::Tensor thresholds_2d,
    int64_t stride,
    int64_t padding) {
  TORCH_CHECK(input_times.dim() == 4, "input_times must be (B,C,H,W)");
  TORCH_CHECK(weights_4d.dim() == 4, "weights_4d must be (F,C,kH,kW)");
  TORCH_CHECK(thresholds_2d.dim() == 2, "thresholds must be (K,F)");
  TORCH_CHECK(input_times.scalar_type() == at::kFloat, "fp32 only");
  TORCH_CHECK(weights_4d.scalar_type() == at::kFloat, "fp32 only");
  TORCH_CHECK(thresholds_2d.scalar_type() == at::kFloat, "fp32 only");
  TORCH_CHECK(input_times.device().is_cpu(), "CPU kernel: input on CPU");

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

  const int CHW = C * H * W;
  const int FoHoW = F * oH * oW;

  // cum_potential is per-sample but shared across K. Allocate (B, F, oH, oW).
  auto cum_potential = at::zeros({B, F, oH, oW}, opts);

  const float* in_p = in_c.data_ptr<float>();
  const float* W_p = W_c.data_ptr<float>();
  const float* th_p = th_c.data_ptr<float>();
  float* st_p = spike_times.data_ptr<float>();
  float* pot_p = cum_potential.data_ptr<float>();

#pragma omp parallel for if (B > 1)
  for (int b = 0; b < B; ++b) {
    // For multi-threshold, spike_times is (K, B, F, oH, oW); per-sample slice
    // has stride FoHoW within the K dim, which is K_FoHoW per K step.
    // Easier: pass a per-sample view via base pointer arithmetic that the
    // kernel re-strides itself.
    // Build a temp st_b layout: (K, F*oH*oW) — strided by K*B*FoHoW originally.
    // To avoid copying, we pass the global base and tell run_multi how to
    // address. Simpler: have run_multi take st_p (full), b, K, B, FoHoW so
    // that st_b[k] = st_p + k * B * FoHoW + b * FoHoW.
    //
    // Implementation note: rewrite run_multi accordingly.
    const float* in_b = in_p + b * CHW;
    float* pot_b = pot_p + b * FoHoW;

    // Per-sample, per-K base pointers
    // st_p layout: [k][b][f][oh][ow]; stride for k is B * FoHoW; for b is FoHoW.
    // We pass a small lambda environment via separate pointers.
    // For OpenMP simplicity, inline the inner loop directly rather than
    // calling run_multi (which expected (K, F, oH, oW) contiguous slice).

    // Collect unique times in this sample.
    std::vector<float> uniq;
    uniq.reserve(64);
    for (int i = 0; i < CHW; ++i) {
      float v = in_b[i];
      if (std::isfinite(v)) {
        bool seen = false;
        for (float u : uniq)
          if (u == v) {
            seen = true;
            break;
          }
        if (!seen) uniq.push_back(v);
      }
    }
    std::sort(uniq.begin(), uniq.end());
    if (uniq.empty()) continue;

    const int oHoW = oH * oW;

    for (float t : uniq) {
      for (int c = 0; c < C; ++c) {
        const float* in_c_slab = in_b + c * H * W;
        const float* W_fc_base = W_p + c * kH * kW;
        for (int y = 0; y < H; ++y) {
          for (int x = 0; x < W; ++x) {
            if (in_c_slab[y * W + x] != t) continue;
            const int yp = y + padding;
            const int xp = x + padding;
            auto [oh_lo, oh_hi] = affected_out_range(yp, kH, stride, oH);
            auto [ow_lo, ow_hi] = affected_out_range(xp, kW, stride, oW);
            if (oh_lo >= oh_hi || ow_lo >= ow_hi) continue;
            for (int oh = oh_lo; oh < oh_hi; ++oh) {
              const int ky = yp - oh * stride;
              for (int ow = ow_lo; ow < ow_hi; ++ow) {
                const int kx = xp - ow * stride;
                const int wt_offset = ky * kW + kx;
                float* pot_pos = pot_b + oh * oW + ow;
                const float* W_pos = W_fc_base + wt_offset;
                for (int f = 0; f < F; ++f) {
                  pot_pos[f * oHoW] += W_pos[f * (C * kH * kW)];
                }
              }
            }
          }
        }
      }

      for (int k = 0; k < K; ++k) {
        float* st_kb = st_p + k * B * FoHoW + b * FoHoW;
        const float* th_k = th_p + k * F;
        for (int f = 0; f < F; ++f) {
          const float th_f = th_k[f];
          const int f_off = f * oHoW;
          for (int p = 0; p < oHoW; ++p) {
            const int idx = f_off + p;
            if (std::isinf(st_kb[idx]) && pot_b[idx] >= th_f) {
              st_kb[idx] = t;
            }
          }
        }
      }
    }
  }

  return spike_times;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "spike-driven conv accumulator (CPU)";
  m.def(
      "spike_driven_conv_accumulate_cpu",
      &spike_driven_conv_accumulate_cpu,
      "Sparse-event spike-driven conv accumulate (CPU).",
      py::arg("input_times"),
      py::arg("weights_4d"),
      py::arg("thresholds"),
      py::arg("stride") = 1,
      py::arg("padding") = 0);
  m.def(
      "spike_driven_conv_accumulate_multi_threshold_cpu",
      &spike_driven_conv_accumulate_multi_threshold_cpu,
      "Multi-threshold variant.",
      py::arg("input_times"),
      py::arg("weights_4d"),
      py::arg("thresholds_2d"),
      py::arg("stride") = 1,
      py::arg("padding") = 0);
}

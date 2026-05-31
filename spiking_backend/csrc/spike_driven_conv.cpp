#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <tuple>
#include <vector>

namespace {

constexpr float kInf = std::numeric_limits<float>::infinity();

// Below this many output sites, parallelise over (site × filter) for enough work units.
constexpr size_t kSpatialParallelThreshold = 8;

inline std::pair<int, int> affected_out_range(
    int pos_padded, int kernel, int stride, int out_size) {
  int lo_num = pos_padded - kernel + 1;
  int lo = lo_num <= 0 ? 0 : (lo_num + stride - 1) / stride;
  int hi = std::min(out_size, pos_padded / stride + 1);
  return {std::max(0, lo), std::max(0, hi)};
}

inline std::vector<float> collect_unique_times(
    const float* in, int CHW, int num_bins
) {
  std::vector<float> uniq;
  uniq.reserve((size_t)num_bins);
  for (int i = 0; i < CHW; ++i) {
    float v = in[i];
    if (!std::isfinite(v)) continue;
    bool seen = false;
    for (float u : uniq) if (u == v) { seen = true; break; }
    if (!seen) uniq.push_back(v);
  }
  std::sort(uniq.begin(), uniq.end());
  return uniq;
}

struct ConvShape {
  int B, C, H, W, F, kH, kW, oH, oW;
  int CHW() const { return C * H * W; }
  int FoHoW() const { return F * oH * oW; }
};

inline ConvShape extract_shape(
    const at::Tensor& in, const at::Tensor& W, int stride, int padding
) {
  TORCH_CHECK(in.dim() == 4 && W.dim() == 4, "in (B,C,H,W), W (F,C,kH,kW)");
  TORCH_CHECK(in.scalar_type() == at::kFloat && W.scalar_type() == at::kFloat,
              "fp32 only");
  ConvShape s;
  s.B = in.size(0); s.C = in.size(1); s.H = in.size(2); s.W = in.size(3);
  s.F = W.size(0); s.kH = W.size(2); s.kW = W.size(3);
  TORCH_CHECK(W.size(1) == s.C, "weights C must match input C");
  s.oH = (s.H + 2 * padding - s.kH) / stride + 1;
  s.oW = (s.W + 2 * padding - s.kW) / stride + 1;
  TORCH_CHECK(s.oH > 0 && s.oW > 0, "non-positive output size");
  return s;
}

// USE_DONE skips already-spiked output positions (whole F filters done).
template <bool USE_DONE>
void run_single_t(
    int C, int H, int W, int F, int kH, int kW,
    int stride, int padding, int oH, int oW, int num_bins,
    const float* in_b, const float* W_base, const float* th,
    float* st_b, float* pot_b, uint8_t* done_b
) {
  const int oHoW = oH * oW;
  const int CkHkW = C * kH * kW;
  auto uniq = collect_unique_times(in_b, C * H * W, num_bins);
  if (uniq.empty()) return;

  std::vector<int> num_remaining;
  if constexpr (USE_DONE) num_remaining.assign((size_t)oHoW, F);

  for (float t : uniq) {
    for (int c = 0; c < C; ++c) {
      const float* in_c = in_b + c * H * W;
      const float* W_fc_base = W_base + c * kH * kW;
      for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
          if (in_c[y * W + x] != t) continue;
          const int yp = y + padding, xp = x + padding;
          auto [oh_lo, oh_hi] = affected_out_range(yp, kH, stride, oH);
          auto [ow_lo, ow_hi] = affected_out_range(xp, kW, stride, oW);
          for (int oh = oh_lo; oh < oh_hi; ++oh) {
            const int ky = yp - oh * stride;
            for (int ow = ow_lo; ow < ow_hi; ++ow) {
              const int p_off = oh * oW + ow;
              if constexpr (USE_DONE) if (num_remaining[p_off] == 0) continue;
              float* pot_pos = pot_b + p_off;
              const float* W_pos = W_fc_base + ky * kW + (xp - ow * stride);
              for (int f = 0; f < F; ++f) pot_pos[f * oHoW] += W_pos[f * CkHkW];
            }
          }
        }
      }
    }
    for (int f = 0; f < F; ++f) {
      const float th_f = th[f];
      const int f_off = f * oHoW;
      for (int p = 0; p < oHoW; ++p) {
        const int idx = f_off + p;
        if (std::isinf(st_b[idx]) && pot_b[idx] >= th_f) {
          st_b[idx] = t;
          if constexpr (USE_DONE) { done_b[idx] = 1; --num_remaining[p]; }
        }
      }
    }
  }
}

void run_multi_scatter(
    const ConvShape& s, int K, int b, int stride, int padding, int num_bins,
    const float* in_b, const float* W_p, const float* th_p,
    float* pot_b, float* st_p
) {
  auto uniq = collect_unique_times(in_b, s.CHW(), num_bins);
  if (uniq.empty()) return;
  const int oHoW = s.oH * s.oW;
  const int CkHkW = s.C * s.kH * s.kW;
  const int FoHoW = s.FoHoW();

  for (float t : uniq) {
    for (int c = 0; c < s.C; ++c) {
      const float* in_c_slab = in_b + c * s.H * s.W;
      const float* W_fc_base = W_p + c * s.kH * s.kW;
      for (int y = 0; y < s.H; ++y) {
        for (int x = 0; x < s.W; ++x) {
          if (in_c_slab[y * s.W + x] != t) continue;
          const int yp = y + padding, xp = x + padding;
          auto [oh_lo, oh_hi] = affected_out_range(yp, s.kH, stride, s.oH);
          auto [ow_lo, ow_hi] = affected_out_range(xp, s.kW, stride, s.oW);
          for (int oh = oh_lo; oh < oh_hi; ++oh) {
            const int ky = yp - oh * stride;
            for (int ow = ow_lo; ow < ow_hi; ++ow) {
              float* pot_pos = pot_b + oh * s.oW + ow;
              const float* W_pos = W_fc_base + ky * s.kW + (xp - ow * stride);
              for (int f = 0; f < s.F; ++f) pot_pos[f * oHoW] += W_pos[f * CkHkW];
            }
          }
        }
      }
    }
    for (int k = 0; k < K; ++k) {
      float* st_kb = st_p + (size_t)k * s.B * FoHoW + (size_t)b * FoHoW;
      const float* th_k = th_p + k * s.F;
      for (int f = 0; f < s.F; ++f) {
        const float th_f = th_k[f];
        const int f_off = f * oHoW;
        for (int p = 0; p < oHoW; ++p) {
          const int idx = f_off + p;
          if (std::isinf(st_kb[idx]) && pot_b[idx] >= th_f) st_kb[idx] = t;
        }
      }
    }
  }
}

struct SpikeEvent {
  int local_idx;
  int bin;
};

inline int time_to_bin(float t, int num_bins) {
  int bin = (int)(t * num_bins);
  if (bin < 0) return 0;
  if (bin >= num_bins) return num_bins - 1;
  return bin;
}

inline void collect_spike_events(
    const float* in_b, const ConvShape& s, int stride, int padding,
    int num_bins, int oh, int ow, std::vector<SpikeEvent>& events
) {
  events.clear();
  events.reserve((size_t)s.C * s.kH * s.kW);
  if (padding == 0 && s.H == s.kH && s.W == s.kW) {
    const int flat = s.C * s.kH * s.kW;
    for (int i = 0; i < flat; ++i) {
      float t = in_b[i];
      if (std::isfinite(t)) events.push_back({i, time_to_bin(t, num_bins)});
    }
    return;
  }
  for (int c = 0; c < s.C; ++c) {
    for (int ky = 0; ky < s.kH; ++ky) {
      int y = oh * stride + ky - padding;
      for (int kx = 0; kx < s.kW; ++kx) {
        int x = ow * stride + kx - padding;
        if (y < 0 || y >= s.H || x < 0 || x >= s.W) continue;
        float t = in_b[((c * s.H) + y) * s.W + x];
        if (!std::isfinite(t)) continue;
        events.push_back({((c * s.kH) + ky) * s.kW + kx, time_to_bin(t, num_bins)});
      }
    }
  }
}

inline void accumulate_bins(
    const std::vector<SpikeEvent>& events, const float* W_f, std::vector<float>& bins
) {
  std::fill(bins.begin(), bins.end(), 0.0f);
  for (const auto& ev : events) bins[ev.bin] += W_f[ev.local_idx];
}

inline float sum_bins(const std::vector<float>& bins) {
  float total = 0.0f;
  for (float v : bins) total += v;
  return total;
}

inline float scan_first_crossing(const float* bins, int num_bins, float th) {
  float cum = 0.0f;
  for (int bin = 0; bin < num_bins; ++bin) {
    cum += bins[bin];
    if (cum >= th) return (float)bin / (float)num_bins;
  }
  return kInf;
}

inline float scan_first_crossing_with_potential(
    const float* bins, int num_bins, float th, float* pot_out
) {
  float cum = 0.0f, spike_t = kInf;
  for (int bin = 0; bin < num_bins; ++bin) {
    cum += bins[bin];
    if (spike_t == kInf && cum >= th) spike_t = (float)bin / (float)num_bins;
  }
  *pot_out = cum;
  return spike_t;
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> spike_driven_conv_accumulate_cpu(
    at::Tensor input_times, at::Tensor weights_4d, at::Tensor thresholds,
    int64_t stride, int64_t padding, int64_t num_bins, bool compute_cum_potential
) {
  TORCH_CHECK(thresholds.dim() == 1 && thresholds.scalar_type() == at::kFloat,
              "thresholds must be (F,) fp32");
  TORCH_CHECK(input_times.is_cpu() && weights_4d.is_cpu() && thresholds.is_cpu(),
              "CPU kernel: all tensors on CPU");
  auto in_c = input_times.contiguous();
  auto W_c = weights_4d.contiguous();
  auto th_c = thresholds.contiguous();
  auto s = extract_shape(in_c, W_c, stride, padding);
  TORCH_CHECK(th_c.size(0) == s.F, "thresholds size must equal F");

  auto opts = in_c.options();
  auto spike_times = at::full({s.B, s.F, s.oH, s.oW}, kInf, opts);
  auto cum_potential = at::zeros({s.B, s.F, s.oH, s.oW}, opts);
  at::Tensor done = compute_cum_potential
      ? at::Tensor{} : at::zeros({s.B, s.F, s.oH, s.oW}, opts.dtype(at::kByte));

  const float* in_p = in_c.data_ptr<float>();
  const float* W_p = W_c.data_ptr<float>();
  const float* th_p = th_c.data_ptr<float>();
  float* st_p = spike_times.data_ptr<float>();
  float* pot_p = cum_potential.data_ptr<float>();
  uint8_t* done_p = compute_cum_potential ? nullptr : done.data_ptr<uint8_t>();

#pragma omp parallel for if (s.B > 1)
  for (int b = 0; b < s.B; ++b) {
    const float* in_b = in_p + b * s.CHW();
    float* st_b = st_p + b * s.FoHoW();
    float* pot_b = pot_p + b * s.FoHoW();
    if (compute_cum_potential) {
      run_single_t<false>(s.C, s.H, s.W, s.F, s.kH, s.kW, stride, padding,
                          s.oH, s.oW, (int)num_bins,
                          in_b, W_p, th_p, st_b, pot_b, nullptr);
    } else {
      run_single_t<true>(s.C, s.H, s.W, s.F, s.kH, s.kW, stride, padding,
                         s.oH, s.oW, (int)num_bins,
                         in_b, W_p, th_p, st_b, pot_b,
                         done_p + b * s.FoHoW());
    }
  }
  return {spike_times, cum_potential};
}

at::Tensor spike_driven_conv_accumulate_multi_threshold_cpu(
    at::Tensor input_times, at::Tensor weights_4d, at::Tensor thresholds_2d,
    int64_t stride, int64_t padding, int64_t num_bins
) {
  TORCH_CHECK(thresholds_2d.dim() == 2 && thresholds_2d.scalar_type() == at::kFloat,
              "thresholds must be (K,F) fp32");
  TORCH_CHECK(input_times.is_cpu(), "CPU kernel: input on CPU");
  auto in_c = input_times.contiguous();
  auto W_c = weights_4d.contiguous();
  auto th_c = thresholds_2d.contiguous();
  auto s = extract_shape(in_c, W_c, stride, padding);
  const int K = th_c.size(0);
  TORCH_CHECK(th_c.size(1) == s.F, "threshold last dim must equal F");

  auto opts = in_c.options();
  auto spike_times = at::full({K, s.B, s.F, s.oH, s.oW}, kInf, opts);
  auto cum_potential = at::zeros({s.B, s.F, s.oH, s.oW}, opts);

  const float* in_p = in_c.data_ptr<float>();
  const float* W_p = W_c.data_ptr<float>();
  const float* th_p = th_c.data_ptr<float>();
  float* st_p = spike_times.data_ptr<float>();
  float* pot_p = cum_potential.data_ptr<float>();

#pragma omp parallel for if (s.B > 1)
  for (int b = 0; b < s.B; ++b) {
    run_multi_scatter(s, K, b, stride, padding, (int)num_bins,
                      in_p + b * s.CHW(), W_p, th_p,
                      pot_p + b * s.FoHoW(), st_p);
  }
  return spike_times;
}

// Parallelise over outputs so B=1 (online / STDP) still uses all cores.
std::tuple<at::Tensor, at::Tensor> first_spike_times_cpu(
    at::Tensor input_times, at::Tensor weights_4d, at::Tensor thresholds,
    int64_t num_bins, int64_t stride, int64_t padding, bool compute_cum_potential
) {
  TORCH_CHECK(thresholds.dim() == 1 && num_bins > 0, "thresholds (F,), num_bins > 0");
  TORCH_CHECK(input_times.is_cpu(), "CPU kernel: input on CPU");
  auto in_c = input_times.contiguous();
  auto W_c = weights_4d.contiguous();
  auto th_c = thresholds.contiguous();
  auto s = extract_shape(in_c, W_c, stride, padding);
  TORCH_CHECK(th_c.size(0) == s.F, "thresholds size must equal F");

  auto opts = in_c.options();
  auto spike_times = at::full({s.B, s.F, s.oH, s.oW}, kInf, opts);
  auto cum_potential = compute_cum_potential
      ? at::zeros({s.B, s.F, s.oH, s.oW}, opts) : at::empty({0}, opts);

  const float* in_p = in_c.data_ptr<float>();
  const float* W_p = W_c.data_ptr<float>();
  const float* th_p = th_c.data_ptr<float>();
  float* st_p = spike_times.data_ptr<float>();
  float* pot_p = compute_cum_potential ? cum_potential.data_ptr<float>() : nullptr;

  const int nb = (int)num_bins;
  const int oHoW = s.oH * s.oW;
  const int CkHkW = s.C * s.kH * s.kW;
  const size_t num_spatial = (size_t)s.B * oHoW;

  auto write_result = [&](const std::vector<float>& bins, float th_f, size_t idx) {
    if (compute_cum_potential)
      st_p[idx] = scan_first_crossing_with_potential(bins.data(), nb, th_f, pot_p + idx);
    else
      st_p[idx] = scan_first_crossing(bins.data(), nb, th_f);
  };

  if (num_spatial >= kSpatialParallelThreshold) {
#pragma omp parallel
    {
      std::vector<float> bins((size_t)nb);
      std::vector<SpikeEvent> events;
#pragma omp for schedule(static)
      for (size_t spatial_idx = 0; spatial_idx < num_spatial; ++spatial_idx) {
        const int ow = (int)(spatial_idx % s.oW);
        const int oh = (int)((spatial_idx / s.oW) % s.oH);
        const int b = (int)(spatial_idx / oHoW);
        collect_spike_events(in_p + (size_t)b * s.CHW(), s, stride, padding, nb, oh, ow, events);
        for (int f = 0; f < s.F; ++f) {
          accumulate_bins(events, W_p + (size_t)f * CkHkW, bins);
          write_result(bins, th_p[f], ((size_t)b * s.F + f) * oHoW + oh * s.oW + ow);
        }
      }
    }
  } else {
    std::vector<std::vector<SpikeEvent>> events(num_spatial);
    for (size_t spatial_idx = 0; spatial_idx < num_spatial; ++spatial_idx) {
      const int ow = (int)(spatial_idx % s.oW);
      const int oh = (int)((spatial_idx / s.oW) % s.oH);
      const int b = (int)(spatial_idx / oHoW);
      collect_spike_events(in_p + (size_t)b * s.CHW(), s, stride, padding, nb, oh, ow,
                           events[spatial_idx]);
    }
    const size_t total_tasks = num_spatial * s.F;
#pragma omp parallel
    {
      std::vector<float> bins((size_t)nb);
#pragma omp for schedule(static)
      for (size_t task_idx = 0; task_idx < total_tasks; ++task_idx) {
        const int f = (int)(task_idx % s.F);
        const size_t spatial_idx = task_idx / s.F;
        const int ow = (int)(spatial_idx % s.oW);
        const int oh = (int)((spatial_idx / s.oW) % s.oH);
        const int b = (int)(spatial_idx / oHoW);
        accumulate_bins(events[spatial_idx], W_p + (size_t)f * CkHkW, bins);
        write_result(bins, th_p[f], ((size_t)b * s.F + f) * oHoW + oh * s.oW + ow);
      }
    }
  }

  return {spike_times, cum_potential};
}

std::tuple<at::Tensor, at::Tensor> first_spike_times_multi_threshold_cpu(
    at::Tensor input_times, at::Tensor weights_4d, at::Tensor thresholds_2d,
    int64_t num_bins, int64_t stride, int64_t padding, bool compute_cum_potential
) {
  TORCH_CHECK(thresholds_2d.dim() == 2 && num_bins > 0, "(K,F), num_bins > 0");
  TORCH_CHECK(input_times.is_cpu(), "CPU kernel: input on CPU");
  auto in_c = input_times.contiguous();
  auto W_c = weights_4d.contiguous();
  auto th_c = thresholds_2d.contiguous();
  auto s = extract_shape(in_c, W_c, stride, padding);
  const int K = th_c.size(0);
  TORCH_CHECK(th_c.size(1) == s.F, "threshold last dim must equal F");

  auto opts = in_c.options();
  auto spike_times = at::full({K, s.B, s.F, s.oH, s.oW}, kInf, opts);
  auto cum_potential = compute_cum_potential
      ? at::zeros({s.B, s.F, s.oH, s.oW}, opts) : at::empty({0}, opts);

  const float* in_p = in_c.data_ptr<float>();
  const float* W_p = W_c.data_ptr<float>();
  const float* th_p = th_c.data_ptr<float>();
  float* st_p = spike_times.data_ptr<float>();
  float* pot_p = compute_cum_potential ? cum_potential.data_ptr<float>() : nullptr;

  const int nb = (int)num_bins;
  const int oHoW = s.oH * s.oW;
  const int CkHkW = s.C * s.kH * s.kW;
  const size_t per_K = (size_t)s.B * s.FoHoW();
  const size_t num_spatial = (size_t)s.B * oHoW;

  auto write_result = [&](const std::vector<float>& bins, int f, size_t idx) {
    if (compute_cum_potential) pot_p[idx] = sum_bins(bins);
    for (int k = 0; k < K; ++k)
      st_p[(size_t)k * per_K + idx] =
          scan_first_crossing(bins.data(), nb, th_p[(size_t)k * s.F + f]);
  };

  if (num_spatial >= kSpatialParallelThreshold) {
#pragma omp parallel
    {
      std::vector<float> bins((size_t)nb);
      std::vector<SpikeEvent> events;
#pragma omp for schedule(static)
      for (size_t spatial_idx = 0; spatial_idx < num_spatial; ++spatial_idx) {
        const int ow = (int)(spatial_idx % s.oW);
        const int oh = (int)((spatial_idx / s.oW) % s.oH);
        const int b = (int)(spatial_idx / oHoW);
        collect_spike_events(in_p + (size_t)b * s.CHW(), s, stride, padding, nb, oh, ow, events);
        for (int f = 0; f < s.F; ++f) {
          accumulate_bins(events, W_p + (size_t)f * CkHkW, bins);
          write_result(bins, f, ((size_t)b * s.F + f) * oHoW + oh * s.oW + ow);
        }
      }
    }
  } else {
    std::vector<std::vector<SpikeEvent>> events(num_spatial);
    for (size_t spatial_idx = 0; spatial_idx < num_spatial; ++spatial_idx) {
      const int ow = (int)(spatial_idx % s.oW);
      const int oh = (int)((spatial_idx / s.oW) % s.oH);
      const int b = (int)(spatial_idx / oHoW);
      collect_spike_events(in_p + (size_t)b * s.CHW(), s, stride, padding, nb, oh, ow,
                           events[spatial_idx]);
    }
    const size_t total_tasks = num_spatial * s.F;
#pragma omp parallel
    {
      std::vector<float> bins((size_t)nb);
#pragma omp for schedule(static)
      for (size_t task_idx = 0; task_idx < total_tasks; ++task_idx) {
        const int f = (int)(task_idx % s.F);
        const size_t spatial_idx = task_idx / s.F;
        const int ow = (int)(spatial_idx % s.oW);
        const int oh = (int)((spatial_idx / s.oW) % s.oH);
        const int b = (int)(spatial_idx / oHoW);
        accumulate_bins(events[spatial_idx], W_p + (size_t)f * CkHkW, bins);
        write_result(bins, f, ((size_t)b * s.F + f) * oHoW + oh * s.oW + ow);
      }
    }
  }

  return {spike_times, cum_potential};
}

#ifdef WITH_CUDA
std::tuple<at::Tensor, at::Tensor> spike_driven_conv_accumulate_cuda(
    at::Tensor, at::Tensor, at::Tensor, int64_t, int64_t, bool);
at::Tensor spike_driven_conv_accumulate_multi_threshold_cuda(
    at::Tensor, at::Tensor, at::Tensor, int64_t, int64_t);
std::tuple<at::Tensor, at::Tensor> first_spike_times_cuda(
    at::Tensor, at::Tensor, at::Tensor, int64_t, int64_t, int64_t, bool);
std::tuple<at::Tensor, at::Tensor> first_spike_times_multi_threshold_cuda(
    at::Tensor, at::Tensor, at::Tensor, int64_t, int64_t, int64_t, bool);
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "spike-driven conv accumulator";
  m.def("spike_driven_conv_accumulate_cpu", &spike_driven_conv_accumulate_cpu,
        "Sparse-event spike-driven conv accumulate (CPU).",
        py::arg("input_times"), py::arg("weights_4d"), py::arg("thresholds"),
        py::arg("stride") = 1, py::arg("padding") = 0,
        py::arg("num_bins") = 64,
        py::arg("compute_cum_potential") = true);
  m.def("spike_driven_conv_accumulate_multi_threshold_cpu",
        &spike_driven_conv_accumulate_multi_threshold_cpu,
        "Multi-threshold variant (CPU).",
        py::arg("input_times"), py::arg("weights_4d"), py::arg("thresholds_2d"),
        py::arg("stride") = 1, py::arg("padding") = 0,
        py::arg("num_bins") = 64);
  m.def("first_spike_times_cpu", &first_spike_times_cpu,
        "Gather-style first-spike times (CPU).",
        py::arg("input_times"), py::arg("weights_4d"), py::arg("thresholds"),
        py::arg("num_bins"), py::arg("stride") = 1, py::arg("padding") = 0,
        py::arg("compute_cum_potential") = false);
  m.def("first_spike_times_multi_threshold_cpu",
        &first_spike_times_multi_threshold_cpu,
        "Gather-style first-spike times multi-threshold (CPU).",
        py::arg("input_times"), py::arg("weights_4d"), py::arg("thresholds_2d"),
        py::arg("num_bins"), py::arg("stride") = 1, py::arg("padding") = 0,
        py::arg("compute_cum_potential") = false);
#ifdef WITH_CUDA
  m.def("spike_driven_conv_accumulate_cuda", &spike_driven_conv_accumulate_cuda,
        "Sparse-event spike-driven conv accumulate (CUDA).",
        py::arg("input_times"), py::arg("weights_4d"), py::arg("thresholds"),
        py::arg("stride") = 1, py::arg("padding") = 0,
        py::arg("compute_cum_potential") = true);
  m.def("spike_driven_conv_accumulate_multi_threshold_cuda",
        &spike_driven_conv_accumulate_multi_threshold_cuda,
        "Multi-threshold variant (CUDA).",
        py::arg("input_times"), py::arg("weights_4d"), py::arg("thresholds_2d"),
        py::arg("stride") = 1, py::arg("padding") = 0);
  m.def("first_spike_times_cuda", &first_spike_times_cuda,
        "Gather-style first-spike times (CUDA).",
        py::arg("input_times"), py::arg("weights_4d"), py::arg("thresholds"),
        py::arg("num_bins"), py::arg("stride") = 1, py::arg("padding") = 0,
        py::arg("compute_cum_potential") = false);
  m.def("first_spike_times_multi_threshold_cuda",
        &first_spike_times_multi_threshold_cuda,
        "Gather-style first-spike times multi-threshold (CUDA).",
        py::arg("input_times"), py::arg("weights_4d"), py::arg("thresholds_2d"),
        py::arg("num_bins"), py::arg("stride") = 1, py::arg("padding") = 0,
        py::arg("compute_cum_potential") = false);
#endif
}

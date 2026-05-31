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
std::tuple<at::Tensor, at::Tensor> first_spike_times_cuda(
    at::Tensor, at::Tensor, at::Tensor, int64_t, int64_t, int64_t, bool);
std::tuple<at::Tensor, at::Tensor> first_spike_times_multi_threshold_cuda(
    at::Tensor, at::Tensor, at::Tensor, int64_t, int64_t, int64_t, bool);
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "gather-style first-spike-time conv kernels";
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

include "stride/moe.hpp"
#include "stride/ops.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace stride {

MoERouter::MoERouter(std::int64_t d_model, std::int64_t n_experts_, std::int64_t top_k_)
    : gate_proj(Tensor({n_experts_, d_model})), n_experts(n_experts_), top_k(top_k_) {
    if (top_k_ <= 0 || n_experts_ <= 0 || top_k_ > n_experts_) {
        throw StatusError("invalid MoE router configuration");
    }
}

MoERouter::RoutingResult MoERouter::route(const Tensor& x) const {
    if (x.ndim() != 2) throw StatusError("router expects [seq, d_model]");
    const std::int64_t seq = x.dim(0);
    Tensor logits = gate_proj.forward(x);
    Tensor weights_out({seq, top_k});
    Tensor indices_out({seq, top_k});
    std::vector<std::int64_t> idx(static_cast<std::size_t>(n_experts));
    for (std::int64_t s = 0; s < seq; ++s) {
        const float* row = logits.data() + s * n_experts;
        std::iota(idx.begin(), idx.end(), 0);
        std::partial_sort(idx.begin(), idx.begin() + top_k, idx.end(),
                          [&](std::int64_t a, std::int64_t b) { return row[a] > row[b]; });
        float mx = row[idx[0]];
        float sum = 0.0f;
        std::vector<float> sel(static_cast<std::size_t>(top_k));
        for (std::int64_t k = 0; k < top_k; ++k) {
            sel[static_cast<std::size_t>(k)] = std::exp(row[idx[static_cast<std::size_t>(k)]] - mx);
            sum += sel[static_cast<std::size_t>(k)];
        }
        for (std::int64_t k = 0; k < top_k; ++k) {
            weights_out.at(s, k) = sel[static_cast<std::size_t>(k)] / sum;
            indices_out.at(s, k) = static_cast<float>(idx[static_cast<std::size_t>(k)]);
        }
    }
    return {weights_out, indices_out};
}

ExpertMLP::ExpertMLP(std::int64_t d_model, std::int64_t d_ff)
    : gate_proj(Tensor({d_ff, d_model})),
      up_proj(Tensor({d_ff, d_model})),
      down_proj(Tensor({d_model, d_ff})),
      loaded(true) {}

Tensor ExpertMLP::forward(const Tensor& x) const {
    if (!loaded) throw StatusError("expert weights are not loaded");
    return down_proj.forward(ops::mul(ops::silu(gate_proj.forward(x)), up_proj.forward(x)));
}

SparseMoELayer::SparseMoELayer(std::int64_t d_model, std::int64_t d_ff, std::int64_t n_experts,
                               std::int64_t top_k)
    : router(d_model, n_experts, top_k) {
    experts.reserve(static_cast<std::size_t>(n_experts));
    for (std::int64_t e = 0; e < n_experts; ++e) experts.emplace_back(d_model, d_ff);
}

Tensor SparseMoELayer::forward(const Tensor& x) const {
    if (x.ndim() != 2) throw StatusError("MoE expects [seq, d_model]");
    const std::int64_t seq = x.dim(0);
    const std::int64_t d_model = x.dim(1);
    auto routing = router.route(x);
    Tensor out(x.shape());
    for (std::int64_t s = 0; s < seq; ++s) {
        Tensor tok = Tensor::from_f32(
            {1, d_model}, std::vector<float>(x.data() + s * d_model, x.data() + (s + 1) * d_model));
        for (std::int64_t k = 0; k < router.top_k; ++k) {
            const auto eid = static_cast<std::int64_t>(routing.expert_indices.at(s, k));
            const float w = routing.weights.at(s, k);
            if (eid < 0 || eid >= static_cast<std::int64_t>(experts.size())) {
                throw StatusError("invalid expert index");
            }
            ExpertMLP expert = experts[static_cast<std::size_t>(eid)];
            if (!expert.loaded) {
                if (!loader) throw StatusError("expert " + std::to_string(eid) + " is unloaded and no loader is set");
                expert = loader(eid);
            }
            Tensor y = expert.forward(tok);
            for (std::int64_t d = 0; d < d_model; ++d) {
                out.data()[s * d_model + d] += w * y.data()[d];
            }
        }

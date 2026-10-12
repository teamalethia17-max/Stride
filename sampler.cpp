#include "stride/sampler.hpp"
#include "stride/ops.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace stride {

TokenId sample_token(const Tensor& logits, const SamplingParams& params, std::uint64_t& rng_state) {
    const std::int64_t v = logits.numel();
    const float* lp = logits.data();
    if (params.temperature <= 0.0f) {
        return static_cast<TokenId>(ops::argmax(logits));
    }
    std::vector<float> probs(static_cast<std::size_t>(v));
    const float inv_t = 1.0f / params.temperature;
    const float mx = *std::max_element(lp, lp + v);
    float sum = 0.0f;
    for (std::int64_t i = 0; i < v; ++i) {
        probs[static_cast<std::size_t>(i)] = std::exp((lp[i] - mx) * inv_t);
        sum += probs[static_cast<std::size_t>(i)];
    }
    std::vector<std::int64_t> indices(static_cast<std::size_t>(v));
    std::iota(indices.begin(), indices.end(), 0);
    if (params.top_k > 0 && params.top_k < v) {
        std::partial_sort(indices.begin(), indices.begin() + params.top_k, indices.end(),
                          [&](std::int64_t a, std::int64_t b) { return probs[a] > probs[b]; });
        for (std::int64_t i = params.top_k; i < v; ++i) probs[static_cast<std::size_t>(indices[i])] = 0.0f;
    }
    if (params.top_p < 1.0f) {
        std::sort(indices.begin(), indices.end(),
                  [&](std::int64_t a, std::int64_t b) { return probs[a] > probs[b]; });
        float cum = 0.0f;
        bool cutoff = false;
        for (std::int64_t i = 0; i < v; ++i) {
            cum += probs[static_cast<std::size_t>(indices[i])] / sum;
            if (cutoff) probs[static_cast<std::size_t>(indices[i])] = 0.0f;
            if (cum > params.top_p) cutoff = true;
        }
    }
    sum = std::accumulate(probs.begin(), probs.end(), 0.0f);
    if (sum <= 0.0f) return static_cast<TokenId>(ops::argmax(logits));

    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    const float r = (static_cast<float>(rng_state) / static_cast<float>(UINT64_MAX)) * sum;
    float cum = 0.0f;
    for (std::int64_t i = 0; i < v; ++i) {
        cum += probs[static_cast<std::size_t>(i)];
        if (cum >= r) return static_cast<TokenId>(i);
    }
    return static_cast<TokenId>(v - 1);
}

}  // namespace stride

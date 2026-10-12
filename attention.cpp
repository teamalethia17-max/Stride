#include "stride/attention.hpp"
#include "stride/linear.hpp"
#include "stride/ops.hpp"

#include <algorithm>
#include <cmath>

namespace stride {

void apply_rope(Tensor& x, std::int64_t position_offset, float theta) {
    if (x.ndim() != 3) throw StatusError("apply_rope expects [seq, heads, head_dim]");
    const std::int64_t seq = x.dim(0);
    const std::int64_t n_heads = x.dim(1);
    const std::int64_t head_dim = x.dim(2);
    if (head_dim % 2 != 0) throw StatusError("RoPE head_dim must be even");
    float* xp = x.data();
    for (std::int64_t pos = 0; pos < seq; ++pos) {
        const float fpos = static_cast<float>(pos + position_offset);
        for (std::int64_t h = 0; h < n_heads; ++h) {
            float* row = xp + pos * n_heads * head_dim + h * head_dim;
            for (std::int64_t i = 0; i < head_dim / 2; ++i) {
                const float freq =
                    fpos / std::pow(theta, 2.0f * static_cast<float>(i) / static_cast<float>(head_dim));
                const float c = std::cos(freq);
                const float s = std::sin(freq);
                const float x0 = row[2 * i];
                const float x1 = row[2 * i + 1];
                row[2 * i] = x0 * c - x1 * s;
                row[2 * i + 1] = x0 * s + x1 * c;
            }
        }
    }
}

Tensor scaled_dot_product_attention(const Tensor& q, const Tensor& k, const Tensor& v, bool causal,
                                    std::int64_t causal_offset) {
    if (q.ndim() != 3 || k.ndim() != 3 || v.ndim() != 3) {
        throw StatusError("SDPA requires rank-3 q/k/v");
    }
    const std::int64_t seq_q = q.dim(0);
    const std::int64_t n_heads = q.dim(1);
    const std::int64_t head_dim = q.dim(2);
    const std::int64_t seq_k = k.dim(0);
    const std::int64_t n_kv_heads = k.dim(1);
    if (k.dim(2) != head_dim || v.dim(0) != seq_k || v.dim(1) != n_kv_heads || v.dim(2) != head_dim) {
        throw StatusError("SDPA k/v shape mismatch");
    }
    if (n_heads % n_kv_heads != 0) throw StatusError("n_heads must be divisible by n_kv_heads");
    const std::int64_t group = n_heads / n_kv_heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    Tensor out({seq_q, n_heads, head_dim});
    for (std::int64_t hq = 0; hq < n_heads; ++hq) {
        const std::int64_t hkv = hq / group;
        Tensor scores({seq_q, seq_k});
        for (std::int64_t iq = 0; iq < seq_q; ++iq) {
            const float* qrow = q.data() + iq * n_heads * head_dim + hq * head_dim;
            for (std::int64_t ik = 0; ik < seq_k; ++ik) {
                if (causal && (ik > iq + causal_offset)) {
                    scores.at(iq, ik) = -1e9f;
                    continue;
                }
                const float* krow = k.data() + ik * n_kv_heads * head_dim + hkv * head_dim;
                float dot = 0.0f;
                for (std::int64_t d = 0; d < head_dim; ++d) dot += qrow[d] * krow[d];
                scores.at(iq, ik) = dot * scale;
            }
        }
        Tensor probs = ops::softmax(scores);
        for (std::int64_t iq = 0; iq < seq_q; ++iq) {
            float* orow = out.data() + iq * n_heads * head_dim + hq * head_dim;
            std::fill(orow, orow + head_dim, 0.0f);
            for (std::int64_t ik = 0; ik < seq_k; ++ik) {
                const float p = probs.at(iq, ik);
                const float* vrow = v.data() + ik * n_kv_heads * head_dim + hkv * head_dim;
                for (std::int64_t d = 0; d < head_dim; ++d) orow[d] += p * vrow[d];
            }
        }
    }
    return out;
}

MultiHeadAttention::MultiHeadAttention(std::int64_t d_model, std::int64_t n_heads_,
                                       std::int64_t n_kv_heads_, float rope_theta_)
    : n_heads(n_heads_), n_kv_heads(n_kv_heads_), rope_theta(rope_theta_) {
    if (n_heads_ <= 0 || n_kv_heads_ <= 0 || d_model % n_heads_ != 0) {
        throw StatusError("invalid attention dimensions");
    }
    head_dim = d_model / n_heads_;
    wq = Tensor({n_heads * head_dim, d_model});
    wk = Tensor({n_kv_heads * head_dim, d_model});
    wv = Tensor({n_kv_heads * head_dim, d_model});
    wo = Tensor({d_model, n_heads * head_dim});
}

static Tensor project_heads(const Tensor& x, const Tensor& w, std::int64_t n_heads,
                            std::int64_t head_dim) {
    Tensor y = Linear(w).forward(x);  // [seq, n_heads*head_dim]
    const std::int64_t seq = x.dim(0);
    Tensor shaped({seq, n_heads, head_dim});
    const float* src = y.data();
    float* dst = shaped.data();
    for (std::int64_t s = 0; s < seq; ++s) {
        for (std::int64_t h = 0; h < n_heads; ++h) {
            std::copy(src + s * n_heads * head_dim + h * head_dim,
                      src + s * n_heads * head_dim + (h + 1) * head_dim,
                      dst + s * n_heads * head_dim + h * head_dim);
        }
    }
    return shaped;
}

Tensor MultiHeadAttention::forward(const Tensor& x, std::int64_t position_offset) const {
    if (x.ndim() != 2) throw StatusError("attention expects [seq, d_model]");
    Tensor q = project_heads(x, wq, n_heads, head_dim);
    Tensor k = project_heads(x, wk, n_kv_heads, head_dim);
    Tensor v = project_heads(x, wv, n_kv_heads, head_dim);
    apply_rope(q, position_offset, rope_theta);
    apply_rope(k, position_offset, rope_theta);
    Tensor ctx = scaled_dot_product_attention(q, k, v, true, 0);
    const std::int64_t seq = x.dim(0);
    Tensor flat = ctx.view({seq, n_heads * head_dim});
    return Linear(wo).forward(flat);
}

Tensor MultiHeadAttention::forward_cached(const Tensor& x, std::int64_t layer_idx,
                                          KVCache& cache) const {
    if (x.ndim() != 2 || x.dim(0) != 1) throw StatusError("cached attention expects [1, d_model]");
    if (layer_idx < 0 || layer_idx >= cache.n_layers) throw StatusError("layer_idx out of range");
    if (cache.filled >= cache.max_seq_len) throw StatusError("KV cache capacity exceeded");

    Tensor q = project_heads(x, wq, n_heads, head_dim);
    Tensor k = project_heads(x, wk, n_kv_heads, head_dim);
    Tensor v = project_heads(x, wv, n_kv_heads, head_dim);
    apply_rope(q, cache.filled, rope_theta);
    apply_rope(k, cache.filled, rope_theta);

    const std::int64_t pos = cache.filled;
    std::copy(k.data(), k.data() + n_kv_heads * head_dim,
              cache.k_cache[static_cast<std::size_t>(layer_idx)].data() +
                  pos * n_kv_heads * head_dim);
    std::copy(v.data(), v.data() + n_kv_heads * head_dim,
              cache.v_cache[static_cast<std::size_t>(layer_idx)].data() +
                  pos * n_kv_heads * head_dim);

    Tensor k_used = cache.k_cache[static_cast<std::size_t>(layer_idx)].view(
        {cache.max_seq_len, n_kv_heads, head_dim});
    Tensor v_used = cache.v_cache[static_cast<std::size_t>(layer_idx)].view(
        {cache.max_seq_len, n_kv_heads, head_dim});

    Tensor k_prefix({pos + 1, n_kv_heads, head_dim});
    Tensor v_prefix({pos + 1, n_kv_heads, head_dim});
    std::copy(k_used.data(), k_used.data() + (pos + 1) * n_kv_heads * head_dim, k_prefix.data());
    std::copy(v_used.data(), v_used.data() + (pos + 1) * n_kv_heads * head_dim, v_prefix.data());

    Tensor ctx = scaled_dot_product_attention(q, k_prefix, v_prefix, true, pos);
    Tensor flat = ctx.view({1, n_heads * head_dim});
    return Linear(wo).forward(flat);
}

}  // namespace stride

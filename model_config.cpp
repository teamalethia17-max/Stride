#include "stride/model_config.hpp"

#include "stride/dtype.hpp"

#include <sstream>

namespace stride {

std::string ModelConfig::validate() {
    if (vocab_size <= 0) return "vocab_size must be positive";
    if (d_model <= 0) return "d_model must be positive";
    if (n_layers <= 0) return "n_layers must be positive";
    if (n_heads <= 0) return "n_heads must be positive";
    if (n_kv_heads <= 0) n_kv_heads = n_heads;
    if (n_heads % n_kv_heads != 0) return "n_heads must be divisible by n_kv_heads";
    if (head_dim <= 0) {
        if (d_model % n_heads != 0) return "d_model must be divisible by n_heads";
        head_dim = d_model / n_heads;
    }
    if (head_dim % 2 != 0) return "head_dim must be even for RoPE";
    if (d_ff <= 0) return "d_ff must be positive";
    if (max_seq_len <= 0) return "max_seq_len must be positive";
    if (ffn_type == FFNType::SparseMoE) {
        if (n_experts <= 0 || top_k_experts <= 0) return "MoE requires n_experts and top_k_experts";
        if (top_k_experts > n_experts) return "top_k_experts > n_experts";
    }
    return {};
}

std::int64_t ModelConfig::expected_weight_bytes_f32() const {
    std::int64_t bytes = 0;
    auto add = [&](std::int64_t n) { bytes += n * static_cast<std::int64_t>(sizeof(float)); };
    add(vocab_size * d_model);  // embeddings
    if (!tie_embeddings) add(vocab_size * d_model);
    add(d_model);  // final norm
    for (std::int64_t i = 0; i < n_layers; ++i) {
        add(d_model);                              // attn norm
        add(n_heads * head_dim * d_model);         // q
        add(n_kv_heads * head_dim * d_model);      // k
        add(n_kv_heads * head_dim * d_model);      // v
        add(d_model * n_heads * head_dim);         // o
        add(d_model);                              // ffn norm
        if (ffn_type == FFNType::Dense) {
            add(d_ff * d_model * 2);  // gate+up
            add(d_model * d_ff);      // down
        } else {
            add(n_experts * d_model);          // router
            add(n_experts * d_ff * d_model * 2);
            add(n_experts * d_model * d_ff);
        }
    }
    return bytes;
}

}  // namespace stride

#include "stride/model_loader.hpp"

#include "gguf.hpp"
#include "json.hpp"
#include "stride/test_model.hpp"

#include <fstream>
#include <iostream>
#include <map>
#include <sstream>

namespace stride {
namespace {

std::string read_text(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw StatusError("cannot open " + path.string());
    std::ostringstream oss;
    oss << in.rdbuf();
    return oss.str();
}

void write_text(const std::filesystem::path& path, const std::string& text) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw StatusError("cannot write " + path.string());
    out << text;
}

json::Value jstr(const std::string& s) {
    json::Value v;
    v.data = s;
    return v;
}
json::Value jnum(double n) {
    json::Value v;
    v.data = n;
    return v;
}
json::Value jbool(bool b) {
    json::Value v;
    v.data = b;
    return v;
}

ModelConfig config_from_json(const json::Value& v) {
    if (!v.is_object()) throw StatusError("config.json must be an object");
    ModelConfig cfg;
    auto get_s = [&](const char* k, const std::string& d = {}) {
        auto* f = v.field(k);
        return (f && f->is_string()) ? f->as_string() : d;
    };
    auto get_i = [&](const char* k, std::int64_t d = 0) {
        auto* f = v.field(k);
        return (f && f->is_number()) ? f->as_int() : d;
    };
    auto get_f = [&](const char* k, float d) {
        auto* f = v.field(k);
        return (f && f->is_number()) ? static_cast<float>(f->as_number()) : d;
    };
    cfg.architecture = get_s("architecture", "llama");
    cfg.model_name = get_s("model_name");
    cfg.vocab_size = get_i("vocab_size");
    cfg.d_model = get_i("d_model");
    cfg.n_layers = get_i("n_layers");
    cfg.n_heads = get_i("n_heads");
    cfg.n_kv_heads = get_i("n_kv_heads", cfg.n_heads);
    cfg.head_dim = get_i("head_dim");
    cfg.d_ff = get_i("d_ff");
    cfg.max_seq_len = get_i("max_seq_len", 2048);
    cfg.norm_eps = get_f("norm_eps", 1e-5f);
    cfg.rope_theta = get_f("rope_theta", 10000.0f);
    cfg.n_experts = get_i("n_experts");
    cfg.top_k_experts = get_i("top_k_experts");
    cfg.tie_embeddings = v.field("tie_embeddings") && v.field("tie_embeddings")->is_bool() &&
                         v.field("tie_embeddings")->as_bool();
    const std::string ffn = get_s("ffn_type", "dense");
    cfg.ffn_type = (ffn == "moe") ? FFNType::SparseMoE : FFNType::Dense;
    auto err = cfg.validate();
    if (!err.empty()) throw StatusError(err);
    return cfg;
}

json::Value config_to_json(const ModelConfig& cfg) {
    json::Object o;
    o["architecture"] = jstr(cfg.architecture);
    o["model_name"] = jstr(cfg.model_name);
    o["vocab_size"] = jnum(static_cast<double>(cfg.vocab_size));
    o["d_model"] = jnum(static_cast<double>(cfg.d_model));
    o["n_layers"] = jnum(static_cast<double>(cfg.n_layers));
    o["n_heads"] = jnum(static_cast<double>(cfg.n_heads));
    o["n_kv_heads"] = jnum(static_cast<double>(cfg.n_kv_heads));
    o["head_dim"] = jnum(static_cast<double>(cfg.head_dim));
    o["d_ff"] = jnum(static_cast<double>(cfg.d_ff));
    o["max_seq_len"] = jnum(static_cast<double>(cfg.max_seq_len));
    o["norm_eps"] = jnum(cfg.norm_eps);
    o["rope_theta"] = jnum(cfg.rope_theta);
    o["ffn_type"] = jstr(cfg.ffn_type == FFNType::SparseMoE ? "moe" : "dense");
    o["n_experts"] = jnum(static_cast<double>(cfg.n_experts));
    o["top_k_experts"] = jnum(static_cast<double>(cfg.top_k_experts));
    o["tie_embeddings"] = jbool(cfg.tie_embeddings);
    json::Value v;
    v.data = std::move(o);
    return v;
}

BpeTokenizer tokenizer_from_json(const json::Value& v) {
    if (!v.is_object()) throw StatusError("tokenizer.json must be an object");
    auto* vocab_v = v.field("vocab");
    if (!vocab_v || !vocab_v->is_array()) throw StatusError("tokenizer.json missing vocab array");
    std::vector<std::string> vocab;
    for (const auto& item : vocab_v->as_array()) {
        if (!item.is_string()) throw StatusError("vocab entries must be strings");
        vocab.push_back(item.as_string());
    }
    std::vector<std::string> merges;
    if (auto* m = v.field("merges"); m && m->is_array()) {
        for (const auto& item : m->as_array()) {
            if (!item.is_string()) throw StatusError("merges must be strings");
            merges.push_back(item.as_string());
        }
    }
    TokenId bos = v.field("bos_id") ? static_cast<TokenId>(v.field("bos_id")->as_int()) : 1;
    TokenId eos = v.field("eos_id") ? static_cast<TokenId>(v.field("eos_id")->as_int()) : 2;
    TokenId unk = v.field("unk_id") ? static_cast<TokenId>(v.field("unk_id")->as_int()) : 0;
    return BpeTokenizer::from_vocab_and_merges(std::move(vocab), merges, bos, eos, unk).expect("tokenizer");
}

json::Value tokenizer_to_json(const BpeTokenizer& tok) {
    json::Array vocab;
    for (TokenId i = 0; i < tok.vocab_size(); ++i) {
        json::Value s;
        s.data = tok.id_to_piece(i);
        vocab.push_back(s);
    }
    json::Object o;
    o["type"] = jstr("bpe");
    o["bos_id"] = jnum(tok.bos_id());
    o["eos_id"] = jnum(tok.eos_id());
    o["unk_id"] = jnum(tok.unk_id());
    json::Value vocab_v;
    vocab_v.data = std::move(vocab);
    o["vocab"] = vocab_v;
    json::Array merges;
    for (const auto& m : tok.merges()) {
        json::Value s;
        s.data = m;
        merges.push_back(s);
    }
    json::Value merges_v;
    merges_v.data = std::move(merges);
    o["merges"] = merges_v;
    json::Value v;
    v.data = std::move(o);
    return v;
}

std::map<std::string, Tensor> collect_named_weights(const DecoderModel& model) {
    std::map<std::string, Tensor> w;
    w["token_embd.weight"] = model.token_embedding;
    w["output_norm.weight"] = model.final_norm_w;
    w["output.weight"] = model.lm_head;
    for (std::size_t i = 0; i < model.layers.size(); ++i) {
        const auto& blk = model.layers[i];
        const std::string p = "blk." + std::to_string(i) + ".";
        w[p + "attn_norm.weight"] = blk.attn_norm_w;
        w[p + "attn_q.weight"] = blk.attn.wq;
        w[p + "attn_k.weight"] = blk.attn.wk;
        w[p + "attn_v.weight"] = blk.attn.wv;
        w[p + "attn_output.weight"] = blk.attn.wo;
        w[p + "ffn_norm.weight"] = blk.ffn_norm_w;
        if (std::holds_alternative<GatedMLP>(blk.ffn)) {
            const auto& mlp = std::get<GatedMLP>(blk.ffn);
            w[p + "ffn_gate.weight"] = mlp.gate_proj.weight();
            w[p + "ffn_up.weight"] = mlp.up_proj.weight();
            w[p + "ffn_down.weight"] = mlp.down_proj.weight();
        }
    }
    return w;
}

void apply_named_weights(DecoderModel& model, WeightStore& store) {
    auto take = [&](const std::string& name) {
        auto t = store.load(name);
        if (!t) throw StatusError(t.status());
        return t.take();
    };
    model.token_embedding = take("token_embd.weight");
    model.final_norm_w = take("output_norm.weight");
    if (store.contains("output.weight")) {
        model.lm_head = take("output.weight");
    } else {
        model.lm_head = model.token_embedding;
        model.config.tie_embeddings = true;
    }
    for (std::int64_t i = 0; i < model.config.n_layers; ++i) {
        auto& blk = model.layers[static_cast<std::size_t>(i)];
        const std::string p = "blk." + std::to_string(i) + ".";
        blk.attn_norm_w = take(p + "attn_norm.weight");
        blk.attn.wq = take(p + "attn_q.weight");
        blk.attn.wk = take(p + "attn_k.weight");
        blk.attn.wv = take(p + "attn_v.weight");
        blk.attn.wo = take(p + "attn_output.weight");
        blk.ffn_norm_w = take(p + "ffn_norm.weight");
        GatedMLP mlp;
        mlp.gate_proj = Linear(take(p + "ffn_gate.weight"));
        mlp.up_proj = Linear(take(p + "ffn_up.weight"));
        mlp.down_proj = Linear(take(p + "ffn_down.weight"));
        blk.ffn = std::move(mlp);
    }
}

Result<LoadedModel> load_stride_dir(const std::filesystem::path& dir, std::size_t budget) {
    const auto cfg_path = dir / "config.json";
    const auto tok_path = dir / "tokenizer.json";
    const auto idx_path = dir / "tensors.json";
    const auto w_path = dir / "weights.bin";
    if (!std::filesystem::exists(cfg_path) || !std::filesystem::exists(w_path)) {
        return Status::error("not a Stride model directory: missing config.json or weights.bin");
    }
    auto cfg_json = json::parse(read_text(cfg_path));
    if (!cfg_json) return cfg_json.status();
    ModelConfig cfg = config_from_json(cfg_json.value());
    cfg.format = "stride";
    auto tok_json = json::parse(read_text(tok_path));
    if (!tok_json) return tok_json.status();

    auto store_r = WeightStore::open_file(w_path, budget);
    if (!store_r) return store_r.status();
    WeightStore store = store_r.take();
    auto idx_json = json::parse(read_text(idx_path));
    if (!idx_json) return idx_json.status();
    if (!idx_json.value().is_array()) return Status::error("tensors.json must be an array");
    for (const auto& item : idx_json.value().as_array()) {
        TensorLocation loc;
        loc.name = item.field("name")->as_string();
        loc.dtype = DType::F32;
        loc.offset = static_cast<std::uint64_t>(item.field("offset")->as_int());
        loc.nbytes = static_cast<std::uint64_t>(item.field("nbytes")->as_int());
        for (const auto& d : item.field("shape")->as_array()) loc.shape.push_back(d.as_int());
        store.add_location(std::move(loc));
    }

    LoadedModel loaded;
    loaded.model = DecoderModel(cfg);
    loaded.tokenizer = std::make_unique<BpeTokenizer>(tokenizer_from_json(tok_json.value()));
    apply_named_weights(loaded.model, store);
    loaded.weights = std::move(store);
    loaded.format = "stride";
    loaded.path = dir;
    return loaded;
}

Result<LoadedModel> load_gguf(const std::filesystem::path& path, std::size_t budget) {
    auto index = read_gguf_index(path);
    if (!index) return index.status();
    auto cfg = llama_config_from_gguf(index.value());
    if (!cfg) return cfg.status();
    auto tok = tokenizer_from_gguf(index.value());
    if (!tok) return tok.status();
    auto store_r = WeightStore::open_file(path, budget);
    if (!store_r) return store_r.status();
    WeightStore store = store_r.take();
    for (auto loc : index.value().tensors) store.add_location(std::move(loc));
    LoadedModel loaded;
    loaded.model = DecoderModel(cfg.take());
    loaded.tokenizer = std::make_unique<BpeTokenizer>(tok.take());
    apply_named_weights(loaded.model, store);
    loaded.weights = std::move(store);
    loaded.format = "gguf";
    loaded.path = path;
    return loaded;
}

}  // namespace

Result<LoadedModel> load_model(const std::filesystem::path& path, std::size_t memory_budget_bytes) {
    std::error_code ec;
    if (!std::filesystem::exists(path, ec)) {
        return Status::error("model path not found: " + path.string());
    }
    if (std::filesystem::is_directory(path)) {
        return load_stride_dir(path, memory_budget_bytes);
    }
    const auto ext = path.extension().string();
    if (ext == ".gguf" || ext == ".GGUF") {
        return load_gguf(path, memory_budget_bytes);
    }
    return Status::error("unsupported model path '" + path.string() +
                         "'. Provide a Stride model directory (config.json + weights.bin) or a GGUF file. "
                         "Only llama F32/F16 GGUF tensors are implemented.");
}

Status inspect_model(const std::filesystem::path& path) {
    auto loaded = load_model(path);
    if (!loaded) return loaded.status();
    const auto& cfg = loaded.value().model.config;
    std::cout << "path: " << path.string() << "\n";
    std::cout << "format: " << loaded.value().format << "\n";
    std::cout << "architecture: " << cfg.architecture << "\n";
    std::cout << "name: " << cfg.model_name << "\n";
    std::cout << "vocab_size: " << cfg.vocab_size << "\n";
    std::cout << "d_model: " << cfg.d_model << "\n";
    std::cout << "n_layers: " << cfg.n_layers << "\n";
    std::cout << "n_heads: " << cfg.n_heads << "\n";
    std::cout << "n_kv_heads: " << cfg.n_kv_heads << "\n";
    std::cout << "head_dim: " << cfg.head_dim << "\n";
    std::cout << "d_ff: " << cfg.d_ff << "\n";
    std::cout << "max_seq_len: " << cfg.max_seq_len << "\n";
    std::cout << "expected_f32_weight_bytes: " << cfg.expected_weight_bytes_f32() << "\n";
    std::cout << "file_bytes: " << loaded.value().weights.file_size() << "\n";
    std::cout << "bytes_read: " << loaded.value().weights.stats().bytes_read << "\n";
    std::cout << "peak_resident_bytes: " << loaded.value().weights.stats().peak_resident_bytes << "\n";
    std::cout << "kv_cache_bytes_at_max_seq: "
              << static_cast<std::uint64_t>(cfg.n_layers) * 2 * cfg.max_seq_len * cfg.n_kv_heads *
                     cfg.head_dim * sizeof(float)
              << "\n";
    return Status::ok();
}

Status write_test_model(const std::filesystem::path& directory, bool moe) {
    std::error_code ec;
    std::filesystem::create_directories(directory, ec);
    if (ec) return Status::error(ec.message());
    auto bundle = moe ? make_tiny_moe_model() : make_tiny_dense_model();
    write_text(directory / "config.json", json::dump(config_to_json(bundle.model.config)));
    write_text(directory / "tokenizer.json", json::dump(tokenizer_to_json(bundle.tokenizer)));

    auto weights = collect_named_weights(bundle.model);
    json::Array index;
    std::vector<std::uint8_t> blob;
    for (const auto& [name, t] : weights) {
        json::Object o;
        o["name"] = jstr(name);
        o["dtype"] = jstr("f32");
        o["offset"] = jnum(static_cast<double>(blob.size()));
        o["nbytes"] = jnum(static_cast<double>(t.nbytes()));
        json::Array shape;
        for (auto d : t.shape()) {
            json::Value dv;
            dv.data = static_cast<double>(d);
            shape.push_back(dv);
        }
        json::Value sv;
        sv.data = std::move(shape);
        o["shape"] = sv;
        json::Value iv;
        iv.data = std::move(o);
        index.push_back(iv);
        const auto* p = reinterpret_cast<const std::uint8_t*>(t.data());
        blob.insert(blob.end(), p, p + t.nbytes());
    }
    json::Value idx;
    idx.data = std::move(index);
    write_text(directory / "tensors.json", json::dump(idx));
    std::ofstream bin(directory / "weights.bin", std::ios::binary);
    bin.write(reinterpret_cast<const char*>(blob.data()), static_cast<std::streamsize>(blob.size()));
    if (!bin) return Status::error("failed writing weights.bin");

    std::map<std::string, Tensor> gguf_tensors(weights.begin(), weights.end());
    return write_gguf(directory / "model.gguf", bundle.model.config, bundle.tokenizer, gguf_tensors);
}

}  // namespace stride

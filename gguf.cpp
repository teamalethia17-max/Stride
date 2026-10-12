#include "gguf.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <sstream>

namespace stride {
namespace {

constexpr std::uint32_t kGgufMagic = 0x46554747;  // GGUF
constexpr std::uint64_t kAlign = 32;

enum class GgufType : std::uint32_t {
    UINT8 = 0,
    UINT16 = 1,
    UINT32 = 2,
    INT32 = 3,
    FLOAT32 = 4,
    BOOL = 5,
    STRING = 6,
    ARRAY = 7,
    UINT64 = 8,
    INT64 = 9,
    FLOAT64 = 10,
};

struct Reader {
    std::vector<std::uint8_t> bytes;
    std::size_t i = 0;
    std::filesystem::path path;

    Status need(std::size_t n) const {
        if (i + n > bytes.size()) return Status::error("truncated GGUF file: " + path.string());
        return Status::ok();
    }

    template <typename T>
    Result<T> read_pod() {
        auto st = need(sizeof(T));
        if (!st) return st;
        T v{};
        std::memcpy(&v, bytes.data() + i, sizeof(T));
        i += sizeof(T);
        return v;
    }

    Result<std::string> read_string() {
        auto n = read_pod<std::uint64_t>();
        if (!n) return n.status();
        auto st = need(static_cast<std::size_t>(n.value()));
        if (!st) return st;
        std::string s(reinterpret_cast<const char*>(bytes.data() + i), static_cast<std::size_t>(n.value()));
        i += static_cast<std::size_t>(n.value());
        return s;
    }
};

DType ggml_to_dtype(std::uint32_t ggml) {
    // GGML_TYPE_F32 = 0, F16 = 1
    if (ggml == 0) return DType::F32;
    if (ggml == 1) return DType::F16;
    return DType::U8;
}

std::uint64_t align_up(std::uint64_t x, std::uint64_t a) { return (x + a - 1) / a * a; }

Result<GgufMetaValue> read_value(Reader& r, GgufType type);

Result<GgufMetaValue> read_value(Reader& r, std::uint32_t type_u) {
    return read_value(r, static_cast<GgufType>(type_u));
}

Result<GgufMetaValue> read_value(Reader& r, GgufType type) {
    GgufMetaValue out;
    switch (type) {
        case GgufType::UINT8: {
            auto v = r.read_pod<std::uint8_t>();
            if (!v) return v.status();
            out.data = static_cast<std::uint64_t>(v.value());
            return out;
        }
        case GgufType::UINT16: {
            auto v = r.read_pod<std::uint16_t>();
            if (!v) return v.status();
            out.data = static_cast<std::uint64_t>(v.value());
            return out;
        }
        case GgufType::UINT32: {
            auto v = r.read_pod<std::uint32_t>();
            if (!v) return v.status();
            out.data = static_cast<std::uint64_t>(v.value());
            return out;
        }
        case GgufType::UINT64: {
            auto v = r.read_pod<std::uint64_t>();
            if (!v) return v.status();
            out.data = v.value();
            return out;
        }
        case GgufType::INT32: {
            auto v = r.read_pod<std::int32_t>();
            if (!v) return v.status();
            out.data = static_cast<std::int64_t>(v.value());
            return out;
        }
        case GgufType::INT64: {
            auto v = r.read_pod<std::int64_t>();
            if (!v) return v.status();
            out.data = v.value();
            return out;
        }
        case GgufType::FLOAT32: {
            auto v = r.read_pod<float>();
            if (!v) return v.status();
            out.data = static_cast<double>(v.value());
            return out;
        }
        case GgufType::FLOAT64: {
            auto v = r.read_pod<double>();
            if (!v) return v.status();
            out.data = v.value();
            return out;
        }
        case GgufType::BOOL: {
            auto v = r.read_pod<std::uint8_t>();
            if (!v) return v.status();
            out.data = v.value() != 0;
            return out;
        }
        case GgufType::STRING: {
            auto v = r.read_string();
            if (!v) return v.status();
            out.data = v.take();
            return out;
        }
        case GgufType::ARRAY: {
            auto subtype = r.read_pod<std::uint32_t>();
            if (!subtype) return subtype.status();
            auto n = r.read_pod<std::uint64_t>();
            if (!n) return n.status();
            if (static_cast<GgufType>(subtype.value()) == GgufType::STRING) {
                std::vector<std::string> arr;
                arr.reserve(static_cast<std::size_t>(n.value()));
                for (std::uint64_t i = 0; i < n.value(); ++i) {
                    auto s = r.read_string();
                    if (!s) return s.status();
                    arr.push_back(s.take());
                }
                out.data = std::move(arr);
                return out;
            }
            for (std::uint64_t i = 0; i < n.value(); ++i) {
                auto skip = read_value(r, subtype.value());
                if (!skip) return skip.status();
            }
            out.data = std::vector<std::string>{};
            return out;
        }
    }
    return Status::error("unsupported GGUF metadata type");
}

std::int64_t meta_int(const GgufFile& f, const std::string& key, std::int64_t fallback = 0) {
    auto it = f.metadata.find(key);
    if (it == f.metadata.end()) return fallback;
    if (auto* u = std::get_if<std::uint64_t>(&it->second.data)) return static_cast<std::int64_t>(*u);
    if (auto* i = std::get_if<std::int64_t>(&it->second.data)) return *i;
    if (auto* d = std::get_if<double>(&it->second.data)) return static_cast<std::int64_t>(*d);
    return fallback;
}

double meta_f(const GgufFile& f, const std::string& key, double fallback) {
    auto it = f.metadata.find(key);
    if (it == f.metadata.end()) return fallback;
    if (auto* d = std::get_if<double>(&it->second.data)) return *d;
    return fallback;
}

std::string meta_s(const GgufFile& f, const std::string& key) {
    auto it = f.metadata.find(key);
    if (it == f.metadata.end()) return {};
    if (auto* s = std::get_if<std::string>(&it->second.data)) return *s;
    return {};
}

}  // namespace

Result<GgufFile> read_gguf_index(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return Status::error("cannot open GGUF file: " + path.string());
    in.seekg(0, std::ios::end);
    const auto size = static_cast<std::size_t>(in.tellg());
    in.seekg(0, std::ios::beg);
    Reader r;
    r.path = path;
    r.bytes.resize(size);
    in.read(reinterpret_cast<char*>(r.bytes.data()), static_cast<std::streamsize>(size));
    if (!in) return Status::error("failed reading GGUF file");

    auto magic = r.read_pod<std::uint32_t>();
    if (!magic) return magic.status();
    if (magic.value() != kGgufMagic) return Status::error("not a GGUF file (bad magic)");
    auto version = r.read_pod<std::uint32_t>();
    if (!version) return version.status();
    auto tensor_count = r.read_pod<std::uint64_t>();
    if (!tensor_count) return tensor_count.status();
    auto kv_count = r.read_pod<std::uint64_t>();
    if (!kv_count) return kv_count.status();

    GgufFile file;
    file.version = version.value();
    for (std::uint64_t i = 0; i < kv_count.value(); ++i) {
        auto key = r.read_string();
        if (!key) return key.status();
        auto type = r.read_pod<std::uint32_t>();
        if (!type) return type.status();
        auto val = read_value(r, type.value());
        if (!val) return val.status();
        file.metadata.emplace(key.take(), val.take());
    }
    for (std::uint64_t t = 0; t < tensor_count.value(); ++t) {
        auto name = r.read_string();
        if (!name) return name.status();
        auto n_dims = r.read_pod<std::uint32_t>();
        if (!n_dims) return n_dims.status();
        Shape shape;
        shape.resize(n_dims.value());
        for (std::uint32_t d = 0; d < n_dims.value(); ++d) {
            auto dim = r.read_pod<std::uint64_t>();
            if (!dim) return dim.status();
            shape[d] = static_cast<std::int64_t>(dim.value());
        }
        std::reverse(shape.begin(), shape.end());  // GGUF stores reversed dims
        auto ggml_type = r.read_pod<std::uint32_t>();
        if (!ggml_type) return ggml_type.status();
        auto offset = r.read_pod<std::uint64_t>();
        if (!offset) return offset.status();
        TensorLocation loc;
        loc.name = name.take();
        loc.dtype = ggml_to_dtype(ggml_type.value());
        loc.shape = std::move(shape);
        loc.offset = offset.value();
        auto numel = shape_numel_checked(loc.shape);
        if (!numel) return numel.status();
        loc.nbytes = static_cast<std::uint64_t>(numel.value()) * dtype_size(loc.dtype);
        if (ggml_type.value() > 1) {
            return Status::error("unsupported GGUF tensor type for " + loc.name +
                                 " (only F32/F16 are implemented)");
        }
        file.tensors.push_back(std::move(loc));
    }
    file.data_offset = align_up(static_cast<std::uint64_t>(r.i), kAlign);
    for (auto& loc : file.tensors) loc.offset += file.data_offset;
    return file;
}

Result<ModelConfig> llama_config_from_gguf(const GgufFile& gguf) {
    const std::string arch = meta_s(gguf, "general.architecture");
    if (!arch.empty() && arch != "llama" && arch != "stride-llama") {
        return Status::error("unsupported GGUF architecture '" + arch +
                             "'. Stride currently implements llama (decoder-only, F32/F16).");
    }
    ModelConfig cfg;
    cfg.architecture = arch.empty() ? "llama" : arch;
    cfg.format = "gguf";
    cfg.model_name = meta_s(gguf, "general.name");
    const std::string p = cfg.architecture;
    cfg.vocab_size = meta_int(gguf, p + ".vocab_size", meta_int(gguf, "llama.vocab_size"));
    cfg.d_model = meta_int(gguf, p + ".embedding_length", meta_int(gguf, "llama.embedding_length"));
    cfg.n_layers = meta_int(gguf, p + ".block_count", meta_int(gguf, "llama.block_count"));
    cfg.n_heads = meta_int(gguf, p + ".attention.head_count", meta_int(gguf, "llama.attention.head_count"));
    cfg.n_kv_heads =
        meta_int(gguf, p + ".attention.head_count_kv", meta_int(gguf, "llama.attention.head_count_kv", cfg.n_heads));
    cfg.d_ff = meta_int(gguf, p + ".feed_forward_length", meta_int(gguf, "llama.feed_forward_length"));
    cfg.max_seq_len = meta_int(gguf, p + ".context_length", meta_int(gguf, "llama.context_length", 2048));
    cfg.rope_theta = static_cast<float>(
        meta_f(gguf, p + ".rope.freq_base", meta_f(gguf, "llama.rope.freq_base", 10000.0)));
    cfg.norm_eps = static_cast<float>(
        meta_f(gguf, p + ".attention.layer_norm_rms_epsilon",
               meta_f(gguf, "llama.attention.layer_norm_rms_epsilon", 1e-5)));
    auto err = cfg.validate();
    if (!err.empty()) return Status::error(err);
    return cfg;
}

Result<BpeTokenizer> tokenizer_from_gguf(const GgufFile& gguf) {
    auto it = gguf.metadata.find("tokenizer.ggml.tokens");
    if (it == gguf.metadata.end()) return Status::error("GGUF file has no tokenizer.ggml.tokens");
    auto* tokens = std::get_if<std::vector<std::string>>(&it->second.data);
    if (!tokens) return Status::error("tokenizer.ggml.tokens is not a string array");
    std::vector<std::string> merges;
    if (auto m = gguf.metadata.find("tokenizer.ggml.merges"); m != gguf.metadata.end()) {
        if (auto* mv = std::get_if<std::vector<std::string>>(&m->second.data)) merges = *mv;
    }
    TokenId bos = static_cast<TokenId>(meta_int(gguf, "tokenizer.ggml.bos_token_id", 1));
    TokenId eos = static_cast<TokenId>(meta_int(gguf, "tokenizer.ggml.eos_token_id", 2));
    TokenId unk = static_cast<TokenId>(meta_int(gguf, "tokenizer.ggml.unknown_token_id", 0));
    return BpeTokenizer::from_vocab_and_merges(*tokens, merges, bos, eos, unk);
}

namespace {

void write_u32(std::vector<std::uint8_t>& b, std::uint32_t v) {
    const auto* p = reinterpret_cast<const std::uint8_t*>(&v);
    b.insert(b.end(), p, p + 4);
}
void write_u64(std::vector<std::uint8_t>& b, std::uint64_t v) {
    const auto* p = reinterpret_cast<const std::uint8_t*>(&v);
    b.insert(b.end(), p, p + 8);
}
void write_str(std::vector<std::uint8_t>& b, const std::string& s) {
    write_u64(b, s.size());
    b.insert(b.end(), s.begin(), s.end());
}
void write_kv_u32(std::vector<std::uint8_t>& b, const std::string& k, std::uint32_t v) {
    write_str(b, k);
    write_u32(b, static_cast<std::uint32_t>(GgufType::UINT32));
    write_u32(b, v);
}
void write_kv_f32(std::vector<std::uint8_t>& b, const std::string& k, float v) {
    write_str(b, k);
    write_u32(b, static_cast<std::uint32_t>(GgufType::FLOAT32));
    const auto* p = reinterpret_cast<const std::uint8_t*>(&v);
    b.insert(b.end(), p, p + 4);
}
void write_kv_str(std::vector<std::uint8_t>& b, const std::string& k, const std::string& v) {
    write_str(b, k);
    write_u32(b, static_cast<std::uint32_t>(GgufType::STRING));
    write_str(b, v);
}
void write_kv_str_array(std::vector<std::uint8_t>& b, const std::string& k,
                        const std::vector<std::string>& arr) {
    write_str(b, k);
    write_u32(b, static_cast<std::uint32_t>(GgufType::ARRAY));
    write_u32(b, static_cast<std::uint32_t>(GgufType::STRING));
    write_u64(b, arr.size());
    for (const auto& s : arr) write_str(b, s);
}

}  // namespace

Status write_gguf(const std::filesystem::path& path, const ModelConfig& config,
                  const BpeTokenizer& tokenizer, const std::map<std::string, Tensor>& tensors) {
    std::vector<std::uint8_t> header;
    write_u32(header, kGgufMagic);
    write_u32(header, 3);
    write_u64(header, tensors.size());

    std::vector<std::uint8_t> kv;
    write_kv_str(kv, "general.architecture", "llama");
    write_kv_str(kv, "general.name", config.model_name.empty() ? "stride-tiny" : config.model_name);
    write_kv_u32(kv, "llama.vocab_size", static_cast<std::uint32_t>(config.vocab_size));
    write_kv_u32(kv, "llama.embedding_length", static_cast<std::uint32_t>(config.d_model));
    write_kv_u32(kv, "llama.block_count", static_cast<std::uint32_t>(config.n_layers));
    write_kv_u32(kv, "llama.attention.head_count", static_cast<std::uint32_t>(config.n_heads));
    write_kv_u32(kv, "llama.attention.head_count_kv", static_cast<std::uint32_t>(config.n_kv_heads));
    write_kv_u32(kv, "llama.feed_forward_length", static_cast<std::uint32_t>(config.d_ff));
    write_kv_u32(kv, "llama.context_length", static_cast<std::uint32_t>(config.max_seq_len));
    write_kv_f32(kv, "llama.rope.freq_base", config.rope_theta);
    write_kv_f32(kv, "llama.attention.layer_norm_rms_epsilon", config.norm_eps);
    write_kv_u32(kv, "tokenizer.ggml.bos_token_id", static_cast<std::uint32_t>(tokenizer.bos_id()));
    write_kv_u32(kv, "tokenizer.ggml.eos_token_id", static_cast<std::uint32_t>(tokenizer.eos_id()));
    write_kv_u32(kv, "tokenizer.ggml.unknown_token_id", static_cast<std::uint32_t>(tokenizer.unk_id()));
    std::vector<std::string> vocab;
    vocab.reserve(static_cast<std::size_t>(tokenizer.vocab_size()));
    for (TokenId i = 0; i < tokenizer.vocab_size(); ++i) vocab.push_back(tokenizer.id_to_piece(i));
    write_kv_str_array(kv, "tokenizer.ggml.tokens", vocab);
    write_kv_str_array(kv, "tokenizer.ggml.merges", tokenizer.merges());
    write_u64(header, 16);  // placeholder kv count; we count below

    // recount kv: we wrote 16 keys
    // rewrite kv_count already written? simpler: build kv_count first.
    // The placeholder is wrong because we already wrote tensor_count then kv_count as 16.
    // Let's rebuild cleanly.

    std::vector<std::uint8_t> out;
    write_u32(out, kGgufMagic);
    write_u32(out, 3);
    write_u64(out, tensors.size());
    write_u64(out, 16);
    out.insert(out.end(), kv.begin(), kv.end());

    std::vector<std::uint8_t> tensor_blob;
    std::uint64_t offset = 0;
    struct Item {
        std::string name;
        Shape shape;
        std::uint64_t off;
        const Tensor* t;
    };
    std::vector<Item> items;
    for (const auto& [name, t] : tensors) {
        items.push_back({name, t.shape(), offset, &t});
        offset += static_cast<std::uint64_t>(t.nbytes());
    }
    for (const auto& item : items) {
        write_str(out, item.name);
        write_u32(out, static_cast<std::uint32_t>(item.shape.size()));
        for (auto it = item.shape.rbegin(); it != item.shape.rend(); ++it) {
            write_u64(out, static_cast<std::uint64_t>(*it));
        }
        write_u32(out, 0);  // F32
        write_u64(out, item.off);
    }
    while (out.size() % kAlign != 0) out.push_back(0);
    for (const auto& item : items) {
        const auto* p = reinterpret_cast<const std::uint8_t*>(item.t->data());
        out.insert(out.end(), p, p + item.t->nbytes());
    }
    std::ofstream f(path, std::ios::binary);
    if (!f) return Status::error("cannot write " + path.string());
    f.write(reinterpret_cast<const char*>(out.data()), static_cast<std::streamsize>(out.size()));
    return Status::ok();
}

}  // namespace stride

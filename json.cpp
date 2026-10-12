#include "json.hpp"

#include <cctype>
#include <sstream>

namespace stride::json {
namespace {

struct Parser {
    const std::string& s;
    std::size_t i = 0;

    void skip() {
        while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
    }

    Result<Value> parse_value() {
        skip();
        if (i >= s.size()) return Status::error("unexpected end of JSON");
        const char c = s[i];
        if (c == '{') return parse_object();
        if (c == '[') return parse_array();
        if (c == '"') return parse_string();
        if (c == 't' || c == 'f') return parse_bool();
        if (c == 'n') return parse_null();
        if (c == '-' || std::isdigit(static_cast<unsigned char>(c))) return parse_number();
        return Status::error(std::string("unexpected JSON char: ") + c);
    }

    Result<Value> parse_null() {
        if (s.compare(i, 4, "null") != 0) return Status::error("invalid null");
        i += 4;
        Value v;
        v.data = nullptr;
        return v;
    }

    Result<Value> parse_bool() {
        Value v;
        if (s.compare(i, 4, "true") == 0) {
            i += 4;
            v.data = true;
            return v;
        }
        if (s.compare(i, 5, "false") == 0) {
            i += 5;
            v.data = false;
            return v;
        }
        return Status::error("invalid bool");
    }

    Result<Value> parse_number() {
        const std::size_t start = i;
        if (s[i] == '-') ++i;
        while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i]))) ++i;
        if (i < s.size() && s[i] == '.') {
            ++i;
            while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i]))) ++i;
        }
        if (i < s.size() && (s[i] == 'e' || s[i] == 'E')) {
            ++i;
            if (i < s.size() && (s[i] == '+' || s[i] == '-')) ++i;
            while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i]))) ++i;
        }
        Value v;
        v.data = std::stod(s.substr(start, i - start));
        return v;
    }

    Result<Value> parse_string() {
        if (s[i] != '"') return Status::error("expected string");
        ++i;
        std::string out;
        while (i < s.size()) {
            char c = s[i++];
            if (c == '"') {
                Value v;
                v.data = std::move(out);
                return v;
            }
            if (c == '\\' && i < s.size()) {
                char e = s[i++];
                switch (e) {
                    case '"':
                    case '\\':
                    case '/':
                        out.push_back(e);
                        break;
                    case 'n':
                        out.push_back('\n');
                        break;
                    case 't':
                        out.push_back('\t');
                        break;
                    case 'r':
                        out.push_back('\r');
                        break;
                    default:
                        out.push_back(e);
                        break;
                }
            } else {
                out.push_back(c);
            }
        }
        return Status::error("unterminated string");
    }

    Result<Value> parse_array() {
        ++i;
        Array arr;
        skip();
        if (i < s.size() && s[i] == ']') {
            ++i;
            Value v;
            v.data = std::move(arr);
            return v;
        }
        while (true) {
            auto item = parse_value();
            if (!item) return item.status();
            arr.push_back(item.take());
            skip();
            if (i >= s.size()) return Status::error("unterminated array");
            if (s[i] == ',') {
                ++i;
                continue;
            }
            if (s[i] == ']') {
                ++i;
                break;
            }
            return Status::error("expected , or ]");
        }
        Value v;
        v.data = std::move(arr);
        return v;
    }

    Result<Value> parse_object() {
        ++i;
        Object obj;
        skip();
        if (i < s.size() && s[i] == '}') {
            ++i;
            Value v;
            v.data = std::move(obj);
            return v;
        }
        while (true) {
            skip();
            auto key = parse_string();
            if (!key) return key.status();
            skip();
            if (i >= s.size() || s[i] != ':') return Status::error("expected :");
            ++i;
            auto val = parse_value();
            if (!val) return val.status();
            obj.emplace(key.value().as_string(), val.take());
            skip();
            if (i >= s.size()) return Status::error("unterminated object");
            if (s[i] == ',') {
                ++i;
                continue;
            }
            if (s[i] == '}') {
                ++i;
                break;
            }
            return Status::error("expected , or }");
        }
        Value v;
        v.data = std::move(obj);
        return v;
    }
};

void dump_into(const Value& value, std::ostringstream& oss) {
    if (value.is_null()) {
        oss << "null";
    } else if (value.is_bool()) {
        oss << (value.as_bool() ? "true" : "false");
    } else if (value.is_number()) {
        oss << value.as_number();
    } else if (value.is_string()) {
        oss << '"';
        for (char c : value.as_string()) {
            if (c == '"' || c == '\\') oss << '\\';
            oss << c;
        }
        oss << '"';
    } else if (value.is_array()) {
        oss << '[';
        bool first = true;
        for (const auto& item : value.as_array()) {
            if (!first) oss << ',';
            first = false;
            dump_into(item, oss);
        }
        oss << ']';
    } else {
        oss << '{';
        bool first = true;
        for (const auto& [k, v] : value.as_object()) {
            if (!first) oss << ',';
            first = false;
            oss << '"';
            oss << k;
            oss << "\":";
            dump_into(v, oss);
        }
        oss << '}';
    }
}

}  // namespace

Result<Value> parse(const std::string& text) {
    Parser p{text};
    auto v = p.parse_value();
    if (!v) return v;
    p.skip();
    if (p.i != text.size()) return Status::error("trailing JSON content");
    return v;
}

std::string dump(const Value& value) {
    std::ostringstream oss;
    dump_into(value, oss);
    return oss.str();
}

}  // namespace stride::json

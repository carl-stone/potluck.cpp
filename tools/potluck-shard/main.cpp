#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

struct bounds {
    uint32_t start = 0;
    uint32_t end = 0;
};

[[noreturn]] void usage(const char * exe, const char * message = nullptr) {
    if (message != nullptr) {
        std::fprintf(stderr, "potluck-shard: %s\n", message);
    }
    std::fprintf(stderr,
        "usage: %s MODEL.gguf (--parts N | --bounds A,B,C,...) [-o OUTDIR] [--dry-run]\n",
        exe);
    std::exit(message == nullptr ? EXIT_SUCCESS : EXIT_FAILURE);
}

uint32_t parse_u32(const std::string & text, const char * option) {
    if (text.empty() || text[0] == '-') {
        usage("potluck-shard", (std::string("invalid value for ") + option).c_str());
    }
    char * end = nullptr;
    const unsigned long value = std::strtoul(text.c_str(), &end, 10);
    if (end == text.c_str() || *end != '\0' || value > std::numeric_limits<uint32_t>::max()) {
        usage("potluck-shard", (std::string("invalid value for ") + option + ": " + text).c_str());
    }
    return static_cast<uint32_t>(value);
}

std::vector<uint32_t> parse_csv(const std::string & spec) {
    std::vector<uint32_t> values;
    size_t at = 0;
    for (;;) {
        const size_t comma = spec.find(',', at);
        const std::string part = spec.substr(at, comma == std::string::npos ? std::string::npos : comma - at);
        if (part.empty()) {
            usage("potluck-shard", "--bounds needs comma-separated integers");
        }
        values.push_back(parse_u32(part, "--bounds"));
        if (comma == std::string::npos) {
            break;
        }
        at = comma + 1;
    }
    return values;
}

std::string basename_of(const std::string & path) {
    const size_t slash = path.find_last_of("/\\");
    return slash == std::string::npos ? path : path.substr(slash + 1);
}

std::string stem_of(const std::string & path) {
    const std::string base = basename_of(path);
    const size_t dot = base.rfind('.');
    return dot == std::string::npos ? base : base.substr(0, dot);
}

void write_zeros(std::ofstream & out, size_t count) {
    static const std::vector<char> zeros(1u << 20, 0);
    while (count > 0) {
        const size_t n = std::min(count, zeros.size());
        out.write(zeros.data(), static_cast<std::streamsize>(n));
        count -= n;
    }
}

void copy_range(std::ifstream & in, std::ofstream & out, uint64_t offset, size_t bytes) {
    static std::vector<char> buffer(1u << 20);
    in.clear();
    in.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    if (!in) {
        throw std::runtime_error("cannot seek input tensor data");
    }
    while (bytes > 0) {
        const size_t n = std::min(bytes, buffer.size());
        in.read(buffer.data(), static_cast<std::streamsize>(n));
        if (in.gcount() != static_cast<std::streamsize>(n)) {
            throw std::runtime_error("input tensor data is truncated");
        }
        out.write(buffer.data(), static_cast<std::streamsize>(n));
        bytes -= n;
    }
}

bool layer_tensor(const std::string & name, uint32_t start, uint32_t end) {
    if (name.rfind("blk.", 0) != 0) {
        return false;
    }
    const size_t dot = name.find('.', 4);
    if (dot == std::string::npos) {
        return false;
    }
    const std::string number = name.substr(4, dot - 4);
    char * finish = nullptr;
    const unsigned long layer = std::strtoul(number.c_str(), &finish, 10);
    return finish != number.c_str() && *finish == '\0' &&
           layer >= start && layer < end;
}

bool belongs_to_shard(const std::string & name, uint32_t start, uint32_t end,
                      uint32_t n_layer, bool tail_needs_token_embd) {
    if (layer_tensor(name, start, end)) {
        return true;
    }
    if ((start == 0 || (end == n_layer && tail_needs_token_embd)) &&
        name.rfind("token_embd.", 0) == 0) {
        return true;
    }
    if (end == n_layer && (name == "output_norm" || name.rfind("output_norm.", 0) == 0 ||
                           name == "output" || name.rfind("output.", 0) == 0)) {
        return true;
    }
    return false;
}

struct input_file {
    gguf_context * gguf = nullptr;
    ggml_context * meta = nullptr;
    std::ifstream data;

    ~input_file() {
        if (gguf != nullptr) {
            gguf_free(gguf);
        }
        if (meta != nullptr) {
            ggml_free(meta);
        }
    }
};

uint32_t model_layers(const gguf_context * ctx) {
    const int arch_id = gguf_find_key(ctx, "general.architecture");
    if (arch_id < 0) {
        throw std::runtime_error("GGUF has no general.architecture metadata");
    }
    const std::string arch = gguf_get_val_str(ctx, arch_id);
    const std::string key = arch + ".block_count";
    const int layer_id = gguf_find_key(ctx, key.c_str());
    if (layer_id < 0) {
        throw std::runtime_error("GGUF has no " + key + " metadata");
    }
    if (gguf_get_kv_type(ctx, layer_id) != GGUF_TYPE_UINT32) {
        throw std::runtime_error(key + " is not a uint32");
    }
    return gguf_get_val_u32(ctx, layer_id);
}

void validate_bounds(const std::vector<uint32_t> & values, uint32_t n_layer) {
    if (values.size() < 2 || values.front() != 0 || values.back() != n_layer) {
        throw std::runtime_error("bounds must start at 0 and end at the model layer count");
    }
    for (size_t i = 1; i < values.size(); ++i) {
        if (values[i] <= values[i - 1]) {
            throw std::runtime_error("bounds must be strictly increasing");
        }
    }
}

void write_shard(input_file & input, const std::string & input_path,
                 const std::string & out_path, const bounds window,
                 uint32_t shard_index, uint32_t shard_count, uint32_t n_layer,
                 bool tail_needs_token_embd, bool dry_run,
                 size_t tensor_count, uint64_t tensor_bytes) {
    gguf_context * out = gguf_init_empty();
    if (out == nullptr) {
        throw std::runtime_error("cannot allocate output GGUF metadata");
    }
    gguf_set_kv(out, input.gguf);
    gguf_set_val_u32(out, "potluck.shard.index", shard_index);
    gguf_set_val_u32(out, "potluck.shard.count", shard_count);
    gguf_set_val_u32(out, "potluck.shard.start", window.start);
    gguf_set_val_u32(out, "potluck.shard.end", window.end);
    gguf_set_val_str(out, "potluck.source.file", basename_of(input_path).c_str());

    const int n_tensors = gguf_get_n_tensors(input.gguf);
    for (int i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(input.gguf, i);
        if (belongs_to_shard(name, window.start, window.end, n_layer, tail_needs_token_embd)) {
            ggml_tensor * tensor = ggml_get_tensor(input.meta, name);
            if (tensor == nullptr) {
                gguf_free(out);
                throw std::runtime_error("cannot add tensor " + std::string(name));
            }
            gguf_add_tensor(out, tensor);
        }
    }

    if (!dry_run) {
        std::ofstream output(out_path, std::ios::binary);
        output.exceptions(std::ofstream::failbit | std::ofstream::badbit);
        const size_t metadata_bytes = gguf_get_meta_size(out);
        std::vector<uint8_t> metadata(metadata_bytes);
        gguf_get_meta_data(out, metadata.data());
        output.write(reinterpret_cast<const char *>(metadata.data()),
                     static_cast<std::streamsize>(metadata.size()));

        for (int i = 0; i < gguf_get_n_tensors(out); ++i) {
            const char * name = gguf_get_tensor_name(out, i);
            const int input_index = gguf_find_tensor(input.gguf, name);
            if (input_index < 0) {
                gguf_free(out);
                throw std::runtime_error("output tensor is missing from input: " + std::string(name));
            }
            const ggml_tensor * tensor = ggml_get_tensor(input.meta, name);
            const size_t bytes = ggml_nbytes(tensor);
            const uint64_t offset = gguf_get_data_offset(input.gguf) +
                                    gguf_get_tensor_offset(input.gguf, input_index);
            copy_range(input.data, output, offset, bytes);
            write_zeros(output, GGML_PAD(bytes, GGUF_DEFAULT_ALIGNMENT) - bytes);
        }
        output.close();
    }
    gguf_free(out);

    std::printf("%u  [%u,%u)  %zu tensors  %" PRIu64 " tensor-bytes%s\n",
                shard_index, window.start, window.end, tensor_count, tensor_bytes,
                dry_run ? "  (dry-run)" : "");
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 3) {
        usage(argv[0]);
    }

    const std::string input_path = argv[1];
    uint32_t parts = 0;
    std::vector<uint32_t> explicit_bounds;
    std::string output_dir = ".";
    bool dry_run = false;

    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--parts") {
            if (++i >= argc) usage(argv[0], "missing value for --parts");
            parts = parse_u32(argv[i], "--parts");
            if (parts == 0) usage(argv[0], "--parts must be positive");
        } else if (arg == "--bounds") {
            if (++i >= argc) usage(argv[0], "missing value for --bounds");
            explicit_bounds = parse_csv(argv[i]);
        } else if (arg == "-o" || arg == "--outdir") {
            if (++i >= argc) usage(argv[0], "missing value for -o/--outdir");
            output_dir = argv[i];
        } else if (arg == "--dry-run") {
            dry_run = true;
        } else {
            usage(argv[0], ("unknown argument: " + arg).c_str());
        }
    }
    if (parts != 0 && !explicit_bounds.empty()) {
        usage(argv[0], "choose --parts or --bounds, not both");
    }
    if (parts == 0 && explicit_bounds.empty()) {
        usage(argv[0], "one of --parts or --bounds is required");
    }

    input_file input;
    input.data.open(input_path, std::ios::binary);
    if (!input.data) {
        std::fprintf(stderr, "potluck-shard: cannot open %s\n", input_path.c_str());
        return EXIT_FAILURE;
    }
    gguf_init_params params = { true, &input.meta };
    input.gguf = gguf_init_from_file(input_path.c_str(), params);
    if (input.gguf == nullptr) {
        std::fprintf(stderr, "potluck-shard: cannot read GGUF metadata from %s\n", input_path.c_str());
        return EXIT_FAILURE;
    }

    try {
        const uint32_t n_layer = model_layers(input.gguf);
        const bool tail_needs_token_embd = gguf_find_tensor(input.gguf, "output.weight") < 0;
        std::filesystem::create_directories(output_dir);
        std::vector<uint32_t> values;
        if (!explicit_bounds.empty()) {
            values = explicit_bounds;
        } else {
            values.resize(parts + 1);
            for (uint32_t i = 0; i <= parts; ++i) {
                values[i] = static_cast<uint32_t>((static_cast<uint64_t>(n_layer) * i) / parts);
            }
        }
        validate_bounds(values, n_layer);
        const uint32_t n_shards = static_cast<uint32_t>(values.size() - 1);
        const std::string stem = stem_of(input_path);
        std::printf("shard  window      tensors  tensor-bytes\n");
        std::printf("-----  ----------  -------  ------------\n");
        for (uint32_t i = 0; i < n_shards; ++i) {
            const bounds window { values[i], values[i + 1] };
            size_t tensor_count = 0;
            uint64_t tensor_bytes = 0;
            for (int t = 0; t < gguf_get_n_tensors(input.gguf); ++t) {
                const char * name = gguf_get_tensor_name(input.gguf, t);
                if (!belongs_to_shard(name, window.start, window.end, n_layer, tail_needs_token_embd)) {
                    continue;
                }
                ++tensor_count;
                tensor_bytes += ggml_nbytes(ggml_get_tensor(input.meta, name));
            }
            const std::string output = output_dir + "/" + stem + ".potluck-" +
                                       std::to_string(i) + "of" + std::to_string(n_shards) + ".gguf";
            write_shard(input, input_path, output, window, i, n_shards, n_layer,
                        tail_needs_token_embd, dry_run, tensor_count, tensor_bytes);
        }
        return EXIT_SUCCESS;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "potluck-shard: %s\n", e.what());
        return EXIT_FAILURE;
    }
}

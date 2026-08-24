#include "llama.h"
#include "arg.h"
#include "chat.h"
#include <cpp-httplib/httplib.h>
#include "internal.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__APPLE__)
#include <mach-o/dyld.h>
#elif defined(__linux__)
#include <limits.h>
#include <unistd.h>
#endif

namespace {

struct cli_options {
    std::string model_path;
    std::string hf_repo;
    std::string prompt;
    uint32_t n_predict = 128;
    uint32_t n_ctx = 4096;
    uint32_t n_seq_max = 1;
    uint32_t n_ubatch = 512;
    uint32_t top_k = 0;
    uint32_t seed = 0;
    float temp = 0.0f;
    float top_p = 0.0f;
    bool conversation = false;
    bool force = false;
    potluck::prefetch_mode prefetch = potluck::prefetch_mode::advise;
    std::vector<uint32_t> layer_window;
    double gpu_mem_gib = 0.0;
    int32_t k_override = -1;
    double master_priority = 1.01;
    std::string spec_draft_model;
    std::vector<std::string> spec_types;
    uint32_t spec_draft_n_max = 0;
};

std::string executable_dir(const char * argv0) {
    std::string path;
#if defined(__APPLE__)
    uint32_t size = 0;
    _NSGetExecutablePath(nullptr, &size);
    std::vector<char> buffer(size + 1, '\0');
    if (_NSGetExecutablePath(buffer.data(), &size) == 0) {
        path = buffer.data();
    }
#elif defined(__linux__)
    char buffer[PATH_MAX + 1] = {};
    const ssize_t count = readlink("/proc/self/exe", buffer, PATH_MAX);
    if (count > 0) {
        buffer[count] = '\0';
        path = buffer;
    }
#endif
    if (path.empty() && argv0 != nullptr) {
        path = argv0;
    }
    const size_t slash = path.find_last_of("/\\");
    return slash == std::string::npos ? std::string() : path.substr(0, slash);
}

std::string basename_of(const std::string & path) {
    const size_t slash = path.find_last_of("/\\");
    return slash == std::string::npos ? path : path.substr(slash + 1);
}

uint32_t parse_u32(const std::string & value, const char * option) {
    try {
        size_t used = 0;
        const unsigned long parsed = std::stoul(value, &used);
        if (used != value.size() || parsed > std::numeric_limits<uint32_t>::max()) {
            throw std::runtime_error("invalid value");
        }
        return static_cast<uint32_t>(parsed);
    } catch (...) {
        throw std::runtime_error(std::string(option) + " expects a non-negative integer");
    }
}
uint32_t parse_positive_u32(const std::string & value, const char * option) {
    const uint32_t parsed = parse_u32(value, option);
    if (parsed == 0) {
        throw std::runtime_error(std::string(option) + " expects a positive integer");
    }
    return parsed;
}

int32_t parse_positive_i32(const std::string & value, const char * option) {
    const uint32_t parsed = parse_positive_u32(value, option);
    if (parsed > static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
        throw std::runtime_error(std::string(option) + " exceeds the supported range");
    }
    return static_cast<int32_t>(parsed);
}

double parse_positive_double(const std::string & value, const char * option) {
    try {
        size_t used = 0;
        const double parsed = std::stod(value, &used);
        if (used != value.size() || !std::isfinite(parsed) || parsed <= 0.0) {
            throw std::runtime_error("invalid value");
        }
        return parsed;
    } catch (...) {
        throw std::runtime_error(std::string(option) + " expects a positive finite number");
    }
}

std::vector<uint32_t> parse_layer_window(const std::string & text) {
    std::vector<uint32_t> windows;
    size_t start = 0;
    for (;;) {
        const size_t comma = text.find(',', start);
        const std::string value = text.substr(
            start, comma == std::string::npos ? std::string::npos : comma - start);
        if (value.empty()) {
            throw std::runtime_error("--layer-window values cannot be empty");
        }
        windows.push_back(parse_positive_u32(value, "--layer-window"));
        if (windows.size() > 32) {
            throw std::runtime_error("--layer-window accepts at most 32 entries");
        }
        if (comma == std::string::npos) {
            return windows;
        }
        start = comma + 1;
    }
}

void validate_manual_workload(const std::vector<uint32_t> & layer_window,
                              int32_t k_override, uint32_t n_layer) {
    if (!layer_window.empty()) {
        uint64_t total = 0;
        for (const uint32_t layers : layer_window) {
            total += layers;
        }
        if (total == 0 || n_layer % total != 0) {
            throw std::runtime_error(
                "--layer-window sum must divide the model layer count exactly");
        }
    }
    if (k_override > 0 && n_layer % static_cast<uint32_t>(k_override) != 0) {
        throw std::runtime_error("--n-cycles must divide the model layer count exactly");
    }
}
const char * prefetch_name(potluck::prefetch_mode mode) {
    switch (mode) {
        case potluck::prefetch_mode::off: return "off";
        case potluck::prefetch_mode::advise: return "advise";
        case potluck::prefetch_mode::force: return "force";
    }
    return "advise";
}

void log_manual_overrides(const std::vector<uint32_t> & layer_window,
                          double gpu_mem_gib, int32_t k_override,
                          double master_priority, potluck::prefetch_mode prefetch) {
    if (layer_window.empty() && gpu_mem_gib == 0.0 && k_override < 0 &&
        master_priority == 1.01 && prefetch == potluck::prefetch_mode::advise) {
        return;
    }
    std::fprintf(stderr, "potluck-cli: expert override");
    if (!layer_window.empty()) {
        std::fprintf(stderr, " layer-window=");
        for (size_t index = 0; index < layer_window.size(); ++index) {
            std::fprintf(stderr, "%s%u", index == 0 ? "" : ",", layer_window[index]);
        }
    }
    if (gpu_mem_gib > 0.0) {
        std::fprintf(stderr, " gpu-mem=%.3fGiB", gpu_mem_gib);
    }
    if (k_override > 0) {
        std::fprintf(stderr, " n-cycles=%d", k_override);
    }
    if (master_priority != 1.01) {
        std::fprintf(stderr, " master-priority=%.6g", master_priority);
    }
    if (prefetch != potluck::prefetch_mode::advise) {
        std::fprintf(stderr, " prefetch=%s", prefetch_name(prefetch));
    }
    std::fputc('\n', stderr);
}

float parse_float(const std::string & value, const char * option) {
    try {
        size_t used = 0;
        const float parsed = std::stof(value, &used);
        if (used != value.size() || !std::isfinite(parsed)) {
            throw std::runtime_error("invalid value");
        }
        return parsed;
    } catch (...) {
        throw std::runtime_error(std::string(option) + " expects a finite number");
    }
}

potluck::prefetch_mode parse_prefetch(const std::string & value) {
    if (value == "off") {
        return potluck::prefetch_mode::off;
    }
    if (value == "advise") {
        return potluck::prefetch_mode::advise;
    }
    if (value == "force") {
        return potluck::prefetch_mode::force;
    }
    throw std::runtime_error("--prefetch expects off, advise, or force");
}
std::vector<std::string> parse_spec_types(const std::string & text) {
    std::vector<std::string> names;
    size_t start = 0;
    for (;;) {
        const size_t comma = text.find(',', start);
        const std::string name = text.substr(
            start, comma == std::string::npos ? std::string::npos : comma - start);
        if (name.empty()) {
            throw std::runtime_error("--spec-type values cannot be empty");
        }
        names.push_back(name);
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }
    const std::vector<common_speculative_type> types =
        common_speculative_types_from_names(names);
    if (std::find(types.begin(), types.end(), COMMON_SPECULATIVE_TYPE_NONE) != types.end()) {
        throw std::runtime_error("--spec-type must select an implementation");
    }
    return names;
}


std::string resolve_hf_model(const std::string & repo) {
    common_params params;
    params.model.hf_repo = repo;
    try {
        common_models_handler handler = common_models_handler_init(params, LLAMA_EXAMPLE_CLI);
        if (common_models_handler_is_preset_repo(handler)) {
            throw std::runtime_error("repository contains a preset, not a GGUF model");
        }
        if (handler.plan.primary.path.empty() || handler.plan.model_files.empty()) {
            throw std::runtime_error("repository has no compatible GGUF model");
        }
        if (handler.plan.model_files.size() != 1) {
            throw std::runtime_error("split GGUF repositories are not supported");
        }
        common_models_handler_apply(handler, params);
    } catch (const std::exception & exception) {
        throw std::runtime_error("failed to resolve Hugging Face model '" + repo + "': " +
                                 exception.what());
    }
    if (params.model.path.empty()) {
        throw std::runtime_error("Hugging Face download did not produce a model path");
    }
    return params.model.path;
}

void print_usage() {
    std::printf("usage: potluck-cli -m MODEL | -hf REPO [options]\n"
                "  -m, --model MODEL\n"
                "  -hf, --hf REPO\n"
                "  -p, --prompt PROMPT\n"
                "  -n, --n-predict N            generated tokens (default: 128)\n"
                "  -c, --ctx-size, --ctx N      context size (default: 4096)\n"
                "  --slots, --batch N           conversation slots (default: 1)\n"
                "  --ubatch N                   physical batch size (default: 512)\n"
                "  -cnv, --conversation         interactive multi-turn conversation\n"
                "  --temp F                     sampling temperature\n"
                "  --top-p F                    nucleus sampling\n"
                "  --top-k N                    top-k sampling\n"
                "  --seed N                     sampling seed\n"
                "  -lw, --layer-window, --n-layer-window A,B\n"
                "                               layers per ring window\n"
                "  -gm, --gpu-mem N             accelerator memory cap in GiB\n"
                "  -k, --n-cycles N             ring cycle count\n"
                "  --master-priority F\n"
                "  --prefetch MODE off|advise|force\n"
                "  --force                       force per-window prefetch\n"
                "  --spec-type TYPE[,TYPE]\n"
                "  --spec-draft-model FILE\n"
                "  --spec-draft-n-max N\n");
}

cli_options parse_options(int argc, char ** argv) {
    cli_options options;
    for (int index = 1; index < argc; ++index) {
        const std::string arg = argv[index];
        const auto take = [&](const char * name) {
            if (++index >= argc) {
                throw std::runtime_error(std::string("missing value for ") + name);
            }
            return std::string(argv[index]);
        };
        if (arg == "-h" || arg == "--help") {
            print_usage();
            std::exit(0);
        } else if (arg == "-m" || arg == "--model") {
            options.model_path = take("-m");
        } else if (arg == "-hf" || arg == "--hf") {
            options.hf_repo = take("-hf");
        } else if (arg == "-p" || arg == "--prompt") {
            options.prompt = take("-p");
        } else if (arg == "-n" || arg == "--n-predict") {
            options.n_predict = parse_u32(take("-n"), "-n");
        } else if (arg == "-c" || arg == "--ctx-size" || arg == "--ctx") {
            options.n_ctx = parse_u32(take("-c"), "-c");
        } else if (arg == "--slots" || arg == "--batch") {
            options.n_seq_max = parse_positive_u32(take(arg.c_str()), arg.c_str());
        } else if (arg == "--ubatch") {
            options.n_ubatch = parse_positive_u32(take("--ubatch"), "--ubatch");
        } else if (arg == "-cnv" || arg == "--conversation") {
            options.conversation = true;
        } else if (arg == "--temp") {
            options.temp = parse_float(take("--temp"), "--temp");
        } else if (arg == "--top-p") {
            options.top_p = parse_float(take("--top-p"), "--top-p");
        } else if (arg == "--top-k") {
            options.top_k = parse_u32(take("--top-k"), "--top-k");
        } else if (arg == "--seed") {
            options.seed = parse_u32(take("--seed"), "--seed");
        } else if (arg == "-lw" || arg == "--layer-window" || arg == "--n-layer-window") {
            options.layer_window = parse_layer_window(take(arg.c_str()));
        } else if (arg == "-gm" || arg == "--gpu-mem") {
            options.gpu_mem_gib = parse_positive_double(take(arg.c_str()), arg.c_str());
        } else if (arg == "-k" || arg == "--n-cycles") {
            options.k_override = parse_positive_i32(take(arg.c_str()), arg.c_str());
        } else if (arg == "--master-priority") {
            options.master_priority = parse_positive_double(
                take("--master-priority"), "--master-priority");
        } else if (arg == "--prefetch") {
            options.prefetch = parse_prefetch(take("--prefetch"));
        } else if (arg == "--force") {
            options.force = true;
        } else if (arg == "--spec-draft-model" || arg == "-md" ||
                   arg == "--model-draft") {
            options.spec_draft_model = take(arg.c_str());
        } else if (arg == "--spec-type") {
            options.spec_types = parse_spec_types(take("--spec-type"));
        } else if (arg == "--spec-draft-n-max") {
            options.spec_draft_n_max = parse_u32(
                take("--spec-draft-n-max"), "--spec-draft-n-max");
        } else {
            throw std::runtime_error("unknown option '" + arg + "' (use --help)");
        }
    }
    if (options.model_path.empty() == options.hf_repo.empty()) {
        throw std::runtime_error("need exactly one of -m MODEL or -hf REPO");
    }
    if (options.n_ctx == 0) {
        throw std::runtime_error("-c must be greater than zero");
    }
    if (options.temp < 0.0f) {
        throw std::runtime_error("--temp must not be negative");
    }
    if (options.top_p < 0.0f || options.top_p > 1.0f) {
        throw std::runtime_error("--top-p must be between 0 and 1");
    }
    if (options.force) {
        options.prefetch = potluck::prefetch_mode::force;
    }
    return options;
}


std::string prompt_for_turn(const std::string & text, bool conversation,
                            common_chat_templates_ptr & templates,
                            std::vector<common_chat_msg> & history) {
    if (!conversation) {
        return text;
    }
    common_chat_msg message;
    message.role = "user";
    message.content = text;
    const std::string formatted = common_chat_format_single(
        templates.get(), history, message, true, false);
    history.push_back(std::move(message));
    return formatted.empty() ? text : formatted;
}

} // namespace

int main(int argc, char ** argv) {
    try {
        cli_options cli = parse_options(argc, argv);
        if (!cli.hf_repo.empty()) {
            cli.model_path = resolve_hf_model(cli.hf_repo);
        }

        llama_backend_init();
        llama_model_params model_params = llama_model_default_params();
        model_params.no_alloc = true;
        model_params.load_mode = LLAMA_LOAD_MODE_NONE;
        model_params.check_tensors = false;
        llama_model * metadata_model = llama_model_load_from_file(cli.model_path.c_str(), model_params);
        if (metadata_model == nullptr) {
            throw std::runtime_error("cannot load model metadata from " + cli.model_path);
        }
        std::unique_ptr<llama_model, decltype(&llama_model_free)> metadata(
            metadata_model, llama_model_free);

        const llama_vocab * vocab = llama_model_get_vocab(metadata_model);
        const uint32_t n_layer = static_cast<uint32_t>(llama_model_n_layer(metadata_model));
        if (n_layer == 0) {
            throw std::runtime_error("model metadata reports zero layers");
        }
        validate_manual_workload(cli.layer_window, cli.k_override, n_layer);
        uint32_t head_dim = 0;
        char key_length[64] = {};
        if (llama_model_meta_val_str(metadata_model, "attention.key_length", key_length,
                                     sizeof(key_length)) > 0) {
            head_dim = static_cast<uint32_t>(std::atof(key_length));
        }
        if (head_dim == 0) {
            const int32_t n_head = llama_model_n_head(metadata_model);
            head_dim = n_head > 0
                ? static_cast<uint32_t>(llama_model_n_embd(metadata_model) / n_head) : 0;
        }

        const std::string worker_path = executable_dir(argv[0]) + "/potluck-worker";
        const std::filesystem::path adjacent_root =
            std::filesystem::path(worker_path).parent_path().parent_path().parent_path();
        std::filesystem::path stage_dir = std::filesystem::current_path() / "dist/mac-arm64";
        if (!std::filesystem::is_regular_file(stage_dir / "potluck-build-id")) {
            stage_dir = adjacent_root / "dist/mac-arm64";
        }
        std::vector<bootstrap_node> bootstrap_nodes;
        model_digest_cache digest_cache;
        ring_startup_options startup;
        startup.model_path = cli.model_path;
        startup.model_name = basename_of(cli.model_path);
        startup.worker_path = worker_path;
        startup.host = "0.0.0.0";
        startup.head_share = "auto";
        startup.local_platform = first_command_line("uname -sm 2>/dev/null");
        startup.adjacent_root = adjacent_root;
        startup.stage_dir = stage_dir;
        startup.has_staged_payload = std::filesystem::is_regular_file(
            stage_dir / "potluck-build-id");
        startup.bootstrap_nodes = &bootstrap_nodes;
        startup.digest_cache = &digest_cache;
        startup.worker_local = 2;
        startup.n_layer = n_layer;
        startup.head_dim = head_dim;
        startup.n_head_kv = llama_model_n_head_kv(metadata_model);
        startup.n_ctx = cli.n_ctx;
        startup.n_seq_max = cli.n_seq_max;
        startup.n_ubatch = cli.n_ubatch;
        startup.speculative_head_reserve = potluck_speculative_head_reserve(
            startup.n_head_kv, startup.head_dim, startup.n_ctx, startup.n_layer,
            startup.n_seq_max, cli.spec_types, cli.spec_draft_model);
        startup.seed = cli.seed;
        startup.temp = cli.temp;
        startup.top_p = cli.top_p;
        startup.prefetch = cli.prefetch;
        startup.layer_window = cli.layer_window;
        startup.gpu_mem_gib = cli.gpu_mem_gib;
        startup.k_override = cli.k_override;
        startup.master_priority = cli.master_priority;
        log_manual_overrides(cli.layer_window, cli.gpu_mem_gib,
                             cli.k_override, cli.master_priority, cli.prefetch);

        ring_session session;
        slot_speculative_config speculative;
        speculative.draft_model = cli.spec_draft_model;
        speculative.types = cli.spec_types;
        speculative.n_draft = cli.spec_draft_n_max;
        speculative.n_ctx = cli.n_ctx;
        speculative.n_batch = cli.n_ubatch;
        speculative.n_ubatch = cli.n_ubatch;
        std::string startup_error;
        slot_scheduler scheduler(
            session.ring, vocab, cli.n_seq_max, startup.n_ubatch,
            [&](std::string & error) { return rebuild_ring(session, startup, false, error); },
            [&](std::string & error) { return heartbeat_ring(session, error); },
            [&](std::string & error) { return refresh_ring_if_needed(session, startup, error); },
            metadata_model, std::move(speculative));
        if (!bring_up_ring(session, startup, startup_error)) {
            throw std::runtime_error(
                "ring startup failed: " +
                (startup_error.empty() ? std::string("unknown error") : startup_error));
        }
        scheduler.start();
        common_chat_templates_ptr chat_templates = common_chat_templates_init(metadata_model, "");
        std::vector<common_chat_msg> history;
        std::string pending_prompt = cli.prompt;
        uint64_t turn = 0;
        for (;;) {
            if (pending_prompt.empty()) {
                if (cli.conversation) {
                    std::printf("\n> ");
                    std::fflush(stdout);
                }
                if (!std::getline(std::cin, pending_prompt)) {
                    break;
                }
                if (pending_prompt.empty()) {
                    continue;
                }
            }
            const std::string rendered_prompt = prompt_for_turn(
                pending_prompt, cli.conversation, chat_templates, history);
            const std::vector<llama_token> prompt_tokens = tokenize_prompt(vocab, rendered_prompt);
            potluck::slot_config sampling;
            sampling.temp = cli.temp;
            sampling.top_p = cli.top_p;
            sampling.top_k = cli.top_k;
            sampling.seed = cli.seed;
            const std::string request_id = "cli-" + std::to_string(turn);
            const auto slots = scheduler.acquire_many(
                prompt_tokens, cli.n_predict, sampling, {}, true, cli.conversation,
                request_id, turn++, 1);
            if (slots.size() != 1) {
                throw std::runtime_error("cannot acquire a ring conversation slot");
            }
            const std::shared_ptr<scheduled_slot> & slot = slots.front();
            for (;;) {
                std::string piece;
                if (!scheduler.take_piece(slot, piece)) {
                    break;
                }
                std::cout << piece << std::flush;
            }
            scheduler.wait_done(slot);
            std::string generated;
            std::string slot_error;
            {
                std::lock_guard<std::mutex> lock(slot->mutex);
                generated = slot->generated_text;
                slot_error = slot->error;
            }
            scheduler.release(slot);
            if (!slot_error.empty()) {
                throw std::runtime_error(slot_error);
            }
            if (cli.conversation) {
                common_chat_msg assistant;
                assistant.role = "assistant";
                assistant.content = std::move(generated);
                history.push_back(std::move(assistant));
                pending_prompt.clear();
                std::cout << std::endl;
                continue;
            }
            std::cout << std::endl;
            break;
        }

        session.stopping.store(true, std::memory_order_release);
        scheduler.request_stop();
        scheduler.stop();
        if (!session.ring.controls.empty()) {
            std::string reset_error;
            (void) reset_ring_workers(session.ring, reset_error);
        }
        stop_planned_workers(session.workers);
        return 0;
    } catch (const std::exception & exception) {
        std::fprintf(stderr, "potluck-cli: %s\n", exception.what());
        return 1;
    }
}

// potluck-server HTTP and OpenAI-compatible API routes.

#include "internal.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

uint32_t json_u32(const json & object, const char * key, uint32_t fallback) {
    if (!object.contains(key)) {
        return fallback;
    }
    const json & value = object.at(key);
    if (value.is_null()) {
        return fallback;
    }
    uint64_t number = 0;
    if (value.is_number_unsigned()) {
        number = value.get<uint64_t>();
    } else if (value.is_number_integer()) {
        const int64_t signed_number = value.get<int64_t>();
        if (signed_number < 0) {
            throw std::runtime_error(std::string("invalid ") + key + ": expected a non-negative integer");
        }
        number = static_cast<uint64_t>(signed_number);
    } else {
        throw std::runtime_error(std::string("invalid ") + key + ": expected a non-negative integer");
    }
    if (number > UINT32_MAX) {
        throw std::runtime_error(std::string("invalid ") + key + ": value is too large");
    }
    return static_cast<uint32_t>(number);
}

int32_t json_i32(const json & object, const char * key, int32_t fallback) {
    if (!object.contains(key)) {
        return fallback;
    }
    const json & value = object.at(key);
    if (value.is_null()) {
        return fallback;
    }
    int64_t number = 0;
    if (value.is_number_integer()) {
        number = value.get<int64_t>();
    } else if (value.is_number_unsigned()) {
        const uint64_t unsigned_number = value.get<uint64_t>();
        if (unsigned_number > static_cast<uint64_t>(INT32_MAX)) {
            throw std::runtime_error(std::string("invalid ") + key + ": value is out of range");
        }
        number = static_cast<int64_t>(unsigned_number);
    } else {
        throw std::runtime_error(std::string("invalid ") + key + ": expected an integer");
    }
    if (number < INT32_MIN || number > INT32_MAX) {
        throw std::runtime_error(std::string("invalid ") + key + ": value is out of range");
    }
    return static_cast<int32_t>(number);
}

bool json_bool(const json & object, const char * key, bool fallback) {
    if (!object.contains(key)) {
        return fallback;
    }
    if (object.at(key).is_null()) {
        return fallback;
    }
    if (!object.at(key).is_boolean()) {
        throw std::runtime_error(std::string("invalid ") + key + ": expected a boolean");
    }
    return object.at(key).get<bool>();
}

float json_f32(const json & object, const char * key, float fallback) {
    if (!object.contains(key)) {
        return fallback;
    }
    if (object.at(key).is_null()) {
        return fallback;
    }
    if (!object.at(key).is_number()) {
        throw std::runtime_error(std::string("invalid ") + key + ": expected a number");
    }
    return object.at(key).get<float>();
}

std::vector<std::string> json_stops(const json & object) {
    if (!object.contains("stop")) {
        return {};
    }
    const json & value = object.at("stop");
    if (value.is_null()) {
        return {};
    }
    if (value.is_string()) {
        return { value.get<std::string>() };
    }
    if (!value.is_array()) {
        throw std::runtime_error("invalid stop: expected a string or an array of strings");
    }
    if (value.size() > 4) {
        throw std::runtime_error("invalid stop: at most 4 stop strings are supported");
    }
    std::vector<std::string> stops;
    for (const json & item : value) {
        if (!item.is_string()) {
            throw std::runtime_error("invalid stop: expected an array of strings");
        }
        stops.push_back(item.get<std::string>());
    }
    return stops;
}

std::string stop_text(const std::string & text, const std::vector<std::string> & stops) {
    size_t end = text.size();
    for (const std::string & stop : stops) {
        if (!stop.empty()) {
            const size_t pos = text.find(stop);
            if (pos != std::string::npos) {
                end = std::min(end, pos);
            }
        }
    }
    return text.substr(0, end);
}
size_t visible_token_count(const llama_vocab * vocab,
                           const std::vector<llama_token> & generated,
                           size_t stop_offset) {
    if (stop_offset == std::string::npos) {
        return generated.size();
    }
    size_t offset = 0;
    size_t count = 0;
    for (const llama_token token : generated) {
        const std::string piece = token_piece(vocab, token);
        if (offset + piece.size() > stop_offset) {
            break;
        }
        offset += piece.size();
        ++count;
    }
    return count;
}

json utf8_bytes(const std::string & text) {
    json bytes = json::array();
    for (const unsigned char byte : text) {
        bytes.push_back(static_cast<uint32_t>(byte));
    }
    return bytes;
}


std::string stream_text(const std::string & text, const std::vector<std::string> & stops) {
    const std::string limited = stop_text(text, stops);
    if (limited.size() != text.size()) {
        return limited;
    }
    size_t hold = 0;
    for (const std::string & stop : stops) {
        const size_t max_prefix = std::min(stop.size(), text.size());
        for (size_t length = max_prefix; length > hold; --length) {
            if (text.compare(text.size() - length, length, stop, 0, length) == 0) {
                hold = length;
                break;
            }
        }
    }
    return text.substr(0, text.size() - hold);
}
json chat_diff_to_delta(const common_chat_msg_diff & diff) {
    json delta = json::object();
    if (!diff.reasoning_content_delta.empty()) {
        delta["reasoning_content"] = diff.reasoning_content_delta;
    }
    if (!diff.content_delta.empty()) {
        delta["content"] = diff.content_delta;
    }
    if (diff.tool_call_index != std::string::npos) {
        json tool_call = {
            { "index", diff.tool_call_index },
        };
        if (!diff.tool_call_delta.id.empty()) {
            tool_call["id"] = diff.tool_call_delta.id;
            tool_call["type"] = "function";
        }
        if (!diff.tool_call_delta.name.empty() || !diff.tool_call_delta.arguments.empty()) {
            json function = json::object();
            if (!diff.tool_call_delta.name.empty()) {
                function["name"] = diff.tool_call_delta.name;
            }
            if (!diff.tool_call_delta.arguments.empty()) {
                function["arguments"] = diff.tool_call_delta.arguments;
            }
            tool_call["function"] = std::move(function);
        }
        delta["tool_calls"] = json::array({ std::move(tool_call) });
    }
    return delta;
}


json usage_json(size_t prompt_tokens, size_t completion_tokens) {
    return json{
        { "prompt_tokens", prompt_tokens },
        { "completion_tokens", completion_tokens },
        { "total_tokens", prompt_tokens + completion_tokens },
    };
}

json error_json(const std::string & message, const std::string & type = "invalid_request_error") {
    return json{ { "error", {
        { "message", message }, { "type", type }, { "param", nullptr }, { "code", nullptr }
    } } };
}

bool constant_time_equal(const std::string & actual, const std::string & expected) {
    constexpr size_t comparison_limit = 8192;
    volatile uint64_t mismatch = static_cast<uint64_t>(actual.size() ^ expected.size());
    for (size_t i = 0; i < comparison_limit; ++i) {
        const unsigned char actual_byte = i < actual.size()
            ? static_cast<unsigned char>(actual[i]) : 0;
        const unsigned char expected_byte = i < expected.size()
            ? static_cast<unsigned char>(expected[i]) : 0;
        mismatch |= static_cast<uint64_t>(actual_byte ^ expected_byte);
    }
    return actual.size() <= comparison_limit &&
           expected.size() <= comparison_limit &&
           mismatch == 0;
}


std::atomic<uint64_t> request_counter{0};

std::string request_id(uint64_t & created) {
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    created = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::seconds>(now).count());
    const uint64_t sequence = request_counter.fetch_add(1, std::memory_order_relaxed);
    return "chatcmpl-potluck-" + std::to_string(created) + "-" + std::to_string(sequence);
}

json chat_logprobs_json(const llama_vocab * vocab,
                        const std::vector<llama_token> & generated,
                        const std::vector<std::vector<potluck::token_logprob>> & all_logprobs,
                        uint32_t top_logprobs,
                        size_t limit = std::string::npos) {
    json content = json::array();
    for (size_t i = 0; i < generated.size() && i < limit; ++i) {
        json top = json::array();
        float selected = 0.0f;
        if (i < all_logprobs.size()) {
            size_t count = 0;
            for (const auto & item : all_logprobs[i]) {
                if (item.token == generated[i]) {
                    selected = item.logprob;
                }
                if (count++ < top_logprobs) {
                    const std::string item_piece =
                        token_piece(vocab, static_cast<llama_token>(item.token));
                    top.push_back({ { "token", item_piece },
                                    { "bytes", utf8_bytes(item_piece) },
                                    { "logprob", item.logprob } });
                }
            }
        }
        const std::string item_piece = token_piece(vocab, generated[i]);
        content.push_back({ { "token", item_piece },
                            { "bytes", utf8_bytes(item_piece) },
                            { "logprob", selected }, { "top_logprobs", top } });
    }
    return json{ { "content", std::move(content) } };
}

json text_logprobs_json(const llama_vocab * vocab,
                        const std::vector<llama_token> & generated,
                        const std::vector<std::vector<potluck::token_logprob>> & all_logprobs,
                        uint32_t top_logprobs,
                        size_t limit = std::string::npos) {
    json tokens = json::array();
    json token_logprobs = json::array();
    json top = json::array();
    json text_offset = json::array();
    size_t offset = 0;
    for (size_t i = 0; i < generated.size() && i < limit; ++i) {
        const std::string piece = token_piece(vocab, generated[i]);
        float selected = 0.0f;
        json candidates = json::object();
        if (i < all_logprobs.size()) {
            size_t count = 0;
            for (const auto & item : all_logprobs[i]) {
                if (item.token == generated[i]) {
                    selected = item.logprob;
                }
                if (count++ < top_logprobs) {
                    candidates[token_piece(vocab, static_cast<llama_token>(item.token))] = item.logprob;
                }
            }
        }
        tokens.push_back(piece);
        token_logprobs.push_back(selected);
        top.push_back(std::move(candidates));
        text_offset.push_back(offset);
        offset += piece.size();
    }
    return {
        { "tokens", std::move(tokens) },
        { "token_logprobs", std::move(token_logprobs) },
        { "top_logprobs", std::move(top) },
        { "text_offset", std::move(text_offset) }
    };
}

} // namespace


void setup_http_routes(httplib::Server & server, ring_session & session,
                       slot_scheduler & scheduler, const llama_vocab * vocab,
                       const std::string & model_name,
                       common_chat_templates_ptr & chat_templates,
                       uint32_t n_predict_default, float temp, float top_p, uint32_t seed,
                       const std::string & api_key, const std::string & cors_origin) {
        const auto set_common_headers = [](httplib::Response & response) {
            response.set_header("Cache-Control", "no-cache");
        };
        const auto set_cors_header = [cors_origin](const httplib::Request & request,
                                                   httplib::Response & response) {
            if (!cors_origin.empty() &&
                request.get_header_value("Origin") == cors_origin) {
                response.set_header("Access-Control-Allow-Origin", cors_origin);
            }
        };
        const std::string expected_authorization = "Bearer " + api_key;
        server.set_pre_routing_handler(
            [api_key, expected_authorization, set_common_headers, set_cors_header](
                const httplib::Request & request, httplib::Response & response) {
                set_common_headers(response);
                set_cors_header(request, response);
                if (!api_key.empty()) {
                    const std::string authorization =
                        request.get_header_value("Authorization");
                    if (!constant_time_equal(authorization, expected_authorization)) {
                        response.status = 401;
                        response.set_content(
                            error_json("Invalid API Key", "authentication_error").dump(),
                            "application/json");
                        return httplib::Server::HandlerResponse::Handled;
                    }
                }
                if (request.method == "OPTIONS") {
                    response.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
                    response.set_header("Access-Control-Allow-Headers",
                                         "Content-Type, Authorization");
                    response.status = 204;
                    return httplib::Server::HandlerResponse::Handled;
                }
                return httplib::Server::HandlerResponse::Unhandled;
            });
        server.Get("/health", [&session, &scheduler, set_common_headers](const httplib::Request &, httplib::Response & response) {
            const bool recovery_exhausted = scheduler.recovery_exhausted();
            const bool ring_rebuilding = scheduler.rebuilding();
            const bool rebuilding = ring_rebuilding && !recovery_exhausted;
            const std::string recovery_error = scheduler.recovery_error();
            json slots = scheduler.health();
            std::lock_guard<std::mutex> ring_lock(session.mutex);
            const bool ready = session.healthy && !ring_rebuilding && !recovery_exhausted;
            const bool loading = !ready && !rebuilding && !recovery_exhausted;
            const char * status = recovery_exhausted ? "failed"
                                : rebuilding ? "rebuilding"
                                : loading ? "loading" : "ok";
            json health = { { "status", status },
                            { "loading", loading }, { "ready", ready },
                            { "rebuilding", rebuilding },
                            { "workers", session.ring.workers.size() },
                            { "windows", json::array() }, { "slots", std::move(slots) } };
            if (std::string(status) != "ok") {
                health["reason"] = !recovery_error.empty() ? recovery_error
                                    : session.health_reason.empty() ? "ring recovery in progress"
                                                                    : session.health_reason;
            }
            for (size_t i = 0; i < session.ring.windows.size(); ++i) {
                const potluck::ring_window & window = session.ring.windows[i];
                const ring_worker & worker = session.ring.workers[window.owner];
                health["windows"].push_back(json{
                    { "index", i }, { "owner", window.owner },
                    { "host", worker.address.host }, { "port", worker.address.port },
                    { "start", window.start }, { "end", window.end },
                    { "n_gpu_layers", window.n_gpu_layers }
                });
            }
            set_common_headers(response);
            response.set_content(health.dump(), "application/json");
        });
        server.Get("/props", [&session, &scheduler, &chat_templates, model_name, set_common_headers](const httplib::Request &, httplib::Response & response) {
            const bool recovery_exhausted = scheduler.recovery_exhausted();
            const bool ring_rebuilding = scheduler.rebuilding();
            const bool rebuilding = ring_rebuilding && !recovery_exhausted;
            std::lock_guard<std::mutex> ring_lock(session.mutex);
            const bool ready = session.healthy && !ring_rebuilding && !recovery_exhausted;
            const bool loading = !ready && !rebuilding && !recovery_exhausted;
            const char * status = recovery_exhausted ? "failed"
                                : rebuilding ? "rebuilding"
                                : loading ? "loading" : "ok";
            json props = {
                { "model", model_name }, { "model_name", model_name },
                { "model_alias", model_name }, { "status", status },
                { "loading", loading }, { "ready", ready },
                { "rebuilding", rebuilding }, { "slot_count", scheduler.slot_count() },
                { "slots", scheduler.health() },
            };
            if (chat_templates) {
                props["chat_template_caps"] = common_chat_templates_get_caps(chat_templates.get());
            }
            if (!session.health_reason.empty()) {
                props["reason"] = session.health_reason;
            }
            set_common_headers(response);
            response.set_content(props.dump(), "application/json");
        });
        server.Get("/v1/models", [model_name, set_common_headers](const httplib::Request &, httplib::Response & response) {
            json result = { { "object", "list" }, { "data", json::array({ json{
                { "id", model_name }, { "object", "model" }, { "owned_by", "potluck" }
            } }) } };
            set_common_headers(response);
            response.set_content(result.dump(), "application/json");
        });

        auto handle = [vocab, model_name, &chat_templates, &session, &scheduler, n_predict_default, temp, top_p, seed, set_common_headers](const httplib::Request & request, httplib::Response & response, bool chat, bool openai_text) {
            std::vector<std::shared_ptr<scheduled_slot>> slots;
            std::shared_ptr<scheduled_slot> slot;
            bool healthy = false;
            {
                std::lock_guard<std::mutex> lock(session.mutex);
                healthy = session.healthy;
            }
            if (!healthy) {
                std::string message = "model is loading";
                if (scheduler.recovery_exhausted()) {
                    message = "cluster recovery failed: " + scheduler.recovery_error();
                } else if (scheduler.rebuilding()) {
                    message = "cluster is rebuilding; retry";
                }
                response.status = 503;
                set_common_headers(response);
                response.set_content(error_json(message, "server_error").dump(), "application/json");
                return;
            }
            try {
                json req;
                try {
                    req = json::parse(request.body);
                } catch (const std::exception & e) {
                    throw std::runtime_error(std::string("invalid JSON: ") + e.what());
                }
                if (!req.is_object()) {
                    throw std::runtime_error("invalid JSON: request must be an object");
                }
                static const std::unordered_set<std::string> allowed = {
                    "model", "messages", "prompt", "max_tokens", "max_completion_tokens", "n_predict",
                    "n", "stop", "stream", "stream_options", "reasoning_effort", "temperature", "top_p",
                    "top_k", "min_p", "seed", "presence_penalty", "frequency_penalty", "repeat_penalty",
                    "repeat_last_n", "logprobs", "top_logprobs", "tools", "tool_choice",
                    "parallel_tool_calls", "preserve_thinking", "chat_template_kwargs"
                };
                for (auto it = req.begin(); it != req.end(); ++it) {
                    if (allowed.find(it.key()) == allowed.end()) {
                        throw std::runtime_error("unsupported field: " + it.key());
                    }
                }
                if (req.contains("model")) {
                    if (!req["model"].is_string()) {
                        throw std::runtime_error("invalid model: expected a string");
                    }
                    if (req["model"].get<std::string>() != model_name) {
                        throw std::runtime_error("invalid model: model does not match the loaded model");
                    }
                }
                if (req.contains("reasoning_effort") && !req["reasoning_effort"].is_string()) {
                    throw std::runtime_error("invalid reasoning_effort: expected a string");
                }
                if (chat && req.contains("prompt")) {
                    throw std::runtime_error("invalid prompt: not supported for chat completions");
                }
                if (!chat && req.contains("messages")) {
                    throw std::runtime_error("invalid messages: not supported for completions");
                }
                if (!chat && (req.contains("tools") || req.contains("tool_choice") ||
                              req.contains("parallel_tool_calls") ||
                              req.contains("preserve_thinking") ||
                              req.contains("chat_template_kwargs"))) {
                    throw std::runtime_error("invalid chat template fields: require chat completions");
                }
                const bool stream = json_bool(req, "stream", false);
                bool include_usage = false;
                if (req.contains("stream_options")) {
                    if (!req["stream_options"].is_object()) {
                        throw std::runtime_error("invalid stream_options: expected an object");
                    }
                    for (auto it = req["stream_options"].begin(); it != req["stream_options"].end(); ++it) {
                        if (it.key() != "include_usage") {
                            throw std::runtime_error("unsupported stream_options field: " + it.key());
                        }
                    }
                    include_usage = json_bool(req["stream_options"], "include_usage", false);
                }
                const std::vector<std::string> stops = json_stops(req);
                const uint32_t n = json_u32(req, "n", 1);
                if (n == 0 || n > scheduler.slot_count()) {
                    throw std::runtime_error("invalid n: must be between 1 and the available slot count");
                }
                if (stream && n > 1) {
                    throw std::runtime_error("invalid n: streaming fan-out supports only n=1");
                }
                if (!chat && !openai_text && n > 1) {
                    throw std::runtime_error("invalid n: legacy completion supports only n=1");
                }

                std::string prompt_text;
                uint32_t n_predict = n_predict_default;
                common_chat_params chat_params;
                bool have_chat_params = false;
                std::vector<common_chat_tool> tools;
                bool parse_stream_output = false;
                if (chat) {
                    if (!req.contains("messages")) {
                        throw std::runtime_error("missing messages");
                    }
                    if (!req["messages"].is_array() || req["messages"].empty()) {
                        throw std::runtime_error("invalid messages: expected a non-empty array");
                    }
                    common_chat_templates_inputs inputs;
                    if (req.contains("preserve_thinking") &&
                        !req["preserve_thinking"].is_boolean()) {
                        throw std::runtime_error("invalid preserve_thinking: expected a boolean");
                    }
                    if (req.contains("chat_template_kwargs")) {
                        if (!req["chat_template_kwargs"].is_object()) {
                            throw std::runtime_error(
                                "invalid chat_template_kwargs: expected an object");
                        }
                        for (auto it = req["chat_template_kwargs"].begin();
                             it != req["chat_template_kwargs"].end(); ++it) {
                            inputs.chat_template_kwargs[it.key()] = it.value().dump();
                        }
                    }
                    if (req.contains("preserve_thinking")) {
                        inputs.chat_template_kwargs["preserve_thinking"] =
                            req["preserve_thinking"].dump();
                    }
                    inputs.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
                    try {
                        inputs.messages = common_chat_msgs_parse_oaicompat(req["messages"]);
                    } catch (const std::exception & e) {
                        throw std::runtime_error(std::string("invalid messages: ") + e.what());
                    }
                    if (req.contains("tools")) {
                        if (!req["tools"].is_array()) {
                            throw std::runtime_error("invalid tools: expected an array");
                        }
                        try {
                            tools = common_chat_tools_parse_oaicompat(req["tools"]);
                        } catch (const std::exception & e) {
                            throw std::runtime_error(std::string("invalid tools: ") + e.what());
                        }
                        inputs.tools = tools;
                    }
                    if (req.contains("tool_choice")) {
                        const json & choice = req["tool_choice"];
                        if (choice.is_string()) {
                            const std::string choice_name = choice.get<std::string>();
                            if (choice_name == "required" && tools.empty()) {
                                throw std::runtime_error(
                                    "invalid tool_choice: required needs a non-empty tools array");
                            }
                            try {
                                inputs.tool_choice =
                                    common_chat_tool_choice_parse_oaicompat(choice_name);
                            } catch (const std::exception & e) {
                                throw std::runtime_error(std::string("invalid tool_choice: ") + e.what());
                            }
                        } else if (choice.is_object() &&
                                   choice.value("type", std::string()) == "function" &&
                                   choice.contains("function") && choice["function"].is_object() &&
                                   choice["function"].contains("name") && choice["function"]["name"].is_string()) {
                            if (tools.empty()) {
                                throw std::runtime_error(
                                    "invalid tool_choice: function needs a non-empty tools array");
                            }
                            const std::string name = choice["function"]["name"].get<std::string>();
                            const auto selected = std::find_if(tools.begin(), tools.end(),
                                [&](const common_chat_tool & tool) { return tool.name == name; });
                            if (selected == tools.end()) {
                                throw std::runtime_error("invalid tool_choice: selected function is not in tools");
                            }
                            inputs.tools = { *selected };
                            inputs.tool_choice = COMMON_CHAT_TOOL_CHOICE_REQUIRED;
                        } else {
                            throw std::runtime_error("invalid tool_choice: expected a string or function object");
                        }
                    }
                    const auto caps = chat_templates ? common_chat_templates_get_caps(chat_templates.get())
                                                      : std::map<std::string, bool>{};
                    inputs.parallel_tool_calls = json_bool(
                        req, "parallel_tool_calls",
                        caps.count("supports_parallel_tool_calls") != 0 &&
                        caps.at("supports_parallel_tool_calls"));
                    if (req.contains("reasoning_effort")) {
                        const std::string effort = req["reasoning_effort"].get<std::string>();
                        if (effort == "none") {
                            inputs.enable_thinking = false;
                            inputs.chat_template_kwargs["enable_thinking"] = "false";
                        } else if (!effort.empty()) {
                            inputs.chat_template_kwargs["reasoning_effort"] = json(effort).dump();
                        }
                    }
                    if (!chat_templates) {
                        throw std::runtime_error("model has no chat template");
                    }
                    try {
                        chat_params = common_chat_templates_apply(chat_templates.get(), inputs);
                    } catch (const std::exception & e) {
                        throw std::runtime_error(std::string("invalid chat template input: ") + e.what());
                    }
                    parse_stream_output = !tools.empty() &&
                        inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE;
                    have_chat_params = true;
                    prompt_text = chat_params.prompt;
                } else {
                    if (!req.contains("prompt")) {
                        throw std::runtime_error("missing prompt");
                    }
                    if (!req["prompt"].is_string()) {
                        throw std::runtime_error("invalid prompt: expected a string");
                    }
                    prompt_text = req["prompt"].get<std::string>();
                    if (prompt_text.empty()) {
                        throw std::runtime_error("prompt is empty");
                    }
                }
                if (req.contains("max_tokens")) {
                    n_predict = json_u32(req, "max_tokens", n_predict);
                }
                if (req.contains("max_completion_tokens")) {
                    n_predict = json_u32(req, "max_completion_tokens", n_predict);
                }
                if (req.contains("n_predict")) {
                    n_predict = json_u32(req, "n_predict", n_predict);
                }
                const std::vector<llama_token> prompt = tokenize_prompt(vocab, prompt_text);
                potluck::slot_config sampling;
                sampling.temp = temp;
                sampling.top_p = top_p;
                sampling.seed = seed;
                if (req.contains("temperature") && !req["temperature"].is_null()) {
                    if (!req["temperature"].is_number()) {
                        throw std::runtime_error("invalid temperature: expected a number");
                    }
                    sampling.temp = req["temperature"].get<float>();
                }
                if (!std::isfinite(sampling.temp) || sampling.temp < 0.0f) {
                    throw std::runtime_error("invalid temperature: expected a finite non-negative number");
                }
                if (req.contains("top_p") && !req["top_p"].is_null()) {
                    if (!req["top_p"].is_number()) {
                        throw std::runtime_error("invalid top_p: expected a number");
                    }
                    sampling.top_p = req["top_p"].get<float>();
                }
                if (!std::isfinite(sampling.top_p) || sampling.top_p < 0.0f || sampling.top_p > 1.0f) {
                    throw std::runtime_error("invalid top_p: expected a number from 0 to 1");
                }
                sampling.top_k = json_u32(req, "top_k", 0);
                if (sampling.top_k > static_cast<uint32_t>(INT32_MAX)) {
                    throw std::runtime_error("invalid top_k: out of range");
                }
                sampling.min_p = json_f32(req, "min_p", 0.0f);
                if (!std::isfinite(sampling.min_p) || sampling.min_p < 0.0f || sampling.min_p > 1.0f) {
                    throw std::runtime_error("invalid min_p: expected a number from 0 to 1");
                }
                sampling.seed = json_u32(req, "seed", sampling.seed);
                sampling.presence_penalty = json_f32(req, "presence_penalty", 0.0f);
                sampling.frequency_penalty = json_f32(req, "frequency_penalty", 0.0f);
                sampling.repeat_penalty = json_f32(req, "repeat_penalty", 1.0f);
                sampling.penalty_last_n = json_i32(req, "repeat_last_n", sampling.penalty_last_n);
                if (!std::isfinite(sampling.presence_penalty) || !std::isfinite(sampling.frequency_penalty) ||
                    !std::isfinite(sampling.repeat_penalty) || sampling.repeat_penalty <= 0.0f ||
                    sampling.penalty_last_n < -1) {
                    throw std::runtime_error("invalid penalty: expected finite sampler values");
                }
                if (openai_text && req.contains("logprobs") && !req["logprobs"].is_null()) {
                    const json & logprobs = req["logprobs"];
                    if (logprobs.is_boolean()) {
                        sampling.logprobs = logprobs.get<bool>();
                        sampling.top_logprobs = json_u32(req, "top_logprobs", 0);
                    } else {
                        sampling.top_logprobs = json_u32(req, "logprobs", 0);
                        sampling.logprobs = true;
                    }
                    if (sampling.top_logprobs > 5) {
                        throw std::runtime_error("invalid logprobs: expected an integer from 0 to 5");
                    }
                    if (sampling.top_logprobs > 0 && !sampling.logprobs) {
                        throw std::runtime_error("invalid top_logprobs: logprobs must be true");
                    }
                } else {
                    sampling.logprobs = json_bool(req, "logprobs", false);
                    sampling.top_logprobs = json_u32(req, "top_logprobs", 0);
                    if (sampling.top_logprobs > 20) {
                        throw std::runtime_error("invalid top_logprobs: expected an integer from 0 to 20");
                    }
                    if (sampling.top_logprobs > 0 && !sampling.logprobs) {
                        throw std::runtime_error("invalid top_logprobs: logprobs must be true");
                    }
                }
                if (stream && sampling.logprobs) {
                    throw std::runtime_error("unsupported combination: streaming logprobs");
                }
                uint64_t created = 0;
                const std::string id = request_id(created);
                slots = scheduler.acquire_many(
                    prompt, n_predict, sampling, stops, stream, chat, id, created, n);
                if (slots.empty()) {
                    response.status = 503;
                    std::string message = "all conversation slots are busy";
                    if (scheduler.recovery_exhausted()) {
                        message = "cluster recovery failed: " + scheduler.recovery_error();
                    } else if (scheduler.rebuilding()) {
                        message = "cluster is rebuilding; retry";
                    }
                    set_common_headers(response);
                    response.set_content(error_json(message, "server_error").dump(), "application/json");
                    return;
                }
                slot = slots.front();
                const auto parse_chat_output = [chat_params, have_chat_params](
                        const std::string & text, bool is_partial) {
                    common_chat_msg message;
                    if (!have_chat_params) {
                        return message;
                    }
                    try {
                        common_chat_parser_params parser(chat_params);
                        parser.parse_tool_calls = true;
                        parser.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
                        if (!chat_params.parser.empty()) {
                            parser.parser.load(chat_params.parser);
                        }
                        return common_chat_parse(text, is_partial, parser);
                    } catch (const std::exception &) {
                        return common_chat_msg{};
                    }
                };
                const auto common_chunk = [id, created, model_name, openai_text, include_usage](const json & choice) {
                    json chunk = {
                        { "id", id },
                        { "object", openai_text ? "text_completion" : "chat.completion.chunk" },
                        { "created", created }, { "model", model_name },
                        { "choices", json::array({ choice }) }
                    };
                    if (include_usage) {
                        chunk["usage"] = nullptr;
                    }
                    return chunk.dump();
                };
                if (stream) {
                    scheduler.wait_first(slot);
                    bool initial_cancelled = false;
                    std::string initial_error;
                    {
                        std::lock_guard<std::mutex> lock(slot->mutex);
                        initial_cancelled = slot->cancelled;
                        initial_error = slot->error;
                    }
                    if (initial_cancelled) {
                        scheduler.acknowledge_cancel(slot);
                        scheduler.release(slot);
                        response.status = 503;
                        set_common_headers(response);
                        response.set_content(error_json("request cancelled", "server_error").dump(),
                                             "application/json");
                        return;
                    }
                    if (!initial_error.empty()) {
                        throw std::runtime_error(initial_error);
                    }
                    response.status = 200;
                    set_common_headers(response);
                    response.set_chunked_content_provider(
                        "text/event-stream; charset=utf-8",
                        [&, slot, chat, openai_text, include_usage, parse_stream_output, common_chunk, parse_chat_output, prompt, stops, n_predict, id](size_t, httplib::DataSink & sink) mutable {
                            auto write = [&](const std::string & event) {
                                return sink.write(event.data(), event.size());
                            };
                            auto abort = [&]() {
                                scheduler.cancel(slot);
                                scheduler.acknowledge_cancel(slot);
                                sink.done();
                                return false;
                            };
                            try {
                                if (chat) {
                                    const json role = {
                                        { "index", 0 },
                                        { "delta", { { "role", "assistant" } } },
                                        { "finish_reason", nullptr }
                                    };
                                    if (!write("data: " + common_chunk(role) + "\n\n")) {
                                        return abort();
                                    }
                                }
                                std::string generated_text;
                                std::string emitted_text;
                                common_chat_msg stream_message;
                                std::vector<std::string> stream_tool_call_ids;
                                std::unordered_set<size_t> sent_tool_call_names;
                                std::unordered_map<size_t, std::string> sent_tool_call_name_text;
                                std::unordered_set<size_t> sent_tool_call_arguments;
                                auto set_stream_tool_call_ids = [&](common_chat_msg & message) {
                                    message.set_tool_call_ids(stream_tool_call_ids, [&]() {
                                        return "call_0_" + std::to_string(stream_tool_call_ids.size()) + "_" + id;
                                    });
                                };
                                auto emit_diff = [&](const common_chat_msg_diff & diff) {
                                    const json delta = chat_diff_to_delta(diff);
                                    if (delta.empty()) {
                                        return true;
                                    }
                                    const json choice = {
                                        { "index", 0 }, { "delta", delta }, { "finish_reason", nullptr }
                                    };
                                    return write("data: " + common_chunk(choice) + "\n\n");
                                };
                                auto emit_chat_diffs = [&](const std::vector<common_chat_msg_diff> & all_diffs,
                                                           const common_chat_msg & message, bool is_partial) {
                                    auto emit_tool_call = [&](size_t tool_index, bool include_arguments) {
                                        common_chat_msg_diff header;
                                        header.tool_call_index = tool_index;
                                        header.tool_call_delta.id = message.tool_calls[tool_index].id;
                                        header.tool_call_delta.name = message.tool_calls[tool_index].name;
                                        if (!emit_diff(header)) {
                                            return false;
                                        }
                                        sent_tool_call_names.insert(tool_index);
                                        sent_tool_call_name_text[tool_index] =
                                            message.tool_calls[tool_index].name;
                                        if (include_arguments && !message.tool_calls[tool_index].arguments.empty()) {
                                            common_chat_msg_diff arguments;
                                            arguments.tool_call_index = tool_index;
                                            arguments.tool_call_delta.arguments =
                                                message.tool_calls[tool_index].arguments;
                                            if (!emit_diff(arguments)) {
                                                return false;
                                            }
                                            sent_tool_call_arguments.insert(tool_index);
                                        }
                                        return true;
                                    };
                                    for (const common_chat_msg_diff & original : all_diffs) {
                                        common_chat_msg_diff diff = original;
                                        bool handled = false;
                                        for (size_t tool_index = 0; tool_index < message.tool_calls.size(); ++tool_index) {
                                            if (sent_tool_call_names.count(tool_index) ||
                                                message.tool_calls[tool_index].name.empty()) {
                                                continue;
                                            }
                                            if (diff.tool_call_index != tool_index ||
                                                !diff.tool_call_delta.arguments.empty() || !is_partial) {
                                                const bool include_arguments =
                                                    diff.tool_call_index == tool_index &&
                                                    !diff.tool_call_delta.arguments.empty();
                                                if (!emit_tool_call(tool_index, include_arguments)) {
                                                    return false;
                                                }
                                                if (diff.tool_call_index == tool_index) {
                                                    handled = true;
                                                    break;
                                                }
                                            }
                                        }
                                        if (handled) {
                                            continue;
                                        }
                                        if (diff.tool_call_index == std::string::npos) {
                                            if (!emit_diff(diff)) {
                                                return false;
                                            }
                                        } else {
                                            const size_t tool_index = diff.tool_call_index;
                                            if (sent_tool_call_names.count(tool_index)) {
                                                if (!diff.tool_call_delta.name.empty()) {
                                                    const auto name_it =
                                                        sent_tool_call_name_text.find(tool_index);
                                                    const std::string previous_name =
                                                        name_it == sent_tool_call_name_text.end()
                                                            ? std::string() : name_it->second;
                                                    const std::string & new_name = diff.tool_call_delta.name;
                                                    if (new_name.size() >= previous_name.size() &&
                                                        new_name.compare(0, previous_name.size(),
                                                                         previous_name) == 0) {
                                                        const std::string delta =
                                                            new_name.substr(previous_name.size());
                                                        if (!delta.empty()) {
                                                            common_chat_msg_diff name = diff;
                                                            name.tool_call_delta.name = delta;
                                                            name.tool_call_delta.id.clear();
                                                            name.tool_call_delta.arguments.clear();
                                                            if (!emit_diff(name)) {
                                                                return false;
                                                            }
                                                        }
                                                    }
                                                    sent_tool_call_name_text[tool_index] = new_name;
                                                }
                                                if (sent_tool_call_arguments.count(tool_index) == 0 &&
                                                    !message.tool_calls[tool_index].arguments.empty()) {
                                                    common_chat_msg_diff arguments;
                                                    arguments.tool_call_index = tool_index;
                                                    arguments.tool_call_delta.arguments =
                                                        message.tool_calls[tool_index].arguments;
                                                    if (!emit_diff(arguments)) {
                                                        return false;
                                                    }
                                                    sent_tool_call_arguments.insert(tool_index);
                                                } else if (!diff.tool_call_delta.arguments.empty()) {
                                                    diff.tool_call_delta.name.clear();
                                                    diff.tool_call_delta.id.clear();
                                                    if (!emit_diff(diff)) {
                                                        return false;
                                                    }
                                                    sent_tool_call_arguments.insert(tool_index);
                                                }
                                            } else if (!is_partial &&
                                                       !message.tool_calls[tool_index].name.empty()) {
                                                if (!emit_tool_call(tool_index,
                                                                    !message.tool_calls[tool_index].arguments.empty())) {
                                                    return false;
                                                }
                                            }
                                        }
                                    }
                                    if (!is_partial) {
                                        for (size_t tool_index = 0; tool_index < message.tool_calls.size(); ++tool_index) {
                                            if (!sent_tool_call_names.count(tool_index) &&
                                                !message.tool_calls[tool_index].name.empty()) {
                                                if (!emit_tool_call(tool_index,
                                                                    !message.tool_calls[tool_index].arguments.empty())) {
                                                    return false;
                                                }
                                            }
                                        }
                                    }
                                    return true;
                                };
                                for (;;) {
                                    bool cancelled = false;
                                    {
                                        std::lock_guard<std::mutex> lock(slot->mutex);
                                        cancelled = slot->cancelled;
                                    }
                                    if (cancelled) {
                                        return abort();
                                    }
                                    std::string piece;
                                    if (!scheduler.take_piece(slot, piece)) {
                                        break;
                                    }
                                    generated_text += piece;
                                    const std::string limited = stream_text(generated_text, stops);
                                    if (parse_stream_output && chat) {
                                        common_chat_msg message = parse_chat_output(limited, true);
                                        if (!message.empty() &&
                                            message.tool_calls.size() >= stream_message.tool_calls.size()) {
                                            set_stream_tool_call_ids(message);
                                            const auto diffs = common_chat_msg_diff::compute_diffs(stream_message, message);
                                            stream_message = std::move(message);
                                            if (!emit_chat_diffs(diffs, stream_message, true)) {
                                                return abort();
                                            }
                                        }
                                        continue;
                                    }
                                    if (limited.size() <= emitted_text.size()) {
                                        continue;
                                    }
                                    const std::string output = limited.substr(emitted_text.size());
                                    emitted_text = limited;
                                    json choice = chat
                                        ? json{ { "index", 0 }, { "delta", { { "content", output } } },
                                                { "finish_reason", nullptr } }
                                        : openai_text
                                        ? json{ { "index", 0 }, { "text", output }, { "logprobs", nullptr },
                                                { "finish_reason", nullptr } }
                                        : json{ { "content", output } };
                                    if (include_usage && !chat && !openai_text) {
                                        choice["usage"] = nullptr;
                                    }
                                    const std::string event = chat || openai_text
                                        ? "data: " + common_chunk(choice) + "\n\n"
                                        : "data: " + choice.dump() + "\n\n";
                                    if (!write(event)) {
                                        return abort();
                                    }
                                }
                                scheduler.wait_done(slot);
                                std::vector<llama_token> generated;
                                size_t stop_offset = std::string::npos;
                                std::string error;
                                {
                                    std::lock_guard<std::mutex> lock(slot->mutex);
                                    generated = slot->generated;
                                    stop_offset = slot->stop_offset;
                                    error = slot->error;
                                }
                                if (!error.empty()) {
                                    write("data: " + error_json(error, "server_error").dump() + "\n\n");
                                    sink.done();
                                    scheduler.release(slot);
                                    return false;
                                }
                                const bool stopped = stop_offset != std::string::npos;
                                const std::string final_text = stopped
                                    ? generated_text.substr(0, std::min(stop_offset, generated_text.size()))
                                    : generated_text;
                                if (!parse_stream_output && final_text.size() > emitted_text.size()) {
                                    const std::string output = final_text.substr(emitted_text.size());
                                    json choice = chat
                                        ? json{ { "index", 0 }, { "delta", { { "content", output } } },
                                                { "finish_reason", nullptr } }
                                        : openai_text
                                        ? json{ { "index", 0 }, { "text", output }, { "logprobs", nullptr },
                                                { "finish_reason", nullptr } }
                                        : json{ { "content", output } };
                                    if (include_usage && !chat && !openai_text) {
                                        choice["usage"] = nullptr;
                                    }
                                    const std::string event = chat || openai_text
                                        ? "data: " + common_chunk(choice) + "\n\n"
                                        : "data: " + choice.dump() + "\n\n";
                                    if (!write(event)) {
                                        return abort();
                                    }
                                }
                                common_chat_msg parsed = chat ? parse_chat_output(final_text, false) : common_chat_msg{};
                                if (parse_stream_output && chat) {
                                    if (!parsed.empty() &&
                                        parsed.tool_calls.size() >= stream_message.tool_calls.size()) {
                                        set_stream_tool_call_ids(parsed);
                                        const auto diffs = common_chat_msg_diff::compute_diffs(stream_message, parsed);
                                        stream_message = parsed;
                                        if (!emit_chat_diffs(diffs, stream_message, false)) {
                                            return abort();
                                        }
                                    } else if (stream_message.tool_calls.empty() &&
                                               stream_message.reasoning_content.empty() &&
                                               (stream_message.content.empty() ||
                                                final_text.compare(0, stream_message.content.size(),
                                                                   stream_message.content) == 0)) {
                                        common_chat_msg fallback = stream_message;
                                        fallback.role = "assistant";
                                        fallback.content = final_text;
                                        const auto diffs = common_chat_msg_diff::compute_diffs(stream_message, fallback);
                                        stream_message = std::move(fallback);
                                        if (!emit_chat_diffs(diffs, stream_message, false)) {
                                            return abort();
                                        }
                                    }
                                    parsed = stream_message;
                                }
                                const std::string finish = chat && !parsed.tool_calls.empty()
                                    ? "tool_calls"
                                    : stopped || generated.size() < n_predict ? "stop" : "length";
                                if (chat) {
                                    const json final_choice = {
                                        { "index", 0 }, { "delta", json::object() }, { "finish_reason", finish }
                                    };
                                    if (!write("data: " + common_chunk(final_choice) + "\n\n")) {
                                        return abort();
                                    }
                                } else if (openai_text) {
                                    const json final_choice = {
                                        { "index", 0 }, { "text", "" }, { "logprobs", nullptr },
                                        { "finish_reason", finish }
                                    };
                                    if (!write("data: " + common_chunk(final_choice) + "\n\n")) {
                                        return abort();
                                    }
                                }
                                if (include_usage) {
                                    const json usage_chunk = {
                                        { "id", id },
                                        { "object", chat ? "chat.completion.chunk" : "text_completion" },
                                        { "created", created }, { "model", model_name },
                                        { "choices", json::array() },
                                        { "usage", usage_json(prompt.size(), generated.size()) }
                                    };
                                    if (!write("data: " + usage_chunk.dump() + "\n\n")) {
                                        return abort();
                                    }
                                }
                                if (!write("data: [DONE]\n\n")) {
                                    return abort();
                                }
                                sink.done();
                                scheduler.release(slot);
                                return true;
                            } catch (const std::exception & exception) {
                                std::fprintf(stderr, "potluck-server: streaming response failed: %s\n",
                                             exception.what());
                                return abort();
                            } catch (...) {
                                std::fprintf(stderr, "potluck-server: streaming response failed\n");
                                return abort();
                            }
                        });
                    return;
                }
                std::vector<std::vector<llama_token>> generated_choices;
                std::vector<std::vector<std::vector<potluck::token_logprob>>> generated_logprobs_choices;
                std::vector<size_t> stop_offsets(slots.size(), std::string::npos);
                std::string error;
                generated_choices.resize(slots.size());
                generated_logprobs_choices.resize(slots.size());
                for (size_t i = 0; i < slots.size(); ++i) {
                    scheduler.wait_done(slots[i]);
                    std::lock_guard<std::mutex> lock(slots[i]->mutex);
                    generated_choices[i] = slots[i]->generated;
                    generated_logprobs_choices[i] = slots[i]->generated_logprobs;
                    stop_offsets[i] = slots[i]->stop_offset;
                    if (error.empty()) {
                        error = slots[i]->error;
                    }
                }
                for (const auto & pending : slots) {
                    scheduler.release(pending);
                }
                slots.clear();
                slot.reset();
                if (!error.empty()) {
                    throw std::runtime_error(error);
                }
                json choices = json::array();
                size_t completion_tokens = 0;
                for (size_t i = 0; i < generated_choices.size(); ++i) {
                    const auto & generated = generated_choices[i];
                    const std::string rendered = render_tokens(vocab, generated);
                    const bool stopped = stop_offsets[i] != std::string::npos;
                    const std::string text = stopped
                        ? rendered.substr(0, std::min(stop_offsets[i], rendered.size()))
                        : rendered;
                    const size_t visible_count =
                        visible_token_count(vocab, generated, stop_offsets[i]);
                    const std::string finish = stopped || generated.size() < n_predict ? "stop" : "length";
                    completion_tokens += generated.size();
                    if (chat) {
                        common_chat_msg parsed = parse_chat_output(text, false);
                        for (size_t tool_index = 0; tool_index < parsed.tool_calls.size(); ++tool_index) {
                            if (parsed.tool_calls[tool_index].id.empty()) {
                                parsed.tool_calls[tool_index].id =
                                    "call_" + std::to_string(i) + "_" + std::to_string(tool_index) + "_" + id;
                            }
                        }
                        const bool useful = !parsed.empty() &&
                            (!parsed.content.empty() || !parsed.reasoning_content.empty() ||
                             !parsed.tool_calls.empty());
                        const json message = useful ? parsed.to_json_oaicompat()
                                                    : json{ { "role", "assistant" }, { "content", text } };
                        json choice = {
                            { "index", i }, { "message", message },
                            { "finish_reason", useful && !parsed.tool_calls.empty() ? "tool_calls" : finish }
                        };
                        if (sampling.logprobs) {
                            choice["logprobs"] = chat_logprobs_json(
                                vocab, generated, generated_logprobs_choices[i], sampling.top_logprobs,
                                visible_count);
                        }
                        choices.push_back(std::move(choice));
                    } else if (openai_text) {
                        choices.push_back({
                            { "index", i }, { "text", text },
                            { "logprobs", sampling.logprobs
                                ? text_logprobs_json(vocab, generated, generated_logprobs_choices[i],
                                                sampling.top_logprobs, visible_count)
                                : json(nullptr) },
                            { "finish_reason", finish }
                        });
                    } else {
                        choices.push_back({
                            { "index", i }, { "text", text }, { "finish_reason", finish }
                        });
                    }
                }
                json result;
                if (chat) {
                    result = {
                        { "id", id }, { "object", "chat.completion" }, { "created", created },
                        { "model", model_name }, { "choices", std::move(choices) },
                        { "usage", usage_json(prompt.size(), completion_tokens) }
                    };
                } else if (openai_text) {
                    result = {
                        { "id", id }, { "object", "text_completion" }, { "created", created },
                        { "model", model_name }, { "choices", std::move(choices) },
                        { "usage", usage_json(prompt.size(), completion_tokens) }
                    };
                } else {
                    const json & first = choices.at(0);
                    result = { { "content", first.at("text") },
                               { "n_predict", generated_choices.front().size() },
                               { "finish_reason", first.at("finish_reason") } };
                }
                set_common_headers(response);
                response.set_content(result.dump(), "application/json");
            } catch (const std::exception & e) {
                if (slot) {
                    scheduler.release(slot);
                }
                for (const auto & pending : slots) {
                    if (pending && pending != slot) {
                        scheduler.release(pending);
                    }
                }
                const std::string message = e.what();
                const bool client_error =
                    message == "missing prompt" || message == "prompt is empty" ||
                    message == "missing messages" || message.rfind("invalid ", 0) == 0 ||
                    message.rfind("unsupported field:", 0) == 0 ||
                    message.rfind("unsupported stream_options field:", 0) == 0 ||
                    message.rfind("unsupported combination:", 0) == 0;
                response.status = client_error ? 400 : 503;
                set_common_headers(response);
                response.set_content(error_json(message, client_error ? "invalid_request_error" : "server_error").dump(),
                                     "application/json");
            }
        };

        server.Post("/completion", [handle](const httplib::Request & request, httplib::Response & response) {
            handle(request, response, false, false);
        });
        server.Post("/v1/completions", [handle](const httplib::Request & request, httplib::Response & response) {
            handle(request, response, false, true);
        });
        server.Post("/v1/chat/completions", [handle](const httplib::Request & request, httplib::Response & response) {
            handle(request, response, true, false);
        });
        server.set_error_handler([set_common_headers](const httplib::Request &, httplib::Response & response) {
            set_common_headers(response);
            if (response.status == 404) {
                response.set_content(error_json("not found").dump(), "application/json");
            }
        });
}

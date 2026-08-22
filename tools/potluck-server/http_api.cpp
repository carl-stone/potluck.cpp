// potluck-server HTTP and OpenAI-compatible API routes.

#include "internal.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#include <atomic>

namespace {

uint32_t json_u32(const json & object, const char * key, uint32_t fallback) {
    if (!object.contains(key)) {
        return fallback;
    }
    const json & value = object.at(key);
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

bool json_bool(const json & object, const char * key, bool fallback) {
    if (!object.contains(key)) {
        return fallback;
    }
    if (!object.at(key).is_boolean()) {
        throw std::runtime_error(std::string("invalid ") + key + ": expected a boolean");
    }
    return object.at(key).get<bool>();
}
std::atomic<uint64_t> request_counter{0};

std::string request_id(uint64_t & created) {
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    created = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::seconds>(now).count());
    const uint64_t sequence = request_counter.fetch_add(1, std::memory_order_relaxed);
    return "chatcmpl-potluck-" + std::to_string(created) + "-" + std::to_string(sequence);
}

json error_json(const std::string & message) {
    return json{ { "error", message } };
}

} // namespace
void setup_http_routes(httplib::Server & server, ring_session & session,
                       slot_scheduler & scheduler, const llama_vocab * vocab,
                       const std::string & model_name,
                       common_chat_templates_ptr & chat_templates,
                       uint32_t n_predict_default, float temp, float top_p, uint32_t seed) {
        const auto set_common_headers = [](httplib::Response & response) {
            response.set_header("Access-Control-Allow-Origin", "*");
            response.set_header("Cache-Control", "no-cache");
        };
        server.Options(R"(/.*)", [set_common_headers](const httplib::Request &, httplib::Response & response) {
            set_common_headers(response);
            response.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
            response.set_header("Access-Control-Allow-Headers", "Content-Type");
            response.status = 204;
        });
        server.Get("/health", [&session, &scheduler, set_common_headers](const httplib::Request &, httplib::Response & response) {
            const bool recovery_exhausted = scheduler.recovery_exhausted();
            const bool rebuilding = scheduler.rebuilding();
            const std::string recovery_error = scheduler.recovery_error();
            json slots = scheduler.health();
            std::lock_guard<std::mutex> ring_lock(session.mutex);
            const char * status = recovery_exhausted ? "failed"
                                : (rebuilding || !session.healthy) ? "rebuilding" : "ok";
            json health = { { "status", status },
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
        server.Get("/v1/models", [model_name, set_common_headers](const httplib::Request &, httplib::Response & response) {
            json result = { { "object", "list" }, { "data", json::array({ json{
                { "id", model_name }, { "object", "model" }, { "owned_by", "potluck" }
            } }) } };
            set_common_headers(response);
            response.set_content(result.dump(), "application/json");
        });

        auto handle = [vocab, model_name, &chat_templates, &scheduler, n_predict_default, temp, top_p, seed, set_common_headers](const httplib::Request & request, httplib::Response & response, bool chat) {
            std::shared_ptr<scheduled_slot> slot;
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
                    "messages", "prompt", "max_tokens", "n_predict", "stream",
                    "reasoning_effort", "temperature", "top_p", "top_k", "seed"
                };
                for (auto it = req.begin(); it != req.end(); ++it) {
                    if (allowed.find(it.key()) == allowed.end()) {
                        throw std::runtime_error("unsupported field: " + it.key());
                    }
                }
                if (req.contains("reasoning_effort") &&
                    !req["reasoning_effort"].is_string()) {
                    throw std::runtime_error(
                        "invalid reasoning_effort: expected a string");
                }

                std::string prompt_text;
                uint32_t n_predict = n_predict_default;
                if (chat) {
                    if (!req.contains("messages") || !req["messages"].is_array() ||
                        req["messages"].empty()) {
                        throw std::runtime_error("missing messages");
                    }
                    common_chat_templates_inputs inputs;
                    try {
                        inputs.messages = common_chat_msgs_parse_oaicompat(req["messages"]);
                    } catch (const std::exception & e) {
                        throw std::runtime_error(std::string("invalid messages: ") + e.what());
                    }
                    if (req.contains("reasoning_effort")) {
                        const std::string effort =
                            req["reasoning_effort"].get<std::string>();
                        if (effort == "none") {
                            inputs.enable_thinking = false;
                            inputs.chat_template_kwargs["enable_thinking"] = "false";
                        } else if (!effort.empty()) {
                            inputs.chat_template_kwargs["reasoning_effort"] =
                                json(effort).dump();
                        }
                    }
                    if (!chat_templates) {
                        throw std::runtime_error("model has no chat template");
                    }
                    prompt_text = common_chat_templates_apply(chat_templates.get(), inputs).prompt;
                    n_predict = json_u32(req, "max_tokens", n_predict_default);
                } else {
                    if (!req.contains("prompt") || !req["prompt"].is_string() ||
                        req["prompt"].get<std::string>().empty()) {
                        throw std::runtime_error("missing prompt");
                    }
                    prompt_text = req["prompt"].get<std::string>();
                    n_predict = json_u32(req, "n_predict", n_predict_default);
                }
                const std::vector<llama_token> prompt = tokenize_prompt(vocab, prompt_text);
                const bool stream = json_bool(req, "stream", false);
                potluck::slot_config sampling;
                sampling.temp = temp;
                sampling.top_p = top_p;
                sampling.seed = seed;
                if (req.contains("temperature")) {
                    if (!req["temperature"].is_number()) {
                        throw std::runtime_error("invalid temperature: expected a number");
                    }
                    sampling.temp = req["temperature"].get<float>();
                }
                if (!std::isfinite(sampling.temp) || sampling.temp < 0.0f) {
                    throw std::runtime_error("invalid temperature: expected a finite non-negative number");
                }
                if (req.contains("top_p")) {
                    if (!req["top_p"].is_number()) {
                        throw std::runtime_error("invalid top_p: expected a number");
                    }
                    sampling.top_p = req["top_p"].get<float>();
                }
                if (!std::isfinite(sampling.top_p) ||
                    sampling.top_p < 0.0f || sampling.top_p > 1.0f) {
                    throw std::runtime_error("invalid top_p: expected a number from 0 to 1");
                }
                if (req.contains("top_k")) {
                    sampling.top_k = json_u32(req, "top_k", 0);
                    if (sampling.top_k > static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
                        throw std::runtime_error("invalid top_k: out of range");
                    }
                }
                if (req.contains("seed")) {
                    sampling.seed = json_u32(req, "seed", 0);
                }
                uint64_t created = 0;
                const std::string id = request_id(created);
                slot = scheduler.acquire(prompt, n_predict, sampling, stream, chat,
                                          id, created);
                if (!slot) {
                    response.status = 503;
                    set_common_headers(response);
                    std::string message = "all conversation slots are busy";
                    if (scheduler.recovery_exhausted()) {
                        message = "cluster recovery failed: " + scheduler.recovery_error();
                    } else if (scheduler.rebuilding()) {
                        message = "cluster is rebuilding; retry";
                    }
                    response.set_content(error_json(message).dump(), "application/json");
                    return;
                }
                const auto common_chunk = [id, created, model_name](const json & choice) {
                    return json{
                        { "id", id }, { "object", "chat.completion.chunk" },
                        { "created", created }, { "model", model_name },
                        { "choices", json::array({ choice }) }
                    }.dump();
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
                        response.set_content(error_json("request cancelled").dump(),
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
                        [&, slot, chat, common_chunk](size_t, httplib::DataSink & sink) mutable {
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
                                std::string piece;
                                for (;;) {
                                    bool cancelled = false;
                                    {
                                        std::lock_guard<std::mutex> lock(slot->mutex);
                                        cancelled = slot->cancelled;
                                    }
                                    if (cancelled) {
                                        return abort();
                                    }
                                    if (!scheduler.take_piece(slot, piece)) {
                                        {
                                            std::lock_guard<std::mutex> lock(slot->mutex);
                                            cancelled = slot->cancelled;
                                        }
                                        if (cancelled) {
                                            return abort();
                                        }
                                        break;
                                    }
                                    const json delta = chat
                                        ? json{ { "index", 0 },
                                                { "delta", { { "content", piece } } },
                                                { "finish_reason", nullptr } }
                                        : json{ { "content", piece } };
                                    const std::string event = chat
                                        ? "data: " + common_chunk(delta) + "\n\n"
                                        : "data: " + delta.dump() + "\n\n";
                                    if (!write(event)) {
                                        return abort();
                                    }
                                }
                                scheduler.wait_done(slot);
                                std::string error;
                                bool cancelled = false;
                                {
                                    std::lock_guard<std::mutex> lock(slot->mutex);
                                    cancelled = slot->cancelled;
                                    error = slot->error;
                                }
                                if (cancelled) {
                                    return abort();
                                }
                                if (!error.empty()) {
                                    write("data: " + error_json(error).dump() + "\n\n");
                                    sink.done();
                                    scheduler.release(slot);
                                    return false;
                                }
                                if (chat) {
                                    const json final_choice = {
                                        { "index", 0 }, { "delta", json::object() },
                                        { "finish_reason", "stop" }
                                    };
                                    if (!write("data: " + common_chunk(final_choice) + "\n\n")) {
                                        return abort();
                                    }
                                }
                                if (!write("data: [DONE]\n\n")) {
                                    return abort();
                                }
                                sink.done();
                                scheduler.release(slot);
                                return true;
                            } catch (const std::exception &) {
                                return abort();
                            }
                        });
                    return;
                }

                scheduler.wait_done(slot);
                std::vector<llama_token> generated;
                std::string error;
                {
                    std::lock_guard<std::mutex> lock(slot->mutex);
                    generated = slot->generated;
                    error = slot->error;
                }
                scheduler.release(slot);
                slot.reset();
                if (!error.empty()) {
                    throw std::runtime_error(error);
                }
                const std::string text = render_tokens(vocab, generated);
                json result;
                if (chat) {
                    result = {
                        { "id", id }, { "object", "chat.completion" }, { "created", created },
                        { "model", model_name }, { "choices", json::array({ json{
                            { "index", 0 },
                            { "message", { { "role", "assistant" }, { "content", text } } },
                            { "finish_reason", "stop" }
                        } }) },
                        { "usage", { { "prompt_tokens", prompt.size() },
                                     { "completion_tokens", generated.size() },
                                     { "total_tokens", prompt.size() + generated.size() } } }
                    };
                } else {
                    result = { { "content", text }, { "n_predict", generated.size() },
                               { "finish_reason", "stop" } };
                }
                set_common_headers(response);
                response.set_content(result.dump(), "application/json");
            } catch (const std::exception & e) {
                if (slot) {
                    scheduler.release(slot);
                }
                const std::string message = e.what();
                const bool client_error =
                    message == "missing prompt" || message == "prompt is empty" ||
                    message == "missing messages" || message.rfind("invalid ", 0) == 0 ||
                    message.rfind("unsupported field:", 0) == 0;
                response.status = client_error ? 400 : 503;
                set_common_headers(response);
                response.set_content(error_json(message).dump(), "application/json");
            }
        };

        server.Post("/completion", [handle](const httplib::Request & request, httplib::Response & response) {
            handle(request, response, false);
        });
        server.Post("/v1/chat/completions", [handle](const httplib::Request & request, httplib::Response & response) {
            handle(request, response, true);
        });
        server.set_error_handler([set_common_headers](const httplib::Request &, httplib::Response & response) {
            set_common_headers(response);
            if (response.status == 404) {
                response.set_content(error_json("not found").dump(), "application/json");
            }
        });
}

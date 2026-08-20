// prima-server: a normal HTTP server around the distributed prima chain.
//
// One process loads the model, spawns local prima-worker stage processes on
// demand (plus optional remote workers from a file), wires the layer chain
// once, and serves /completion and /v1/chat/completions (llama-server style,
// with SSE streaming). Each request re-prefills the same resident chain from a
// clean state (clear + position-strided prefill), then generates and streams
// back token pieces.

#include "llama.h"

#include "prima-distributed-protocol.h"
#include "prima-distributed-transport.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <mach-o/dyld.h>

namespace {

[[noreturn]] void fail(const std::string & what) {
    std::fprintf(stderr, "prima-server: %s\n", what.c_str());
    std::fflush(nullptr);
    std::_Exit(1);
}

std::string exe_dir() {
    char buf[4096];
    uint32_t sz = static_cast<uint32_t>(sizeof(buf));
    _NSGetExecutablePath(buf, &sz);
    std::string p(buf);
    const size_t slash = p.rfind('/');
    return slash == std::string::npos ? "" : p.substr(0, slash);
}

std::string json_escape(const std::string & s) {
    std::string o;
    o.reserve(s.size() + 8);
    for (const char c : s) {
        switch (c) {
        case '"':  o += "\\\""; break;
        case '\\': o += "\\\\"; break;
        case '\n': o += "\\n";  break;
        case '\r': o += "\\r";  break;
        case '\t': o += "\\t";  break;
        default:
            if (static_cast<unsigned char>(c) < 0x20) {
                char esc[8];
                std::snprintf(esc, sizeof(esc), "\\u%04x", static_cast<int>(c));
                o += esc;
            } else {
                o += c;
            }
        }
    }
    return o;
}

// ---- Minimal JSON ---------------------------------------------------------
struct JsonVal {
    std::string type = "null";
    std::string str;
    double num = 0;
    bool b = false;
    std::vector<JsonVal> arr;
    std::map<std::string, JsonVal> obj;

    const JsonVal * get(const std::string & key) const {
        const auto it = obj.find(key);
        return it == obj.end() ? nullptr : &it->second;
    }
    std::string as_str(const std::string & dfl) const { return type == "str" ? str : dfl; }
    double as_num(double dfl) const { return type == "num" ? num : dfl; }
    bool as_bool(bool dfl) const { return type == "bool" ? b : dfl; }
};

static void skip_ws(const std::string & s, size_t & i) {
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) ++i;
}

static std::string parse_string(const std::string & s, size_t & i) {
    ++i;
    std::string out;
    while (i < s.size() && s[i] != '"') {
        if (s[i] == '\\' && i + 1 < s.size()) {
            ++i;
            switch (s[i]) {
            case 'n': out += '\n'; break;
            case 'r': out += '\r'; break;
            case 't': out += '\t'; break;
            case 'b': out += '\b'; break;
            case 'f': out += '\f'; break;
            case '/': out += '/';  break;
            case '\\': out += '\\'; break;
            case '"': out += '"';  break;
            case 'u': {
                const std::string hex = s.substr(i + 1, 4);
                const long cp = std::strtol(hex.c_str(), nullptr, 16);
                out += static_cast<char>((cp >> 8) & 0xff);
                out += static_cast<char>(cp & 0xff);
                i += 4;
                break;
            }
            default: out += s[i];
            }
        } else {
            out += s[i];
        }
        ++i;
    }
    if (i < s.size()) ++i;
    return out;
}

static JsonVal parse_json(const std::string & s, size_t & i) {
    skip_ws(s, i);
    if (i >= s.size()) { JsonVal v; v.type = "null"; return v; }
    const char c = s[i];
    if (c == '{') {
        ++i;
        JsonVal v; v.type = "obj";
        while (i < s.size()) {
            skip_ws(s, i);
            if (i < s.size() && s[i] == '}') { ++i; break; }
            if (s[i] != '"') break;
            const std::string key = parse_string(s, i);
            skip_ws(s, i);
            if (s[i] == ':') ++i;
            v.obj[key] = parse_json(s, i);
            skip_ws(s, i);
            if (s[i] == ',') ++i;
        }
        return v;
    }
    if (c == '[') {
        ++i;
        JsonVal v; v.type = "arr";
        while (i < s.size()) {
            skip_ws(s, i);
            if (s[i] == ']') { ++i; break; }
            v.arr.push_back(parse_json(s, i));
            skip_ws(s, i);
            if (s[i] == ',') ++i;
        }
        return v;
    }
    if (c == '"') { JsonVal v; v.type = "str"; v.str = parse_string(s, i); return v; }
    std::string tok;
    while (i < s.size() && (std::isalnum(static_cast<unsigned char>(s[i])) ||
                            s[i] == '-' || s[i] == '.' || s[i] == '+' || s[i] == 'e' || s[i] == 'E')) {
        tok += s[i++];
    }
    JsonVal v;
    if (tok == "true" || tok == "false") { v.type = "bool"; v.b = (tok == "true"); }
    else if (tok == "null") { v.type = "null"; }
    else { v.type = "num"; v.num = std::strtod(tok.c_str(), nullptr); }
    return v;
}

// ---- Minimal HTTP/1.1 -----------------------------------------------------
static void socket_write_all(int fd, const uint8_t * data, size_t n) {
    while (n > 0) {
        const ssize_t w = send(fd, data, n, 0);
        if (w <= 0) break;
        data += w;
        n -= static_cast<size_t>(w);
    }
}

static void http_send(int fd, int status, const std::string & content_type,
                      const std::string & body) {
    std::string head = "HTTP/1.1 " + std::to_string(status) + " " +
                       (status == 200 ? "OK" : (status == 400 ? "Bad Request" : "Internal Server Error")) +
                       "\r\nContent-Type: " + content_type +
                       "\r\nContent-Length: " + std::to_string(body.size()) +
                       "\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\n\r\n" + body;
    socket_write_all(fd, reinterpret_cast<const uint8_t *>(head.c_str()), head.size());
}

static void sse_start(int fd) {
    const std::string h = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream; charset=utf-8\r\n"
                          "Connection: close\r\nAccess-Control-Allow-Origin: *\r\n\r\n";
    socket_write_all(fd, reinterpret_cast<const uint8_t *>(h.c_str()), h.size());
}

static bool http_read(int fd, std::string & method, std::string & path, std::string & body) {
    std::string buf;
    std::vector<uint8_t> chunk(65536);
    for (;;) {
        const ssize_t n = recv(fd, chunk.data(), chunk.size(), 0);
        if (n <= 0) return false;
        buf.append(chunk.begin(), chunk.begin() + n);
        const size_t hdr_end = buf.find("\r\n\r\n");
        if (hdr_end != std::string::npos) {
            std::string header = buf.substr(0, hdr_end);
            size_t clen = 0;
            const size_t cf = header.find("Content-Length:");
            if (cf != std::string::npos) {
                size_t q = cf + 15;
                while (q < header.size() && header[q] == ' ') ++q;
                clen = static_cast<size_t>(std::strtoul(header.substr(q).c_str(), nullptr, 10));
            }
            const size_t body_start = hdr_end + 4;
            while (buf.size() < body_start + clen) {
                const ssize_t m = recv(fd, chunk.data(), chunk.size(), 0);
                if (m <= 0) break;
                buf.append(chunk.begin(), chunk.begin() + m);
            }
            const size_t le = header.find("\r\n");
            const std::string start = header.substr(0, le);
            method.clear(); path.clear();
            const size_t sp1 = start.find(' ');
            if (sp1 != std::string::npos) {
                const size_t sp2 = start.find(' ', sp1 + 1);
                method = start.substr(0, sp1);
                path = (sp2 == std::string::npos) ? start.substr(sp1 + 1)
                                                  : start.substr(sp1 + 1, sp2 - sp1 - 1);
            }
            body = buf.substr(body_start, clen);
            return true;
        }
        if (buf.size() > (1u << 20)) return false;
    }
}

// ---- Worker provisioning --------------------------------------------------
static uint16_t free_port() {
    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) fail("no socket for the port probe");
    struct sockaddr_in a;
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    a.sin_port = 0;
    if (bind(fd, reinterpret_cast<struct sockaddr *>(&a), sizeof(a)) < 0) {
        close(fd);
        fail("cannot bind the port probe");
    }
    struct sockaddr_in got;
    socklen_t gl = sizeof(got);
    getsockname(fd, reinterpret_cast<struct sockaddr *>(&got), &gl);
    const uint16_t port = ntohs(got.sin_port);
    close(fd);
    return port;
}

// Fork N local worker processes on free ports; returns their addresses.
static std::vector<prima::node_addr> spawn_local(const std::string & worker_path,
                                                 const std::string & model,
                                                 const std::string & host, uint32_t count) {
    std::vector<prima::node_addr> out;
    out.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        const uint16_t port = free_port();
        const std::string ps = std::to_string(port);
        const pid_t pid = fork();
        if (pid == 0) {
            // Exec the worker. Argument strings below live through execv.
            std::string m = model;
            std::string h = host;
            std::string p = ps;
            char * av[] = { const_cast<char *>(worker_path.c_str()),
                            const_cast<char *>(m.c_str()),
                            const_cast<char *>(h.c_str()),
                            const_cast<char *>(p.c_str()), nullptr };
            execv(worker_path.c_str(), av);
            std::_Exit(127);
        }
        prima::node_addr a;
        a.host = host;
        a.port = port;
        out.push_back(a);
    }
    return out;
}

static std::vector<prima::node_addr> read_remote(const std::string & path) {
    std::vector<prima::node_addr> addrs;
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        const size_t hash = line.find('#');
        std::string l = hash == std::string::npos ? line : line.substr(0, hash);
        const size_t cr = l.find('\r');
        if (cr != std::string::npos) l.erase(cr);
        if (l.empty()) continue;
        const size_t colon = l.find(':');
        if (colon == std::string::npos || colon == 0) continue;
        std::string port_s = l.substr(colon + 1);
        const size_t sp = port_s.find(' ');
        if (sp != std::string::npos) port_s = port_s.substr(0, sp);
        prima::node_addr a;
        a.host = l.substr(0, colon);
        a.port = static_cast<uint16_t>(std::stoul(port_s));
        addrs.push_back(a);
    }
    return addrs;
}

// ---- The persistent chain -------------------------------------------------
struct ServerChain {
    std::vector<prima::node_addr> workers;
    prima::tcp_channel stage0;
    prima::tcp_channel result;
    std::string error;
};

static ServerChain wire_chain(const std::vector<prima::node_addr> & addrs,
                              const std::string & head_host, uint32_t n_layer,
                              uint32_t seed, float temp, float top_p,
                              const std::vector<int32_t> & ngl) {
    ServerChain c;
    c.workers = addrs;
    const uint32_t n_workers = static_cast<uint32_t>(addrs.size());

    std::vector<uint32_t> bounds(n_workers + 1, 0);
    for (uint32_t i = 0; i < n_workers; ++i) {
        bounds[i + 1] = static_cast<uint32_t>((static_cast<uint64_t>(n_layer) * (i + 1)) / n_workers);
        if (bounds[i + 1] <= bounds[i]) fail("a worker would receive zero layers; too many workers");
    }
    if (bounds[n_workers] != n_layer) bounds[n_workers] = n_layer;

    prima::tcp_listener result_listener = prima::tcp_listener::bind_host(head_host, 0);
    if (!result_listener.valid()) fail("cannot bind the result listener");

    prima::node_config config;
    config.n_workers = n_workers;
    config.index = 0;
    config.n_ctx = 2048;
    config.seed = seed;
    config.temp = temp;
    config.top_p = top_p;
    config.bounds = bounds;
    config.workers = addrs;
    config.ngl = ngl;
    config.tail = false;
    config.head_link.host = head_host;
    config.head_link.port = result_listener.port();

    c.stage0 = prima::connect_retry(addrs[0].host, addrs[0].port, 9000, 200, c.error);
    if (!c.stage0.valid()) fail("cannot connect to worker 0");
    std::vector<uint8_t> payload;
    if (!prima::encode_config(config, payload)) fail("cannot encode config");
    prima::message msg;
    msg.type = prima::message_type::node_config;
    msg.payload = std::move(payload);
    if (!c.stage0.send(msg, c.error)) fail("cannot send config to worker 0");
    c.result = result_listener.accept(c.error);
    if (!c.result.valid()) fail("tail never connected back");
    prima::message ready;
    if (!c.stage0.receive(ready, c.error) || ready.type != prima::message_type::ready) {
        fail("chain never became ready");
    }
    std::printf("prima-server: chain ready (%u workers)\n", n_workers);
    std::fflush(stdout);
    return c;
}

static std::vector<int32_t> drive_batch(ServerChain & chain,
                                        const std::vector<int32_t> & pos,
                                        const std::vector<int32_t> & seq,
                                        const std::vector<int32_t> & tok,
                                        int32_t clear, int32_t trim_to) {
    prima::message in;
    in.type = prima::message_type::batch_decode;
    in.sequence = pos.empty() ? 0 : static_cast<uint64_t>(pos.back());
    std::vector<uint8_t> payload;
    if (!prima::encode_batch_payload(pos, seq, tok, nullptr, 0, clear, trim_to, payload)) {
        fail("cannot encode batch request");
    }
    in.payload = std::move(payload);
    if (!chain.stage0.send(in, chain.error)) fail("cannot send batch to stage 0");
    prima::message out;
    if (!chain.result.receive(out, chain.error)) fail("batch result channel closed");
    if (out.type != prima::message_type::batch_result) fail("unexpected batch result");
    std::vector<int32_t> rpos, rseq, rtok;
    std::vector<float> rhidden;
    int32_t clear_ignored = 0, trim_ignored = -1;
    if (!prima::decode_batch_payload(out.payload.data(), out.payload.size(), 0,
                                     clear_ignored, trim_ignored, rpos, rseq, rtok, rhidden,
                                     chain.error)) {
        fail("cannot decode batch result");
    }
    if (rpos.size() != pos.size() || rtok.size() != pos.size()) fail("batch result entry mismatch");
    return rtok;
}

static std::string token_to_piece(const llama_vocab * vocab, llama_token t) {
    std::vector<uint8_t> buf(64);
    for (;;) {
        const int32_t n = llama_token_to_piece(vocab, t, reinterpret_cast<char *>(buf.data()),
                                               static_cast<int32_t>(buf.size()), 0, 0);
        if (n >= 0) return std::string(buf.begin(), buf.begin() + n);
        buf.resize(static_cast<size_t>(-n) + 8);
    }
}

static std::vector<llama_token> tokenize_prompt(const llama_vocab * vocab, const std::string & text) {
    std::vector<llama_token> toks(static_cast<size_t>(llama_vocab_n_tokens(vocab)));
    const int32_t n = llama_tokenize(vocab, text.c_str(), static_cast<int32_t>(text.size()),
                                     toks.data(), static_cast<int32_t>(toks.size()),
                                     /*add_special=*/false, /*parse_special=*/false);
    if (n <= 0) fail("tokenization failed");
    toks.resize(static_cast<size_t>(n));
    toks.insert(toks.begin(), llama_vocab_bos(vocab));
    return toks;
}

// Generate through the existing chain. emit() receives each decoded token's
// piece as it is produced (for SSE). Returns the raw tokens.
static std::vector<llama_token> serve(ServerChain & chain, const llama_vocab * vocab,
                                      const std::vector<llama_token> & prompt,
                                      uint32_t n_predict,
                                      std::function<void(const std::string &)> emit) {
    const llama_token eos = llama_vocab_eos(vocab);
    // Clear + position-strided prefill (the same per-position construction the
    // batch tests use, so serves are exactly the sequential single-run decode).
    for (size_t i = 0; i < prompt.size(); ++i) {
        (void)drive_batch(chain, {static_cast<int32_t>(i)}, {0},
                          {static_cast<int32_t>(prompt[i])}, i == 0 ? 1 : 0, -1);
    }
    std::vector<llama_token> generated;
    generated.reserve(n_predict);
    llama_token prev = prompt.back();
    uint32_t cur = static_cast<uint32_t>(prompt.size());
    for (uint32_t step = 0; step < n_predict; ++step) {
        const std::vector<int32_t> r = drive_batch(chain, {static_cast<int32_t>(cur)},
                                                   {0}, {static_cast<int32_t>(prev)}, 0, -1);
        if (r.empty()) break;
        const llama_token next = static_cast<llama_token>(r[0]);
        generated.push_back(next);
        if (emit) emit(token_to_piece(vocab, next));
        if (next == eos) break;
        prev = next;
        ++cur;
    }
    return generated;
}

static std::string render_text(const llama_vocab * vocab, const std::vector<llama_token> & tokens) {
    std::string out;
    for (llama_token t : tokens) out += token_to_piece(vocab, t);
    return out;
}

static std::string build_chat(const JsonVal & req, const std::string & tmpl) {
    const JsonVal * m = req.get("messages");
    if (m == nullptr || m->type != "arr" || m->arr.empty()) return "";
    std::vector<std::string> role_str, content_str;
    std::vector<llama_chat_message> msgs;
    for (const JsonVal & msg : m->arr) {
        if (msg.type != "obj") continue;
        const JsonVal * role = msg.get("role");
        const JsonVal * content = msg.get("content");
        if (content == nullptr || content->type != "str") continue;
        role_str.push_back(role != nullptr ? role->as_str("user") : "user");
        content_str.push_back(content->str);
        llama_chat_message cm;
        cm.role = role_str.back().c_str();
        cm.content = content_str.back().c_str();
        msgs.push_back(cm);
    }
    if (msgs.empty()) return "";
    std::vector<char> buf(16384, 0);
    const int32_t len = llama_chat_apply_template(tmpl.c_str(), msgs.data(), msgs.size(),
                                                  /*add_ass=*/true, buf.data(),
                                                  static_cast<int32_t>(buf.size()));
    if (len < 0) return "";
    return std::string(buf.data(), static_cast<size_t>(len));
}

} // namespace

int main(int argc, char ** argv) {
    std::string model_path;
    std::string host = "127.0.0.1";
    uint16_t http_port = 8080;
    uint32_t worker_local = 2;
    std::string remote_file;
    uint32_t gpu_layers = 0, gpu_mem_mb = 0;
    float temp = 0.0f, top_p = 0.0f;
    uint32_t seed = 0;
    uint32_t n_predict_default = 24;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto take = [&](const std::string & name) -> std::string {
            if (i + 1 >= argc) fail("missing value for " + name);
            return argv[++i];
        };
        if (a == "-m" || a == "--model") model_path = take("--model");
        else if (a == "--host") host = take("--host");
        else if (a == "--port") http_port = static_cast<uint16_t>(std::stoul(take("--port")));
        else if (a == "--workers") worker_local = static_cast<uint32_t>(std::stoul(take("--workers")));
        else if (a == "--workers-file") remote_file = take("--workers-file");
        else if (a == "--gpu-layers" || a == "--ngl") gpu_layers = static_cast<uint32_t>(std::stoul(take("--gpu-layers")));
        else if (a == "--gpu-mem") gpu_mem_mb = static_cast<uint32_t>(std::stoul(take("--gpu-mem")));
        else if (a == "--temp") temp = std::stof(take("--temp"));
        else if (a == "--top-p") top_p = std::stof(take("--top-p"));
        else if (a == "--seed") seed = static_cast<uint32_t>(std::stoul(take("--seed")));
        else if (a == "--n-predict") n_predict_default = static_cast<uint32_t>(std::stoul(take("--n-predict")));
        else fail("usage: prima-server -m model.gguf [--host H] [--port P] [--workers N] [--workers-file F] [--gpu-layers K] [--gpu-mem MB] [--temp F] [--top-p F] [--seed N] [--n-predict N]");
    }
    if (model_path.empty()) fail("need -m model.gguf");

    llama_backend_init();
    llama_model_params mparams = llama_model_default_params();
    llama_model * meta = llama_model_load_from_file(model_path.c_str(), mparams);
    if (meta == nullptr) fail("cannot load the model");
    const llama_vocab * vocab = llama_model_get_vocab(meta);
    const uint32_t n_layer = static_cast<uint32_t>(llama_model_n_layer(meta));
    const char * tmpl_ptr = llama_model_chat_template(meta, nullptr);
    const std::string chat_tmpl = tmpl_ptr != nullptr ? std::string(tmpl_ptr) : "";

    const std::string bind_host = (host.empty() || host == "0.0.0.0") ? "127.0.0.1" : host;

    std::vector<prima::node_addr> addrs;
    if (worker_local > 0) {
        const std::string wdir = exe_dir();
        const std::string worker_path = wdir.empty() ? "prima-worker" : wdir + "/prima-worker";
        std::vector<prima::node_addr> spawned = spawn_local(worker_path, model_path, bind_host, worker_local);
        addrs.insert(addrs.end(), spawned.begin(), spawned.end());
    }
    if (!remote_file.empty()) {
        std::vector<prima::node_addr> r = read_remote(remote_file);
        addrs.insert(addrs.end(), r.begin(), r.end());
    }
    if (addrs.empty()) fail("need at least one worker (--workers N or --workers-file F)");

    std::vector<int32_t> ngl(addrs.size(), 0);
    if (gpu_layers > 0 || gpu_mem_mb > 0) {
        uint32_t K = std::min<uint32_t>(gpu_layers, n_layer);
        if (gpu_mem_mb > 0) {
            const uint64_t bpl = std::max<uint64_t>(1, llama_model_size(meta) / n_layer);
            K = static_cast<uint32_t>((static_cast<uint64_t>(gpu_mem_mb) * 1024 * 1024) / bpl);
            K = std::min(K, n_layer);
        }
        for (uint32_t i = 0; i < addrs.size(); ++i) {
            const int64_t bs = static_cast<int64_t>((static_cast<uint64_t>(n_layer) * i) / addrs.size());
            const int64_t be = static_cast<int64_t>((static_cast<uint64_t>(n_layer) * (i + 1)) / addrs.size());
            const int64_t off = static_cast<int64_t>(K) - bs;
            const int64_t win = be - bs;
            ngl[i] = static_cast<int32_t>(off < 0 ? 0 : (off > win ? win : off));
        }
    }

    ServerChain chain = wire_chain(addrs, bind_host, n_layer, seed, temp, top_p, ngl);

    const int listener = socket(AF_INET, SOCK_STREAM, 0);
    if (listener < 0) fail("cannot create the http socket");
    int on = 1;
    setsockopt(listener, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on));
    struct sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_port = htons(http_port);
    if (host.empty() || host == "0.0.0.0") addr.sin_addr.s_addr = htonl(INADDR_ANY);
    else inet_pton(AF_INET, bind_host.c_str(), &addr.sin_addr);
    if (bind(listener, reinterpret_cast<struct sockaddr *>(&addr), sizeof(addr)) < 0) {
        fail("cannot bind http port " + std::to_string(http_port));
    }
    listen(listener, 16);
    std::printf("prima-server: listening on http://%s:%u (%u local workers, model %s)\n",
                host.c_str(), http_port, worker_local, model_path.c_str());
    std::fflush(stdout);

    for (;;) {
        const int client = accept(listener, nullptr, nullptr);
        if (client < 0) continue;
        std::string method, path, body;
        if (!http_read(client, method, path, body)) { close(client); continue; }

        size_t j = 0;
        JsonVal req = body.empty() ? JsonVal() : parse_json(body, j);
        const bool stream = req.get("stream") != nullptr && req.get("stream")->as_bool(false);

        std::vector<llama_token> prompt_tokens;
        uint32_t n_predict = n_predict_default;
        if (path == "/v1/chat/completions") {
            std::string formatted = chat_tmpl.empty() ? "" : build_chat(req, chat_tmpl);
            if (formatted.empty()) {
                // Fall back to the raw messages and a plain wrap if the
                // template is unavailable or fails to render.
                const JsonVal * msgs = req.get("messages");
                std::string text = "Hello";
                if (msgs != nullptr && msgs->type == "arr" && !msgs->arr.empty()) {
                    const JsonVal * c = msgs->arr.back().get("content");
                    text = c != nullptr ? c->as_str("Hello") : "Hello";
                }
                formatted = "User: " + text + "\n";
            }
            prompt_tokens = tokenize_prompt(vocab, formatted);
            const JsonVal * mt = req.get("max_tokens");
            n_predict = mt != nullptr ? static_cast<uint32_t>(std::max(0.0, mt->as_num(1))) : n_predict_default;
        } else {
            const std::string prompt_text = req.as_str("The capital of France is");
            prompt_tokens = tokenize_prompt(vocab, prompt_text);
            const JsonVal * np = req.get("n_predict");
            n_predict = np != nullptr ? static_cast<uint32_t>(std::max(0.0, np->as_num(1))) : n_predict_default;
        }
        if (n_predict == 0) n_predict = n_predict_default;

        if (stream) {
            sse_start(client);
            std::vector<llama_token> tokens = serve(chain, vocab, prompt_tokens, n_predict,
                [&](const std::string & piece) {
                    std::string ev;
                    if (path == "/v1/chat/completions") {
                        ev = "data: {\"choices\":[{\"delta\":{\"content\":\"" + json_escape(piece) + "\"}}]}\n\n";
                    } else {
                        ev = "data: {\"content\":\"" + json_escape(piece) + "\"}\n\n";
                    }
                    socket_write_all(client, reinterpret_cast<const uint8_t *>(ev.c_str()), ev.size());
                });
            (void)tokens;
            const std::string done = "data: [DONE]\n\n";
            socket_write_all(client, reinterpret_cast<const uint8_t *>(done.c_str()), done.size());
            close(client);
            continue;
        }

        const std::vector<llama_token> tokens = serve(chain, vocab, prompt_tokens, n_predict, nullptr);
        const std::string text = render_text(vocab, tokens);
        std::string resp;
        if (path == "/v1/chat/completions") {
            resp = "{\"id\":\"chatcmpl-1\",\"object\":\"chat.completion\",\"model\":\"prima\","
                   "\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"" +
                   json_escape(text) +
                   "\"},\"finish_reason\":\"stop\"}],"
                   "\"usage\":{\"prompt_tokens\":" + std::to_string(prompt_tokens.size()) +
                   ",\"completion_tokens\":" + std::to_string(tokens.size()) +
                   ",\"total_tokens\":" + std::to_string(prompt_tokens.size() + tokens.size()) + "}}";
        } else {
            resp = "{\"content\":\"" + json_escape(text) + "\",\"tokens\":[0],\"finish_reason\":\"stop\",\"n_predict\":" +
                   std::to_string(tokens.size()) + "}";
        }
        http_send(client, 200, "application/json", resp);
        close(client);
    }
    return 0;
}
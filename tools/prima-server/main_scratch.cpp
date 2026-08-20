// prima-server: a normal server around the distributed prima chain.
//
// One process loads the model's topology, spawns local prima-worker stage
// processes on free ports (plus optional remote workers from a file), wires
// the layer chain once, and serves HTTP /completion and /v1/chat/completions
// (llama-server style, with optional SSE streaming). Each request re-prefills
// the same resident chain from a clean state (a clear + position-strided
// prefill, the same construction the §12 batch path uses), generates greedily
// (or sampled when --temp / per-request temp > 0), and detokenizes.
//
// gcc/C++17 for the umbrella; the only external link is llama (+pthread on
// POSIX is not used here; the server is single-threaded per connection so no
// futexes are needed). On shutdown it tears the spawned workers down.

#include "llama.h"

#include "prima-distributed-protocol.h"
#include "prima-distributed-transport.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <cmath>
#include <sys/socket.h>
#include <unistd.h>

#if defined(__APPLE__)
#include <mach-o/dyld.h>
#else
#include <sys/stat.h>
#endif

namespace {

[[noreturn]] void fail(const std::string & what) {
    std::fprintf(stderr, "prima-server: %s\n", what.c_str());
    std::fflush(nullptr);
    std::_Exit(1);
}

std::string exe_dir() {
    char buf[4096];
#if defined(__APPLE__)
    uint32_t sz = sizeof(buf);
    if (_NSGetExecutablePath(buf, &sz) == 0) {}
#else
    if (::readlink("/proc/self/exe", buf, sizeof(buf)) > 0) {}
#endif
    std::string p(buf);
    std::string dir = buf;
    const size_t slash = p.find_last_of('/');
    dir = slash == std::string::npos ? "" : p.substr(0, slash);
    return dir;
}

std::string json_escape(const std::string & s) {
    std::string o;
    o.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
        case '"':  o += "\\\""; break;
        case '\\': o += "\\\\"; break;
        case '\n': o += "\\n"; break;
        case '\r': o += "\\r"; break;
        case '\t': o += "\\t"; break;
        default:
            if (static_cast<unsigned char>(c) < 0x20) {
                char esc[8];
                std::snprintf(esc, sizeof(esc), "\\u%04x", c);
                o += esc;
            } else {
                o += c;
            }
        }
    }
    return o;
}

// ---- Minimal JSON value + subset parser (objects of string/number/bool,
//      arrays of strings) sufficient for /completion and /v1/chat/completions.
struct JsonVal {
    std::string type;                 // "str" | "num" | "bool" | "null" | "arr" | "obj"
    std::string str;
    double num = 0;
    bool b = false;
    std::vector<JsonVal> arr;
    std::map<std::string, JsonVal> obj;

    const JsonVal * get(const std::string & key) const {
        const auto it = obj.find(key);
        return it == obj.end() ? nullptr : &it->second;
    }
    std::string as_str(const std::string & dfl) const {
        return type == "str" ? str : dfl;
    }
    double as_num(double dfl) const {
        return type == "num" ? num : dfl;
    }
};

static void skip_ws(const std::string & s, size_t & i) {
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) ++i;
}

static std::string parse_string(const std::string & s, size_t & i) {
    ++i; // opening quote
    std::string out;
    while (i < s.size() && s[i] != '"') {
        if (s[i] == '\\' && i + 1 < s.size()) {
            ++i;
            switch (s[i]) {
            case 'n': out += '\n'; break;
            case 'r': out += '\r'; break;
            case 't': out += '\t'; break;
            case '\\': out += '\\'; break;
            case '"': out += '"'; break;
            case '/': out += '/'; break;
            case 'u': {
                std::string hex = s.substr(i + 1, 4);
                out += static_cast<char>(std::strtol(hex.c_str(), nullptr, 16));
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
    if (i < s.size()) ++i; // closing quote
    return out;
}

static JsonVal parse_json(const std::string & s, size_t & i) {
    skip_ws(s, i);
    if (i >= s.size()) {
        JsonVal v; v.type = "null"; return v;
    }
    char c = s[i];
    if (c == '{') {
        ++i;
        JsonVal v; v.type = "obj";
        while (i < s.size()) {
            skip_ws(s, i);
            if (i < s.size() && s[i] == '}') { ++i; break; }
            if (i >= s.size() || s[i] != '"') break;
            std::string key = parse_string(s, i);
            skip_ws(s, i);
            if (i < s.size() && s[i] == ':') ++i;
            v.obj[key] = parse_json(s, i);
            skip_ws(s, i);
            if (i < s.size() && s[i] == ',') ++i;
        }
        return v;
    }
    if (c == '[') {
        ++i;
        JsonVal v; v.type = "arr";
        while (i < s.size()) {
            skip_ws(s, i);
            if (i < s.size() && s[i] == ']') { ++i; break; }
            v.arr.push_back(parse_json(s, i));
            skip_ws(s, i);
            if (i < s.size() && s[i] == ',') ++i;
        }
        return v;
    }
    if (c == '"') { JsonVal v; v.type = "str"; v.str = parse_string(s, i); return v; }
    // number / bool / null
    std::string tok;
    while (i < s.size() && (std::isalnum(c) || c == '-' || c == '.' || c == '+' || c == 'e' || c == 'E')) {
        tok += c;
        ++i;
        c = s[i];
    }
    JsonVal v;
    if (tok == "true" || tok == "false") { v.type = "bool"; v.b = (tok == "true"); }
    else if (tok == "null") { v.type = "null"; }
    else { v.type = "num"; v.num = std::strtod(tok.c_str(), nullptr); }
    return v;
}

// ---- Tiny HTTP/1.1 over POSIX sockets (no dependency). ------------------
struct HttpRequest {
    std::string method;
    std::string path;
    std::string body;
};

struct HttpResponse {
    int status = 200;
    std::string content_type = "application/json";
    std::vector<uint8_t> body;
    bool sse = false;
};

static void socket_write_all(int fd, const uint8_t * data, size_t n) {
    while (n > 0) {
        const ssize_t w = ::send(fd, data, n, 0);
        if (w <= 0) break;
        data += w;
        n -= static_cast<size_t>(w);
    }
}

static void http_send(int fd, const HttpResponse & r) {
    std::string head = "HTTP/1.1 " + std::to_string(r.status) + " " +
                       (r.status == 200 ? "OK" : (r.status == 400 ? "Bad Request" : "Internal Server Error")) + "\r\n"
                       "Content-Type: " + r.content_type + "\r\n"
                       "Content-Length: " + std::to_string(r.body.size()) + "\r\n"
                       "Connection: close\r\n"
                       "Access-Control-Allow-Origin: *\r\n"
                       "\r\n";
    socket_write_all(fd, reinterpret_cast<const uint8_t *>(head.c_str()), head.size());
    socket_write_all(fd, r.body.data(), r.body.size());
}

// Read a single HTTP request (headers + body). Returns false if the peer
// closed before any request or the framing is unusable.
static bool http_read(int fd, HttpRequest & req) {
    std::string buf;
    std::vector<uint8_t> chunk(65536);
    for (;;) {
        const ssize_t n = ::recv(fd, chunk.data(), chunk.size(), 0);
        if (n <= 0) {
            return req.method.empty() ? false : buf.size() >= 4;
        }
        buf.append(chunk.begin(), chunk.begin() + n);
        const size_t hdr_end = buf.find("\r\n\r\n");
        if (hdr_end != std::string::npos) {
            std::string header = buf.substr(0, hdr_end);
            size_t clen = 0;
            const size_t idx = header.find("Content-Length:");
            if (idx != std::string::npos) {
                size_t p = idx + 15;
                while (p < header.size() && header[p] == ' ') ++p;
                clen = static_cast<size_t>(std::strtoul(header.substr(p).c_str(), nullptr, 10));
            }
            const size_t body_start = hdr_end + 4;
            while (buf.size() < body_start + clen) {
                const ssize_t m = ::recv(fd, chunk.data(), chunk.size(), 0);
                if (m <= 0) break;
                buf.append(chunk.begin(), chunk.begin() + m);
            }
            const size_t line_end = header.find("\r\n");
            const std::string start = header.substr(0, line_end);
            std::string method, path;
            size_t sp1 = start.find(' ');
            if (sp1 != std::string::npos) {
                size_t sp2 = start.find(' ', sp1 + 1);
                method = start.substr(0, sp1);
                path = (sp2 == std::string::npos) ? start.substr(sp1 + 1) : start.substr(sp1 + 1, sp2 - sp1 - 1);
            }
            req.method = method;
            req.path = path;
            req.body = buf.substr(body_start, clen);
            return true;
        }
        if (buf.size() > 1 << 20) return false; // cap a malformed header
    }
}

// ---- Worker provisioning --------------------------------------------------
struct SpawnedWorker {
    std::string host;
    uint16_t port;
    pid_t pid = 0;
};

// Bind a loopback listener to get a free port, close it, return it. Slight
// race between close and the worker binding, but the worker binds within the
// same millisecond window; a collision surfaces as a worker bind failure which
// aborts startup cleanly rather than silently deadlocking.
static uint16_t free_port() {
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) fail("no socket for port probe");
    int on = 1;
    ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on));
    struct sockaddr_in a;
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = ::htonl(INADDR_LOOPBACK);
    a.sin_port = 0;
    if (::bind(fd, reinterpret_cast<struct sockaddr *>(&a), sizeof(a)) < 0) {
        ::close(fd);
        fail("cannot bind probe socket");
    }
    struct sockaddr_in got;
    socklen_t gl = sizeof(got);
    ::getsockname(fd, reinterpret_cast<struct sockaddr *>(&got), &gl);
    const uint16_t port = ntohs(got.sin_port);
    ::close(fd);
    return port;
}

// Compute the directory containing this executable and return the path to a
// sibling binary (e.g. the worker the server spawns).
static std::string sibling(const std::string & name) {
    const std::string dir = exe_dir();
    std::string out = dir.empty() ? name : dir + "/" + name;
    return out;
}

static bool file_is_exec(const std::string & p) {
    if (::access(p.c_str(), R_OK | X_OK) == 0) return true;
    return false;
}

static std::vector<pid_t> spawn_workers(const std::string & worker_path, const std::string & model,
                                        const std::string & host, uint32_t count,
                                        std::vector<SpawnedWorker> & out) {
    std::vector<pid_t> pids;
    pids.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        const uint16_t port = free_port();
        const std::string port_s = std::to_string(port);
        pid_t pid = ::fork();
        if (pid == 0) {
            const char * argv[] = { worker_path.c_str(), model.c_str(), host.c_str(), port_s.c_str(), nullptr };
            ::execv(worker_path.c_str(), argv);
            std::_Exit(127);
        }
        SpawnedWorker w;
        w.host = host;
        w.port = port;
        w.pid = pid;
        out.push_back(w);
        pids.push_back(pid);
    }
    return pids;
}

// Read an optional remote workers file: "host:port" (and optionally
// "host:port weight ram_mb vram_mb") one per line. Only non-comment, nonempty
// lines count.
static std::vector<prima::node_addr> read_remote(const std::string & path) {
    std::vector<prima::node_addr> addrs;
    std::ifstream in(path);
    std::string line;
    std::vector<std::unique_ptr<char>> unused;
    (void)unused;
    while (std::getline(in, line)) {
        const size_t hash = line.find('#');
        std::string l = hash == std::string::npos ? line : line.substr(0, hash);
        if (l.find('\r') != std::string::npos) l.erase(l.find('\r'));
        if (l.empty()) continue;
        const size_t colon = l.find(':');
        if (colon == std::string::npos || colon == 0) continue;
        prima::node_addr a;
        a.host = l.substr(0, colon);
        const std::string port_s = l.substr(colon + 1);
        // Drop any trailing tokens (weight / ram / vram) that follow whitespace.
        std::string port_only = port_s;
        const size_t sp = port_only.find(' ');
        port_only = sp == std::string::npos ? port_only : port_only.substr(0, sp);
        a.port = static_cast<uint16_t>(std::stoul(port_only));
        addrs.push_back(a);
    }
    return addrs;
}

// ---- The chain -----------------------------------------------------------
struct ServerChain {
    std::vector<prima::node_addr> workers;
    prima::node_config config;
    prima::tcp_channel stage0;
    prima::tcp_channel result;
    std::string error;

    prima::tcp_listener result_listener;
};

static ServerChain wire_chain(const std::vector<prima::node_addr> & worker_addrs,
                              std::string head_host, uint32_t seed, float temp, float top_p) {
    ServerChain c;
    c.workers = worker_addrs;
    const uint32_t n_workers = static_cast<uint32_t>(worker_addrs.size());

    std::vector<uint32_t> bounds(n_workers + 1, 0);
    // Equal-weight tessellation by default (server has no profiling of the
    // remote set); a later --profile hook can weight these per worker.
    {
        uint32_t n_layer = c.n_layer_cache; // placeholder reassigned below
    }

    // Bind the head back-link for the tail.
    c.result_listener = prima::tcp_listener::bind_host(head_host, 0);
    if (!c.result_listener.valid()) fail("cannot bind result listener");

    c.config.n_workers = n_workers;
    c.config.index = 0;
    c.config.n_ctx = 2048;
    c.config.seed = seed;
    c.config.warm = warm;
    c.config.top_p = top_p; // note below
    c.config.bounds = bounds;
    c.config.workers = worker_addrs;
    c.config.ngl.resize(n_workers, 0);
    c.config.tail = false;
    c.config.head_link.host = head_host;
    c.config.head_link.port = c.result_listener.port();

    std::string error;
    c.stage0 = prima::connect_retry(worker_addrs[0].host, worker_addrs[0].port, 600, 100, error);
    if (!c.stage0.valid()) {
        fail("cannot connect to worker 0");
    }
    std::vector<uint8_t> config_payload;
    if (!prima::encode_config(c.config, config_payload)) {
        fail("cannot encode global config");
    }
    prima::message cfg;
    cfg.type = prima::message_type::node_config;
    cfg.payload = std::move(config_payload);
    if (!c.stage0.send(cfg, error)) fail("cannot send config to worker 0");
    c.result = c.result_listener.accept(error);
    if (!c.result.valid()) fail("tail never connected back");
    prima::message ready;
    if (!c.stage0.receive(ready, error) || ready.type != prima::message_type::ready) {
        fail("chain never became ready");
    }
    std::printf("prima-server: chain ready (%u workers)\n", n_workers);
    std::fflush(stdout);
    return c;
}

} // namespace (no)
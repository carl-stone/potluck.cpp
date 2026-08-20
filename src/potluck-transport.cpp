#include "potluck-transport.h"

#include <arpa/inet.h>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <netinet/in.h>
#include <netdb.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <vector>

namespace potluck {
namespace {

constexpr size_t frame_prefix_bytes = sizeof(uint64_t);
constexpr size_t max_frame_bytes = frame_prefix_bytes + max_payload_bytes + 1024;

void close_socket(int & fd) {
    if (fd >= 0) {
        ::close(fd);
        fd = -1;
    }
}

std::string socket_error(const char * operation) {
    return std::string(operation) + ": " + std::strerror(errno);
}

uint64_t decode_u64_le(const uint8_t * data) {
    uint64_t value = 0;
    for (int i = 0; i < 8; ++i) {
        value |= static_cast<uint64_t>(data[i]) << (8 * i);
    }
    return value;
}

} // namespace

tcp_channel::tcp_channel(int fd) : fd_(fd) {}

namespace {

int env_timeout_s(const char * name, int dflt) {
    const char * v = std::getenv(name);
    if (v == nullptr || *v == '\0') {
        return dflt;
    }
    char * end = nullptr;
    const long n = std::strtol(v, &end, 10);
    return (end == v || n <= 0) ? dflt : static_cast<int>(n);
}

} // namespace

int handshake_timeout_s() {
    return env_timeout_s("POTLUCK_TIMEOUT_HANDSHAKE_S", 60);
}

int decode_timeout_s() {
    return env_timeout_s("POTLUCK_TIMEOUT_DECODE_S", 300);
}

void tcp_channel::set_timeouts(int rcv_seconds, int snd_seconds) {
    if (fd_ < 0) {
        return;
    }
    struct timeval tv;
    tv.tv_sec = rcv_seconds;
    tv.tv_usec = 0;
    ::setsockopt(fd_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    tv.tv_sec = snd_seconds;
    ::setsockopt(fd_, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
}

tcp_channel::~tcp_channel() {
    close_socket(fd_);
}

tcp_channel::tcp_channel(tcp_channel && other) noexcept : fd_(other.fd_) {
    other.fd_ = -1;
}

tcp_channel & tcp_channel::operator=(tcp_channel && other) noexcept {
    if (this != &other) {
        close_socket(fd_);
        fd_ = other.fd_;
        other.fd_ = -1;
    }
    return *this;
}

bool tcp_channel::valid() const {
    return fd_ >= 0;
}

bool tcp_channel::write_all(const uint8_t * data, size_t size, std::string & error) {
    size_t written = 0;
    while (written < size) {
        const ssize_t n = ::send(fd_, data + written, size - written, 0);
        if (n < 0 && errno == EINTR) {
            continue;
        }
        if (n <= 0) {
            error = socket_error("send");
            return false;
        }
        written += static_cast<size_t>(n);
    }
    return true;
}

bool tcp_channel::read_all(uint8_t * data, size_t size, std::string & error) {
    size_t received = 0;
    while (received < size) {
        const ssize_t n = ::recv(fd_, data + received, size - received, 0);
        if (n < 0 && errno == EINTR) {
            continue;
        }
        if (n <= 0) {
            error = n == 0 ? "peer closed connection" : socket_error("recv");
            return false;
        }
        received += static_cast<size_t>(n);
    }
    return true;
}

bool tcp_channel::send(const message & message, std::string & error) {
    const std::vector<uint8_t> frame = encode_frame(message);
    if (frame.empty()) {
        error = "message cannot be encoded";
        return false;
    }
    return write_all(frame.data(), frame.size(), error);
}

bool tcp_channel::receive(message & message, std::string & error) {
    uint8_t prefix[frame_prefix_bytes] = {};
    if (!read_all(prefix, sizeof(prefix), error)) {
        return false;
    }

    const uint64_t body_size = decode_u64_le(prefix);
    if (body_size > max_frame_bytes - frame_prefix_bytes) {
        error = "incoming frame exceeds limit: body_size=" + std::to_string(body_size);
        return false;
    }

    std::vector<uint8_t> frame(frame_prefix_bytes + static_cast<size_t>(body_size));
    std::memcpy(frame.data(), prefix, sizeof(prefix));
    if (!read_all(frame.data() + frame_prefix_bytes, static_cast<size_t>(body_size), error)) {
        return false;
    }
    return decode_frame(frame.data(), frame.size(), message, error);
}

tcp_channel tcp_channel::connect_host(const std::string & host, uint16_t port, std::string & error) {
    addrinfo hints = {};
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo * results = nullptr;
    const std::string port_text = std::to_string(port);
    const int lookup = ::getaddrinfo(host.c_str(), port_text.c_str(), &hints, &results);
    if (lookup != 0) {
        error = "connect: " + std::string(::gai_strerror(lookup));
        return {};
    }

    std::string last_error = "connect: no addresses resolved";
    for (addrinfo * result = results; result != nullptr; result = result->ai_next) {
        const int fd = ::socket(result->ai_family, result->ai_socktype, result->ai_protocol);
        if (fd < 0) {
            last_error = socket_error("socket");
            continue;
        }
        if (::connect(fd, result->ai_addr, result->ai_addrlen) == 0) {
            ::freeaddrinfo(results);
            return tcp_channel(fd);
        }
        last_error = socket_error("connect");
        int owned_fd = fd;
        close_socket(owned_fd);
    }
    ::freeaddrinfo(results);
    error = last_error;
    return {};
}

tcp_channel tcp_channel::connect_loopback(uint16_t port, std::string & error) {
    return connect_host("127.0.0.1", port, error);
}

tcp_listener::tcp_listener(int fd, uint16_t port) : fd_(fd), port_(port) {}

tcp_listener::~tcp_listener() {
    close_socket(fd_);
}

tcp_listener::tcp_listener(tcp_listener && other) noexcept : fd_(other.fd_), port_(other.port_) {
    other.fd_ = -1;
    other.port_ = 0;
}

tcp_listener & tcp_listener::operator=(tcp_listener && other) noexcept {
    if (this != &other) {
        close_socket(fd_);
        fd_ = other.fd_;
        port_ = other.port_;
        other.fd_ = -1;
        other.port_ = 0;
    }
    return *this;
}

tcp_listener tcp_listener::bind_host(const std::string & host, uint16_t port) {
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        return {};
    }

    int reuse = 1;
    if (::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) < 0) {
        int owned_fd = fd;
        close_socket(owned_fd);
        return {};
    }

    sockaddr_in address = {};
    address.sin_family = AF_INET;
    address.sin_port = htons(port);
    if (::inet_pton(AF_INET, host.c_str(), &address.sin_addr) != 1) {
        int owned_fd = fd;
        close_socket(owned_fd);
        return {};
    }

    if (::bind(fd, reinterpret_cast<const sockaddr *>(&address), sizeof(address)) < 0 ||
        ::listen(fd, 1) < 0) {
        int owned_fd = fd;
        close_socket(owned_fd);
        return {};
    }

    socklen_t length = sizeof(address);
    if (::getsockname(fd, reinterpret_cast<sockaddr *>(&address), &length) < 0) {
        int owned_fd = fd;
        close_socket(owned_fd);
        return {};
    }
    return tcp_listener(fd, ntohs(address.sin_port));
}
tcp_listener tcp_listener::bind_loopback(uint16_t port) {
    return bind_host("127.0.0.1", port);
}
tcp_channel tcp_listener::accept(std::string & error) {
    const int peer = ::accept(fd_, nullptr, nullptr);
    if (peer < 0) {
        error = socket_error("accept");
        return {};
    }
    tcp_channel channel(peer);
    // The accepted side of the handshake: the peer must deliver its config
    // (or connect and request one) within the handshake window or the worker
    // surfaces an error instead of parking silently.
    channel.set_timeouts(handshake_timeout_s(), handshake_timeout_s());
    return channel;
}

bool tcp_listener::valid() const {
    return fd_ >= 0;
}

uint16_t tcp_listener::port() const {
    return port_;
}


tcp_channel connect_retry(const std::string & host, uint16_t port, int attempts,
                          int delay_ms, std::string & error) {
    for (int attempt = 0; attempt < attempts; ++attempt) {
        std::string err;
        tcp_channel channel = tcp_channel::connect_host(host, port, err);
        if (channel.valid()) {
            return channel;
        }
        if (attempt + 1 < attempts) {
            ::usleep(static_cast<useconds_t>(delay_ms) * 1000u);
        } else {
            error = err;
        }
    }
    return {};
}

} // namespace potluck

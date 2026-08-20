#include "prima-distributed-transport.h"

#include <arpa/inet.h>
#include <cerrno>
#include <cstring>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <vector>

namespace prima {
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
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        error = socket_error("socket");
        return {};
    }

    sockaddr_in address = {};
    address.sin_family = AF_INET;
    address.sin_port = htons(port);
    if (::inet_pton(AF_INET, host.c_str(), &address.sin_addr) != 1) {
        error = "connect: host must be an IPv4 address";
        int owned_fd = fd;
        close_socket(owned_fd);
        return {};
    }

    if (::connect(fd, reinterpret_cast<const sockaddr *>(&address), sizeof(address)) < 0) {
        error = socket_error("connect");
        int owned_fd = fd;
        close_socket(owned_fd);
        return {};
    }
    return tcp_channel(fd);
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

bool tcp_listener::valid() const {
    return fd_ >= 0;
}

uint16_t tcp_listener::port() const {
    return port_;
}

tcp_channel tcp_listener::accept(std::string & error) {
    const int peer = ::accept(fd_, nullptr, nullptr);
    if (peer < 0) {
        error = socket_error("accept");
        return {};
    }
    return tcp_channel(peer);
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

} // namespace prima

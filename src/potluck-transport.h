#pragma once

#include "potluck-protocol.h"

#include <cstdint>
#include <string>

namespace potluck {

class tcp_channel {
public:
    tcp_channel() = default;
    explicit tcp_channel(int fd);
    ~tcp_channel();

    tcp_channel(const tcp_channel &) = delete;
    tcp_channel & operator=(const tcp_channel &) = delete;
    tcp_channel(tcp_channel && other) noexcept;
    tcp_channel & operator=(tcp_channel && other) noexcept;

    bool valid() const;
    bool send(const message & message, std::string & error);
    bool receive(message & message, std::string & error);

    // SO_RCVTIMEO / SO_SNDTIMEO in seconds. A stalled peer then surfaces as a
    // read/write error instead of a forever-blocking recv()/send().
    void set_timeouts(int rcv_seconds, int snd_seconds);

    static tcp_channel connect_host(const std::string & host, uint16_t port, std::string & error);
    static tcp_channel connect_loopback(uint16_t port, std::string & error);

private:
    int fd_ = -1;

    bool write_all(const uint8_t * data, size_t size, std::string & error);
    bool read_all(uint8_t * data, size_t size, std::string & error);
};

class tcp_listener {
public:
    tcp_listener() = default;
    ~tcp_listener();

    tcp_listener(const tcp_listener &) = delete;
    tcp_listener & operator=(const tcp_listener &) = delete;
    tcp_listener(tcp_listener && other) noexcept;
    tcp_listener & operator=(tcp_listener && other) noexcept;

    static tcp_listener bind_host(const std::string & host, uint16_t port);
    static tcp_listener bind_loopback(uint16_t port);
    bool valid() const;
    uint16_t port() const;
    tcp_channel accept(std::string & error);

private:
    int fd_ = -1;
    uint16_t port_ = 0;

    tcp_listener(int fd, uint16_t port);
};


// Socket timeouts in seconds. Handshake covers config exchange (workers may
// still be loading weights); decode covers per-token round trips on slow
// hardware. Overridable: POTLUCK_TIMEOUT_HANDSHAKE_S / POTLUCK_TIMEOUT_DECODE_S.
int handshake_timeout_s();
int decode_timeout_s();

// connect_host that retries for up to `attempts` times, sleeping `delay_ms`
// between failures. Neighbor stages in a chain start asynchronously (each does
// its own Metal/backend init), so peers may not be listening yet when we try.
tcp_channel connect_retry(const std::string & host, uint16_t port, int attempts,
                          int delay_ms, std::string & error);

} // namespace potluck

#pragma once

#include "potluck-protocol.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace potluck {

constexpr size_t curve_z85_key_size = 40;
constexpr size_t curve_bootstrap_record_size = 8 + 1 + 4 + 4 + 8 + 6 * curve_z85_key_size;

struct curve_keypair {
    std::string public_key;
    std::string secret_key;

    bool valid() const noexcept {
        return public_key.size() == curve_z85_key_size &&
               secret_key.size() == curve_z85_key_size;
    }
};

struct curve_client_credentials {
    curve_keypair keypair;
    std::string server_public_key;
};

struct curve_bootstrap_credentials {
    uint32_t rank = 0;
    uint32_t peer_count = 0;
    uint64_t generation = 0;
    curve_keypair keypair;
    std::string previous_server_key;
    std::string next_server_key;
    std::string result_server_key;
    std::string controller_public_key;
};

using curve_public_key_list = std::vector<std::string>;

bool generate_curve_keypair(curve_keypair & output, std::string & error);
bool valid_curve_z85_key(const std::string & key);
bool encode_curve_bootstrap(const curve_bootstrap_credentials & input,
                            std::vector<uint8_t> & output, std::string & error);
bool decode_curve_bootstrap(const uint8_t * data, size_t size,
                            curve_bootstrap_credentials & output, std::string & error);
void scrub_curve_keypair(curve_keypair & keypair) noexcept;
void scrub_curve_credentials(curve_bootstrap_credentials & credentials) noexcept;

// The metadata frame is intentionally small and independent of the payload
// helpers used by the model protocol.
constexpr uint8_t ring_message_version = 1;
constexpr size_t ring_max_shape_dims = 64;
constexpr size_t ring_max_metadata_bytes = 4096;

class ring_receiver {
public:
    ring_receiver() = default;
    ~ring_receiver();

    ring_receiver(const ring_receiver &) = delete;
    ring_receiver & operator=(const ring_receiver &) = delete;
    ring_receiver(ring_receiver && other) noexcept;
    ring_receiver & operator=(ring_receiver && other) noexcept;

    static ring_receiver bind(const std::string & endpoint,
                              const curve_keypair & keypair,
                              const curve_public_key_list & allowed_clients,
                              std::string & error);

    bool valid() const noexcept;
    const std::string & endpoint() const noexcept;
    bool set_receive_timeout(int timeout_ms, std::string & error);
    bool receive(message & output, std::string & error);

private:
    ring_receiver(void * context, void * socket, void * zap_state, std::string endpoint);

    void * context_ = nullptr;
    void * socket_ = nullptr;
    void * zap_state_ = nullptr;
    std::string endpoint_;
};

class ring_sender {
public:
    ring_sender() = default;
    ~ring_sender();

    ring_sender(const ring_sender &) = delete;
    ring_sender & operator=(const ring_sender &) = delete;
    ring_sender(ring_sender && other) noexcept;
    ring_sender & operator=(ring_sender && other) noexcept;

    static ring_sender connect(const std::string & endpoint,
                               const curve_client_credentials & credentials,
                               std::string & error);

    bool valid() const noexcept;
    const std::string & endpoint() const noexcept;
    bool set_send_timeout(int timeout_ms, std::string & error);
    bool send(const message & input, std::string & error);

private:
    ring_sender(void * context, void * socket, std::string endpoint);

    void * context_ = nullptr;
    void * socket_ = nullptr;
    std::string endpoint_;
};

// A direct adjacent-peer ring endpoint: PULL is bound locally and PUSH is
// connected to the next peer. The sender and receiver primitives above are
// available when a controller needs a short-lived one-way connection.
class ring_peer {
public:
    ring_peer() = default;
    ~ring_peer() = default;

    ring_peer(const ring_peer &) = delete;
    ring_peer & operator=(const ring_peer &) = delete;
    ring_peer(ring_peer && other) noexcept = default;
    ring_peer & operator=(ring_peer && other) noexcept = default;

    static ring_peer bind(const std::string & local_endpoint,
                          const curve_keypair & keypair,
                          const curve_public_key_list & allowed_clients,
                          std::string & error);
    static ring_peer bind(const std::string & local_endpoint,
                          const curve_keypair & keypair,
                          const std::string & next_endpoint,
                          const std::string & next_server_key,
                          const curve_public_key_list & allowed_clients,
                          std::string & error);

    bool connect_next(const std::string & next_endpoint,
                      const curve_keypair & keypair,
                      const std::string & next_server_key,
                      std::string & error);

    bool valid() const noexcept;
    bool receive_valid() const noexcept;
    bool send_valid() const noexcept;
    const std::string & endpoint() const noexcept;
    const std::string & next_endpoint() const noexcept;

    bool set_timeouts(int receive_timeout_ms, int send_timeout_ms, std::string & error);
    bool receive(message & output, std::string & error);
    bool send(const message & input, std::string & error);

    ring_receiver & receiver() noexcept;
    const ring_receiver & receiver() const noexcept;
    ring_sender & sender() noexcept;
    const ring_sender & sender() const noexcept;

private:
    ring_receiver receiver_;
    ring_sender sender_;
    std::string next_endpoint_;
};

} // namespace potluck

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace potluck {

constexpr uint32_t discovery_protocol_version = 1;

struct discovered_node {
    std::string id;
    std::string instance;
    std::string host;
    std::string ssh_user;
    uint16_t ssh_port = 0;
    uint16_t ring_port = 0;
    bool available = false;
    uint32_t protocol_version = 0;
};

class node_advertisement {
public:
    node_advertisement() = default;
    ~node_advertisement();

    node_advertisement(const node_advertisement &) = delete;
    node_advertisement & operator=(const node_advertisement &) = delete;
    node_advertisement(node_advertisement && other) noexcept;
    node_advertisement & operator=(node_advertisement && other) noexcept;

    static node_advertisement start(const std::string & id,
                                    const std::string & instance,
                                    const std::string & ssh_user,
                                    uint16_t ssh_port,
                                    uint16_t ring_port,
                                    std::string & error);

    bool valid() const noexcept;
    bool process(int timeout_ms, std::string & error);

private:
    void reset() noexcept;

    void * service_ref_ = nullptr;
    void * registration_state_ = nullptr;
};

std::vector<discovered_node> discover_nodes(int timeout_ms, std::string & error);

} // namespace potluck

#include "potluck-discovery.h"

#include <arpa/inet.h>
#include <cerrno>
#include <climits>
#include <dns_sd.h>
#include <poll.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <string_view>
#include <utility>
#include <vector>

namespace potluck {
namespace {

constexpr const char * service_type = "_potluck._tcp";
constexpr const char * service_type_reply = "_potluck._tcp.";
constexpr const char * service_domain = "local.";
constexpr size_t max_txt_value_length = 255;

std::string dns_error(const char * operation, DNSServiceErrorType code) {
    return std::string(operation) + " failed (DNS-SD error " + std::to_string(static_cast<int>(code)) + ")";
}

void set_error(std::string & error, std::string value) {
    error = std::move(value);
}
void prepare_dnssd() {
#if defined(__linux__)
    static const bool warning_disabled = setenv("AVAHI_COMPAT_NOWARN", "1", 0) == 0;
    (void) warning_disabled;
#endif
}


bool valid_text(std::string_view value, size_t max_length = max_txt_value_length) {
    if (value.empty() || value.size() > max_length) {
        return false;
    }
    for (const unsigned char byte : value) {
        if (byte < 0x20 || byte == 0x7f) {
            return false;
        }
    }
    return true;
}

bool valid_service_name(std::string_view value) {
    return valid_text(value, 63);
}

bool parse_u32(std::string_view text, uint32_t & output) {
    if (text.empty()) {
        return false;
    }

    uint64_t value = 0;
    for (const unsigned char byte : text) {
        if (byte < '0' || byte > '9') {
            return false;
        }
        value = value * 10 + static_cast<uint64_t>(byte - '0');
        if (value > std::numeric_limits<uint32_t>::max()) {
            return false;
        }
    }
    output = static_cast<uint32_t>(value);
    return true;
}

bool parse_port(std::string_view text, uint16_t & output) {
    uint32_t value = 0;
    if (!parse_u32(text, value) || value == 0 || value > std::numeric_limits<uint16_t>::max()) {
        return false;
    }
    output = static_cast<uint16_t>(value);
    return true;
}

class txt_record_guard {
public:
    explicit txt_record_guard(uint16_t buffer_length, void * buffer) {
        TXTRecordCreate(&record_, buffer_length, buffer);
    }

    ~txt_record_guard() {
        TXTRecordDeallocate(&record_);
    }

    txt_record_guard(const txt_record_guard &) = delete;
    txt_record_guard & operator=(const txt_record_guard &) = delete;

    TXTRecordRef * get() noexcept {
        return &record_;
    }

private:
    TXTRecordRef record_{};
};

class dns_ref_guard {
public:
    explicit dns_ref_guard(DNSServiceRef ref = nullptr) : ref_(ref) {}

    ~dns_ref_guard() {
        reset();
    }

    dns_ref_guard(const dns_ref_guard &) = delete;
    dns_ref_guard & operator=(const dns_ref_guard &) = delete;

    DNSServiceRef get() const noexcept {
        return ref_;
    }

    DNSServiceRef release() noexcept {
        DNSServiceRef result = ref_;
        ref_ = nullptr;
        return result;
    }

    void reset(DNSServiceRef ref = nullptr) noexcept {
        if (ref_ != nullptr) {
            DNSServiceRefDeallocate(ref_);
        }
        ref_ = ref;
    }

private:
    DNSServiceRef ref_ = nullptr;
};

struct registration_state {
    bool callback_called = false;
    DNSServiceErrorType callback_error = kDNSServiceErr_NoError;
};

void DNSSD_API registration_callback(DNSServiceRef,
                                     DNSServiceFlags,
                                     DNSServiceErrorType error_code,
                                     const char *,
                                     const char *,
                                     const char *,
                                     void * context) {
    auto * state = static_cast<registration_state *>(context);
    if (state == nullptr) {
        return;
    }
    state->callback_called = true;
    state->callback_error = error_code;
}

bool parse_txt_record(uint16_t txt_length,
                      const unsigned char * txt_record,
                      std::string & id,
                      uint32_t & protocol_version,
                      std::string & user,
                      uint16_t & ring_port,
                      bool & available) {
    if (txt_record == nullptr || txt_length == 0) {
        return false;
    }

    const uint16_t count = TXTRecordGetCount(txt_length, txt_record);
    if (count == 0) {
        return false;
    }

    bool have_id = false;
    bool have_version = false;
    bool have_user = false;
    bool have_ring_port = false;
    bool have_state = false;
    std::array<std::string, 5> keys;
    size_t key_count = 0;

    for (uint16_t index = 0; index < count; ++index) {
        char key_buffer[256] = {};
        uint8_t value_length = 0;
        const void * value_data = nullptr;
        const DNSServiceErrorType result = TXTRecordGetItemAtIndex(txt_length,
                                                                    txt_record,
                                                                    index,
                                                                    sizeof(key_buffer),
                                                                    key_buffer,
                                                                    &value_length,
                                                                    &value_data);
        if (result != kDNSServiceErr_NoError || key_buffer[0] == '\0') {
            return false;
        }

        const std::string_view key(key_buffer);
        const bool known_key = key == "id" || key == "version" || key == "user" ||
                               key == "ring_port" || key == "state";
        if (known_key) {
            if (key_count == keys.size()) {
                return false;
            }
            for (size_t prior = 0; prior < key_count; ++prior) {
                if (keys[prior] == key) {
                    return false;
                }
            }
            keys[key_count++] = std::string(key);
        }

        if (value_data == nullptr && known_key) {
            return false;
        }
        const std::string_view value = value_data == nullptr
            ? std::string_view()
            : std::string_view(static_cast<const char *>(value_data), value_length);

        if (key == "id") {
            if (!valid_text(value)) {
                return false;
            }
            id.assign(value.data(), value.size());
            have_id = true;
        } else if (key == "version") {
            if (!valid_text(value, 10) || !parse_u32(value, protocol_version)) {
                return false;
            }
            have_version = true;
        } else if (key == "user") {
            if (!valid_text(value)) {
                return false;
            }
            user.assign(value.data(), value.size());
            have_user = true;
        } else if (key == "ring_port") {
            if (!valid_text(value, 5) || !parse_port(value, ring_port)) {
                return false;
            }
            have_ring_port = true;
        } else if (key == "state") {
            if (!valid_text(value)) {
                return false;
            }
            available = value == "available";
            have_state = true;
        }
    }

    return have_id && have_version && have_user && have_ring_port && have_state;
}

struct browse_entry {
    std::string service_name;
    std::string regtype;
    std::string domain;
    uint32_t interface_index = 0;
    bool started = false;
};

struct discovery_state;

struct resolve_request {
    explicit resolve_request(discovery_state * owner) : state(owner) {}

    ~resolve_request() {
        if (ref != nullptr) {
            DNSServiceRefDeallocate(ref);
        }
    }

    resolve_request(const resolve_request &) = delete;
    resolve_request & operator=(const resolve_request &) = delete;

    discovery_state * state = nullptr;
    DNSServiceRef ref = nullptr;
    browse_entry entry;
    bool complete = false;
};

struct discovery_state {
    DNSServiceRef browse_ref = nullptr;
    std::vector<browse_entry> entries;
    std::vector<std::unique_ptr<resolve_request>> active;
    std::vector<discovered_node> nodes;
    std::string callback_error;
};

void add_discovered_node(discovery_state & state, discovered_node candidate) {
    const auto same_id = std::find_if(state.nodes.begin(), state.nodes.end(),
                                      [&candidate](const discovered_node & existing) {
                                          return existing.id == candidate.id;
                                      });
    if (same_id == state.nodes.end()) {
        state.nodes.emplace_back(std::move(candidate));
        return;
    }

    const auto less_node = [](const discovered_node & lhs, const discovered_node & rhs) {
        if (lhs.id != rhs.id) {
            return lhs.id < rhs.id;
        }
        if (lhs.instance != rhs.instance) {
            return lhs.instance < rhs.instance;
        }
        if (lhs.host != rhs.host) {
            return lhs.host < rhs.host;
        }
        if (lhs.ssh_user != rhs.ssh_user) {
            return lhs.ssh_user < rhs.ssh_user;
        }
        if (lhs.ssh_port != rhs.ssh_port) {
            return lhs.ssh_port < rhs.ssh_port;
        }
        if (lhs.ring_port != rhs.ring_port) {
            return lhs.ring_port < rhs.ring_port;
        }
        if (lhs.protocol_version != rhs.protocol_version) {
            return lhs.protocol_version < rhs.protocol_version;
        }
        return lhs.available < rhs.available;
    };
    if (less_node(candidate, *same_id)) {
        *same_id = std::move(candidate);
    }
}

void DNSSD_API browse_callback(DNSServiceRef,
                               DNSServiceFlags flags,
                               uint32_t interface_index,
                               DNSServiceErrorType error_code,
                               const char * service_name,
                               const char * regtype,
                               const char * reply_domain,
                               void * context) {
    auto * state = static_cast<discovery_state *>(context);
    if (state == nullptr) {
        return;
    }
    if (error_code != kDNSServiceErr_NoError) {
        state->callback_error = dns_error("DNSServiceBrowse", error_code);
        return;
    }
    if ((flags & kDNSServiceFlagsAdd) == 0 || service_name == nullptr || regtype == nullptr ||
        reply_domain == nullptr ||
        (std::string_view(regtype) != service_type && std::string_view(regtype) != service_type_reply) ||
        std::string_view(reply_domain) != service_domain || !valid_service_name(service_name)) {
        return;
    }

    const std::string_view service(service_name);
    const std::string_view type(regtype);
    const std::string_view domain(reply_domain);
    const auto duplicate = std::find_if(state->entries.begin(), state->entries.end(),
                                        [&](const browse_entry & entry) {
                                            return entry.interface_index == interface_index &&
                                                   entry.service_name == service &&
                                                   entry.regtype == type && entry.domain == domain;
                                        });
    if (duplicate != state->entries.end()) {
        return;
    }

    state->entries.push_back({std::string(service), std::string(type), std::string(domain), interface_index});
}

void DNSSD_API resolve_callback(DNSServiceRef,
                                DNSServiceFlags,
                                uint32_t,
                                DNSServiceErrorType error_code,
                                const char *,
                                const char * host_target,
                                uint16_t port,
                                uint16_t txt_length,
                                const unsigned char * txt_record,
                                void * context) {
    auto * request = static_cast<resolve_request *>(context);
    if (request == nullptr) {
        return;
    }
    request->complete = true;
    if (error_code != kDNSServiceErr_NoError || host_target == nullptr || port == 0) {
        return;
    }

    discovered_node candidate;
    candidate.instance = request->entry.service_name;
    candidate.host.assign(host_target);
    candidate.ssh_port = ntohs(port);
    candidate.ring_port = 0;
    if (!valid_text(candidate.host) || candidate.ssh_port == 0 ||
        !parse_txt_record(txt_length,
                          txt_record,
                          candidate.id,
                          candidate.protocol_version,
                          candidate.ssh_user,
                          candidate.ring_port,
                          candidate.available)) {
        return;
    }
    add_discovered_node(*request->state, std::move(candidate));
}

void start_pending_resolves(discovery_state & state) {
    for (browse_entry & entry : state.entries) {
        if (entry.started) {
            continue;
        }
        entry.started = true;

        auto request = std::make_unique<resolve_request>(&state);
        request->entry = std::move(entry);
        const DNSServiceErrorType result = DNSServiceResolve(&request->ref,
                                                             0,
                                                             request->entry.interface_index,
                                                             request->entry.service_name.c_str(),
                                                             request->entry.regtype.c_str(),
                                                             request->entry.domain.c_str(),
                                                             resolve_callback,
                                                             request.get());
        if (result == kDNSServiceErr_NoError) {
            state.active.emplace_back(std::move(request));
        }
    }
}

void reap_completed_resolves(discovery_state & state) {
    state.active.erase(std::remove_if(state.active.begin(),
                                      state.active.end(),
                                      [](const std::unique_ptr<resolve_request> & request) {
                                          return request->complete;
                                      }),
                        state.active.end());
}


} // namespace

node_advertisement::~node_advertisement() {
    reset();
}

node_advertisement::node_advertisement(node_advertisement && other) noexcept
    : service_ref_(other.service_ref_), registration_state_(other.registration_state_) {
    other.service_ref_ = nullptr;
    other.registration_state_ = nullptr;
}

node_advertisement & node_advertisement::operator=(node_advertisement && other) noexcept {
    if (this != &other) {
        reset();
        service_ref_ = other.service_ref_;
        registration_state_ = other.registration_state_;
        other.service_ref_ = nullptr;
        other.registration_state_ = nullptr;
    }
    return *this;
}

void node_advertisement::reset() noexcept {
    if (service_ref_ != nullptr) {
        DNSServiceRefDeallocate(static_cast<DNSServiceRef>(service_ref_));
        service_ref_ = nullptr;
    }
    delete static_cast<registration_state *>(registration_state_);
    registration_state_ = nullptr;
}

node_advertisement node_advertisement::start(const std::string & id,
                                             const std::string & instance,
                                             const std::string & ssh_user,
                                             uint16_t ssh_port,
                                             uint16_t ring_port,
                                             std::string & error) {
    error.clear();
    node_advertisement advertisement;
    if (!valid_text(id) || !valid_service_name(instance) || !valid_text(ssh_user) || ssh_port == 0 ||
        ring_port == 0) {
        set_error(error, "invalid Potluck DNS-SD advertisement fields");
        return advertisement;
    }
    prepare_dnssd();

    std::array<uint8_t, 512> txt_buffer{};
    txt_record_guard txt(sizeof(txt_buffer), txt_buffer.data());
    const std::string version = std::to_string(discovery_protocol_version);
    const std::string ring = std::to_string(ring_port);
    const std::array<std::pair<const char *, std::string_view>, 5> fields = {{
        {"id", id},
        {"version", version},
        {"user", ssh_user},
        {"ring_port", ring},
        {"state", "available"},
    }};
    for (const auto & field : fields) {
        const DNSServiceErrorType result = TXTRecordSetValue(txt.get(),
                                                             field.first,
                                                             static_cast<uint8_t>(field.second.size()),
                                                             field.second.data());
        if (result != kDNSServiceErr_NoError) {
            set_error(error, dns_error("TXTRecordSetValue", result));
            return advertisement;
        }
    }

    auto state = std::make_unique<registration_state>();
    DNSServiceRef service_ref = nullptr;
    const DNSServiceErrorType result = DNSServiceRegister(&service_ref,
                                                          0,
                                                          0,
                                                          instance.c_str(),
                                                          service_type,
                                                          service_domain,
                                                          nullptr,
                                                          htons(ssh_port),
                                                          TXTRecordGetLength(txt.get()),
                                                          TXTRecordGetBytesPtr(txt.get()),
                                                          registration_callback,
                                                          state.get());
    if (result != kDNSServiceErr_NoError) {
        set_error(error, dns_error("DNSServiceRegister", result));
        return advertisement;
    }

    advertisement.service_ref_ = service_ref;
    advertisement.registration_state_ = state.release();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    auto * registration = static_cast<registration_state *>(advertisement.registration_state_);
    while (registration != nullptr && !registration->callback_called) {
        const auto remaining =
            std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now()).count();
        if (remaining <= 0) {
            set_error(error, "Potluck DNS-SD registration timed out");
            advertisement.reset();
            return advertisement;
        }
        if (!advertisement.process(static_cast<int>(std::min<int64_t>(remaining, INT_MAX)), error)) {
            advertisement.reset();
            return advertisement;
        }
    }
    return advertisement;
}

bool node_advertisement::valid() const noexcept {
    return service_ref_ != nullptr;
}

bool node_advertisement::process(int timeout_ms, std::string & error) {
    error.clear();
    if (!valid()) {
        set_error(error, "Potluck DNS-SD advertisement is not valid");
        return false;
    }
    if (timeout_ms < 0) {
        set_error(error, "DNS-SD process timeout must be non-negative");
        return false;
    }

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    for (;;) {
        const auto now = std::chrono::steady_clock::now();
        const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count();
        if (remaining <= 0) {
            break;
        }

        const int socket = static_cast<int>(DNSServiceRefSockFD(static_cast<DNSServiceRef>(service_ref_)));
        if (socket < 0) {
            set_error(error, "DNS-SD advertisement has no valid socket");
            return false;
        }
        struct pollfd descriptor {
            socket, POLLIN, 0
        };
        const int result = ::poll(&descriptor, 1, static_cast<int>(std::min<int64_t>(remaining, INT_MAX)));
        if (result < 0) {
            if (errno == EINTR) {
                continue;
            }
            set_error(error, "poll failed while processing DNS-SD advertisement");
            return false;
        }
        if (result == 0) {
            break;
        }
        if ((descriptor.revents & (POLLERR | POLLHUP | POLLNVAL)) != 0) {
            set_error(error, "DNS-SD advertisement socket closed");
            return false;
        }
        const DNSServiceErrorType process_result = DNSServiceProcessResult(static_cast<DNSServiceRef>(service_ref_));
        if (process_result != kDNSServiceErr_NoError) {
            set_error(error, dns_error("DNSServiceProcessResult", process_result));
            return false;
        }
        const auto * state = static_cast<const registration_state *>(registration_state_);
        if (state != nullptr && state->callback_called && state->callback_error != kDNSServiceErr_NoError) {
            set_error(error, dns_error("DNSServiceRegister", state->callback_error));
            return false;
        }
        return true;
    }

    const auto * state = static_cast<const registration_state *>(registration_state_);
    if (state != nullptr && state->callback_called && state->callback_error != kDNSServiceErr_NoError) {
        set_error(error, dns_error("DNSServiceRegister", state->callback_error));
        return false;
    }
    return true;
}

std::vector<discovered_node> discover_nodes(int timeout_ms, std::string & error) {
    error.clear();
    std::vector<discovered_node> empty;
    prepare_dnssd();
    if (timeout_ms < 0) {
        set_error(error, "DNS-SD discovery timeout must be non-negative");
        return empty;
    }

    discovery_state state;
    const DNSServiceErrorType browse_result = DNSServiceBrowse(&state.browse_ref,
                                                               0,
                                                               0,
                                                               service_type,
                                                               service_domain,
                                                               browse_callback,
                                                               &state);
    dns_ref_guard browse_guard(state.browse_ref);
    if (browse_result != kDNSServiceErr_NoError) {
        set_error(error, dns_error("DNSServiceBrowse", browse_result));
        return empty;
    }

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    for (;;) {
        start_pending_resolves(state);
        reap_completed_resolves(state);

        const auto now = std::chrono::steady_clock::now();
        const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count();
        if (remaining <= 0) {
            break;
        }

        std::vector<struct pollfd> descriptors;
        std::vector<resolve_request *> requests;
        descriptors.reserve(state.active.size() + 1);
        requests.reserve(state.active.size() + 1);
        const int browse_socket = static_cast<int>(DNSServiceRefSockFD(state.browse_ref));
        if (browse_socket < 0) {
            set_error(error, "DNS-SD browse has no valid socket");
            return empty;
        }
        descriptors.push_back({browse_socket, POLLIN, 0});
        requests.push_back(nullptr);
        for (const auto & request : state.active) {
            const int resolve_socket = static_cast<int>(DNSServiceRefSockFD(request->ref));
            if (resolve_socket < 0) {
                set_error(error, "DNS-SD resolve has no valid socket");
                return empty;
            }
            descriptors.push_back({resolve_socket, POLLIN, 0});
            requests.push_back(request.get());
        }

        const int poll_result = ::poll(descriptors.data(),
                                       static_cast<nfds_t>(descriptors.size()),
                                       static_cast<int>(std::min<int64_t>(remaining, INT_MAX)));
        if (poll_result < 0) {
            if (errno == EINTR) {
                continue;
            }
            set_error(error, "poll failed while browsing DNS-SD services");
            return empty;
        }
        if (poll_result == 0) {
            break;
        }

        bool processed = false;
        for (size_t index = 0; index < descriptors.size(); ++index) {
            const short events = descriptors[index].revents;
            if (events == 0) {
                continue;
            }
            if ((events & (POLLERR | POLLHUP | POLLNVAL)) != 0) {
                set_error(error, "DNS-SD service socket closed");
                return empty;
            }
            DNSServiceRef ref = index == 0 ? state.browse_ref : requests[index]->ref;
            const DNSServiceErrorType process_result = DNSServiceProcessResult(ref);
            if (process_result != kDNSServiceErr_NoError) {
                set_error(error, dns_error("DNSServiceProcessResult", process_result));
                return empty;
            }
            processed = true;
        }
        if (!processed && state.callback_error.empty()) {
            continue;
        }
        if (!state.callback_error.empty()) {
            set_error(error, state.callback_error);
            return empty;
        }
    }

    if (!state.callback_error.empty()) {
        set_error(error, state.callback_error);
        return empty;
    }

    std::sort(state.nodes.begin(), state.nodes.end(),
              [](const discovered_node & lhs, const discovered_node & rhs) {
                  return lhs.id < rhs.id;
              });
    return std::move(state.nodes);
}

} // namespace potluck

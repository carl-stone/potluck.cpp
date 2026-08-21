// Historical check for the legacy raw-TCP component transport. ADR 0007
// requires replacement coverage for the direct-peer ZeroMQ ring.

#include "../src/potluck-protocol.h"
#include "../src/potluck-transport.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

// Runtime check that survives both Debug and Release (-DNDEBUG) builds.
#define CHECK(cond)                                                                       \
    do {                                                                                  \
        if (!(cond)) {                                                                    \
            std::fprintf(stderr, "CHECK failed: %s (%s:%d)\n", #cond, __FILE__, __LINE__); \
            std::abort();                                                                 \
        }                                                                                 \
    } while (0)

int main() {
    // bind_loopback(0) picks an ephemeral port; valid() + non-zero port.
    potluck::tcp_listener listener = potluck::tcp_listener::bind_loopback(0);
    CHECK(listener.valid());
    const uint16_t port = listener.port();
    CHECK(port != 0);

    // A listener is move-constructible and the moved-from handle is invalid.
    potluck::tcp_listener moved_listener = std::move(listener);
    CHECK(moved_listener.valid());
    CHECK(!listener.valid());

    potluck::message sent;
    sent.type = potluck::message_type::hidden_state;
    sent.rank = 0;
    sent.sequence = 42;
    sent.dtype = potluck::data_type::f32;
    sent.shape = {1, 1, 16};
    sent.payload.assign(64, 0xab);

    std::thread server([&] {
        std::string error;
        potluck::tcp_channel peer = moved_listener.accept(error);
        CHECK(peer.valid());

        // A channel is move-assignable and the moved-from channel is invalid.
        potluck::tcp_channel owned_peer = std::move(peer);
        CHECK(owned_peer.valid());
        CHECK(!peer.valid());

        potluck::message received;
        CHECK(owned_peer.receive(received, error));
        CHECK(received.type == sent.type);
        CHECK(received.sequence == sent.sequence);
        CHECK(received.shape == sent.shape);
        CHECK(received.payload == sent.payload);
        CHECK(owned_peer.send(sent, error));
    });

    std::string error;
    potluck::tcp_channel client = potluck::tcp_channel::connect_loopback(port, error);
    CHECK(client.valid());
    CHECK(client.send(sent, error));

    potluck::message echoed;
    CHECK(client.receive(echoed, error));
    CHECK(echoed.sequence == sent.sequence);
    CHECK(echoed.payload == sent.payload);

    server.join();

    // connect_host resolves a hostname as well as a numeric IPv4 address.
    {
        std::string err;
        potluck::tcp_channel named = potluck::tcp_channel::connect_host("localhost", port, err);
        CHECK(named.valid());
    }

    // connect_loopback to a closed port fails (connection refused).
    {
        std::string err;
        potluck::tcp_channel refused = potluck::tcp_channel::connect_loopback(1, err);
        CHECK(!refused.valid());
        CHECK(!err.empty());
    }

    // bind_host rejects a non-IPv4 host string.
    {
        potluck::tcp_listener bad = potluck::tcp_listener::bind_host("not-an-ip", 0);
        CHECK(!bad.valid());
    }

    // A closed/invalid channel cannot send or receive.
    {
        std::string err;
        potluck::tcp_channel closed;
        CHECK(!closed.valid());
        CHECK(!closed.send(sent, err));
        potluck::message m;
        CHECK(!closed.receive(m, err));
    }

    return 0;
}

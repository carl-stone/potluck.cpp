#include "../src/prima-distributed-protocol.h"
#include "../src/prima-distributed-transport.h"

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
    prima::tcp_listener listener = prima::tcp_listener::bind_loopback(0);
    CHECK(listener.valid());
    const uint16_t port = listener.port();
    CHECK(port != 0);

    // A listener is move-constructible and the moved-from handle is invalid.
    prima::tcp_listener moved_listener = std::move(listener);
    CHECK(moved_listener.valid());
    CHECK(!listener.valid());

    prima::message sent;
    sent.type = prima::message_type::hidden_state;
    sent.rank = 0;
    sent.sequence = 42;
    sent.dtype = prima::data_type::f32;
    sent.shape = {1, 1, 16};
    sent.payload.assign(64, 0xab);

    std::thread server([&] {
        std::string error;
        prima::tcp_channel peer = moved_listener.accept(error);
        CHECK(peer.valid());

        // A channel is move-assignable and the moved-from channel is invalid.
        prima::tcp_channel owned_peer = std::move(peer);
        CHECK(owned_peer.valid());
        CHECK(!peer.valid());

        prima::message received;
        CHECK(owned_peer.receive(received, error));
        CHECK(received.type == sent.type);
        CHECK(received.sequence == sent.sequence);
        CHECK(received.shape == sent.shape);
        CHECK(received.payload == sent.payload);
        CHECK(owned_peer.send(sent, error));
    });

    std::string error;
    prima::tcp_channel client = prima::tcp_channel::connect_loopback(port, error);
    CHECK(client.valid());
    CHECK(client.send(sent, error));

    prima::message echoed;
    CHECK(client.receive(echoed, error));
    CHECK(echoed.sequence == sent.sequence);
    CHECK(echoed.payload == sent.payload);

    server.join();

    // connect_host rejects a non-IPv4 host string.
    {
        std::string err;
        prima::tcp_channel bad = prima::tcp_channel::connect_host("not-an-ip", port, err);
        CHECK(!bad.valid());
        CHECK(err.find("IPv4") != std::string::npos);
    }

    // connect_loopback to a closed port fails (connection refused).
    {
        std::string err;
        prima::tcp_channel refused = prima::tcp_channel::connect_loopback(1, err);
        CHECK(!refused.valid());
        CHECK(!err.empty());
    }

    // bind_host rejects a non-IPv4 host string.
    {
        prima::tcp_listener bad = prima::tcp_listener::bind_host("not-an-ip", 0);
        CHECK(!bad.valid());
    }

    // A closed/invalid channel cannot send or receive.
    {
        std::string err;
        prima::tcp_channel closed;
        CHECK(!closed.valid());
        CHECK(!closed.send(sent, err));
        prima::message m;
        CHECK(!closed.receive(m, err));
    }

    return 0;
}

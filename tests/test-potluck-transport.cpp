#include "../src/potluck-protocol.h"
#include "../src/potluck-transport.h"

#include <cstdint>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define CHECK(cond)                                                                       \
    do {                                                                                  \
        if (!(cond)) {                                                                    \
            std::fprintf(stderr, "CHECK failed: %s (%s:%d)\n", #cond, __FILE__, __LINE__); \
            std::abort();                                                                 \
        }                                                                                 \
    } while (0)

namespace {

potluck::message make_message(uint64_t sequence, uint8_t value) {
    potluck::message result;
    result.type = potluck::message_type::hidden_state;
    result.flags = 0x12345678;
    result.rank = 3;
    result.sequence = sequence;
    result.dtype = potluck::data_type::f32;
    result.shape = {2, 4, 8};
    result.payload.assign(64, value);
    return result;
}

void check_equal(const potluck::message & actual, const potluck::message & expected) {
    CHECK(actual.type == expected.type);
    CHECK(actual.flags == expected.flags);
    CHECK(actual.rank == expected.rank);
    CHECK(actual.sequence == expected.sequence);
    CHECK(actual.dtype == expected.dtype);
    CHECK(actual.shape == expected.shape);
    CHECK(actual.payload == expected.payload);
}

} // namespace

int main() {
    std::string error;
    potluck::ring_peer left = potluck::ring_peer::bind("tcp://127.0.0.1:*", error);
    CHECK(left.receive_valid());
    CHECK(!left.endpoint().empty());
    CHECK(left.endpoint().find("tcp://127.0.0.1:") == 0);

    potluck::ring_peer right = potluck::ring_peer::bind("tcp://127.0.0.1:*", error);
    CHECK(right.receive_valid());
    CHECK(!right.endpoint().empty());

    CHECK(left.connect_next(right.endpoint(), error));
    CHECK(right.connect_next(left.endpoint(), error));
    CHECK(left.valid());
    CHECK(right.valid());
    CHECK(left.next_endpoint() == right.endpoint());
    CHECK(right.next_endpoint() == left.endpoint());
    CHECK(left.set_timeouts(500, 500, error));
    CHECK(right.set_timeouts(500, 500, error));

    potluck::ring_peer moved = std::move(left);
    CHECK(moved.valid());
    CHECK(!left.valid());

    potluck::message sent = make_message(42, 0xab);
    CHECK(moved.send(sent, error));
    potluck::message received;
    CHECK(right.receive(received, error));
    check_equal(received, sent);

    potluck::message reply = make_message(43, 0xcd);
    CHECK(right.send(reply, error));
    potluck::message echoed;
    CHECK(moved.receive(echoed, error));
    check_equal(echoed, reply);

    potluck::message oversized_metadata = sent;
    oversized_metadata.shape.assign(potluck::ring_max_shape_dims + 1, 1);
    CHECK(!moved.send(oversized_metadata, error));
    CHECK(error.find("shape") != std::string::npos);

    potluck::ring_receiver timeout_receiver =
        potluck::ring_receiver::bind("tcp://127.0.0.1:*", error);
    CHECK(timeout_receiver.valid());
    CHECK(timeout_receiver.set_receive_timeout(20, error));
    potluck::message absent;
    CHECK(!timeout_receiver.receive(absent, error));
    CHECK(error.find("timeout") != std::string::npos);

    potluck::ring_sender bad_sender = potluck::ring_sender::connect("not-a-zmq-endpoint", error);
    CHECK(!bad_sender.valid());
    CHECK(!error.empty());

    potluck::ring_sender closed_sender;
    CHECK(!closed_sender.send(sent, error));
    CHECK(error.find("PUSH") != std::string::npos);
    potluck::ring_receiver closed_receiver;
    CHECK(!closed_receiver.receive(absent, error));
    CHECK(error.find("PULL") != std::string::npos);

    return 0;
}

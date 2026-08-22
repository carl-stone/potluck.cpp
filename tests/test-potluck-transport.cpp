#include "../src/potluck-protocol.h"
#include "../src/potluck-transport.h"

#include <cstdint>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>

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
    potluck::curve_keypair left_keys;
    potluck::curve_keypair right_keys;
    CHECK(potluck::generate_curve_keypair(left_keys, error));
    CHECK(potluck::generate_curve_keypair(right_keys, error));
    CHECK(left_keys.public_key != right_keys.public_key);

    potluck::curve_bootstrap_credentials bootstrap;
    bootstrap.rank = 0;
    bootstrap.peer_count = 2;
    bootstrap.generation = 17;
    bootstrap.keypair = left_keys;
    bootstrap.previous_server_key = right_keys.public_key;
    bootstrap.next_server_key = right_keys.public_key;
    bootstrap.result_server_key = left_keys.public_key;
    bootstrap.controller_public_key = left_keys.public_key;
    std::vector<uint8_t> bootstrap_record;
    CHECK(potluck::encode_curve_bootstrap(bootstrap, bootstrap_record, error));
    CHECK(bootstrap_record.size() == potluck::curve_bootstrap_record_size);
    potluck::curve_bootstrap_credentials decoded_bootstrap;
    CHECK(potluck::decode_curve_bootstrap(
        bootstrap_record.data(), bootstrap_record.size(), decoded_bootstrap, error));
    CHECK(decoded_bootstrap.rank == bootstrap.rank);
    CHECK(decoded_bootstrap.peer_count == bootstrap.peer_count);
    CHECK(decoded_bootstrap.generation == bootstrap.generation);
    CHECK(decoded_bootstrap.keypair.public_key == bootstrap.keypair.public_key);
    CHECK(decoded_bootstrap.keypair.secret_key == bootstrap.keypair.secret_key);
    CHECK(decoded_bootstrap.previous_server_key == bootstrap.previous_server_key);
    CHECK(decoded_bootstrap.next_server_key == bootstrap.next_server_key);
    CHECK(decoded_bootstrap.result_server_key == bootstrap.result_server_key);
    CHECK(decoded_bootstrap.controller_public_key == bootstrap.controller_public_key);
    bootstrap_record[0] ^= 1;
    CHECK(!potluck::decode_curve_bootstrap(
        bootstrap_record.data(), bootstrap_record.size(), decoded_bootstrap, error));
    CHECK(!error.empty());

    potluck::ring_peer left = potluck::ring_peer::bind(
        "tcp://127.0.0.1:*", left_keys, { right_keys.public_key }, error);
    CHECK(left.receive_valid());
    CHECK(!left.endpoint().empty());
    CHECK(left.endpoint().find("tcp://127.0.0.1:") == 0);

    potluck::ring_peer right = potluck::ring_peer::bind(
        "tcp://127.0.0.1:*", right_keys, { left_keys.public_key }, error);
    CHECK(right.receive_valid());
    CHECK(!right.endpoint().empty());

    CHECK(left.connect_next(right.endpoint(), left_keys, right_keys.public_key, error));
    CHECK(right.connect_next(left.endpoint(), right_keys, left_keys.public_key, error));
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
    potluck::message heartbeat;
    heartbeat.type = potluck::message_type::heartbeat;
    heartbeat.rank = 1;
    heartbeat.sequence = 44;
    CHECK(moved.send(heartbeat, error));
    potluck::message received_heartbeat;
    CHECK(right.receive(received_heartbeat, error));
    check_equal(received_heartbeat, heartbeat);

    potluck::message heartbeat_ack;
    heartbeat_ack.type = potluck::message_type::heartbeat_ack;
    heartbeat_ack.rank = 1;
    heartbeat_ack.sequence = heartbeat.sequence;
    CHECK(right.send(heartbeat_ack, error));
    potluck::message received_ack;
    CHECK(moved.receive(received_ack, error));
    check_equal(received_ack, heartbeat_ack);
    potluck::message logprob_result = sent;
    logprob_result.type = potluck::message_type::batch_result_logprobs;
    logprob_result.dtype = potluck::data_type::i32;
    logprob_result.shape = { logprob_result.payload.size() };
    CHECK(moved.send(logprob_result, error));
    potluck::message received_logprob_result;
    CHECK(right.receive(received_logprob_result, error));
    check_equal(received_logprob_result, logprob_result);

    potluck::message oversized_metadata = sent;
    oversized_metadata.shape.assign(potluck::ring_max_shape_dims + 1, 1);
    CHECK(!moved.send(oversized_metadata, error));
    CHECK(error.find("shape") != std::string::npos);

    potluck::curve_keypair timeout_keys;
    CHECK(potluck::generate_curve_keypair(timeout_keys, error));
    potluck::ring_receiver timeout_receiver =
        potluck::ring_receiver::bind(
            "tcp://127.0.0.1:*", timeout_keys, { timeout_keys.public_key }, error);
    CHECK(timeout_receiver.valid());
    CHECK(timeout_receiver.set_receive_timeout(20, error));
    potluck::message absent;
    CHECK(!timeout_receiver.receive(absent, error));
    CHECK(error.find("timeout") != std::string::npos);

    potluck::curve_keypair wrong_keys;
    CHECK(potluck::generate_curve_keypair(wrong_keys, error));
    potluck::ring_receiver protected_receiver =
        potluck::ring_receiver::bind(
            "tcp://127.0.0.1:*", timeout_keys, { timeout_keys.public_key }, error);
    CHECK(protected_receiver.valid());
    potluck::curve_client_credentials unauthorized_client;
    unauthorized_client.keypair = wrong_keys;
    unauthorized_client.server_public_key = timeout_keys.public_key;
    potluck::ring_sender unauthorized_sender =
        potluck::ring_sender::connect(protected_receiver.endpoint(), unauthorized_client, error);
    CHECK(unauthorized_sender.valid());
    CHECK(unauthorized_sender.set_send_timeout(100, error));
    CHECK(!unauthorized_sender.send(sent, error));
    potluck::curve_client_credentials wrong_server;
    wrong_server.keypair = timeout_keys;
    wrong_server.server_public_key = wrong_keys.public_key;
    potluck::ring_sender wrong_sender =
        potluck::ring_sender::connect(protected_receiver.endpoint(), wrong_server, error);
    CHECK(wrong_sender.valid());
    CHECK(wrong_sender.set_send_timeout(100, error));
    CHECK(!wrong_sender.send(sent, error));
    CHECK(!error.empty());

    potluck::curve_client_credentials invalid_credentials;
    potluck::ring_sender invalid_sender =
        potluck::ring_sender::connect("tcp://127.0.0.1:1", invalid_credentials, error);
    CHECK(!invalid_sender.valid());
    CHECK(error.find("CURVE") != std::string::npos);

    potluck::curve_client_credentials bad_endpoint;
    bad_endpoint.keypair = left_keys;
    bad_endpoint.server_public_key = right_keys.public_key;
    potluck::ring_sender bad_sender =
        potluck::ring_sender::connect("not-a-zmq-endpoint", bad_endpoint, error);
    CHECK(!bad_sender.valid());
    CHECK(!error.empty());

    potluck::ring_sender closed_sender;
    CHECK(!closed_sender.send(sent, error));
    potluck::scrub_curve_credentials(bootstrap);
    potluck::scrub_curve_credentials(decoded_bootstrap);
    potluck::scrub_curve_keypair(left_keys);
    potluck::scrub_curve_keypair(right_keys);
    potluck::scrub_curve_keypair(unauthorized_client.keypair);
    unauthorized_client.server_public_key.clear();
    potluck::scrub_curve_keypair(wrong_server.keypair);
    wrong_server.server_public_key.clear();
    potluck::scrub_curve_keypair(bad_endpoint.keypair);
    bad_endpoint.server_public_key.clear();
    potluck::scrub_curve_keypair(timeout_keys);
    potluck::scrub_curve_keypair(wrong_keys);
    return 0;
}

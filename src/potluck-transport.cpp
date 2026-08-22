#include "potluck-transport.h"

#include <zmq.h>

#include <cerrno>
#include <cstring>
#include <array>
#include <utility>
#include <vector>

namespace potluck {
namespace {

constexpr size_t metadata_fixed_bytes = 1 + 2 + 4 + 4 + 8 + 2 + 2 + 8;

void close_handles(void * & context, void * & socket) noexcept {
    if (socket != nullptr) {
        int linger = 0;
        zmq_setsockopt(socket, ZMQ_LINGER, &linger, sizeof(linger));
        zmq_close(socket);
        socket = nullptr;
    }
    if (context != nullptr) {
        zmq_ctx_term(context);
        context = nullptr;
    }
}

std::string zmq_error(const char * operation) {
    const int code = zmq_errno();
    std::string result(operation);
    result += ": ";
    if (code == EAGAIN) {
        result += "timeout";
    } else {
        result += zmq_strerror(code);
    }
    return result;
}

bool create_socket(int type, void * & context, void * & socket, std::string & error) {
    context = zmq_ctx_new();
    if (context == nullptr) {
        error = zmq_error("create ZeroMQ context");
        return false;
    }

    socket = zmq_socket(context, type);
    if (socket == nullptr) {
        error = zmq_error("create ZeroMQ socket");
        close_handles(context, socket);
        return false;
    }

    int linger = 0;
    if (zmq_setsockopt(socket, ZMQ_LINGER, &linger, sizeof(linger)) != 0) {
        error = zmq_error("configure ZeroMQ linger");
        close_handles(context, socket);
        return false;
    }

    const int64_t max_message_size = static_cast<int64_t>(max_payload_bytes);
    if (zmq_setsockopt(socket, ZMQ_MAXMSGSIZE, &max_message_size, sizeof(max_message_size)) != 0) {
        error = zmq_error("configure ZeroMQ message limit");
        close_handles(context, socket);
        return false;
    }
    return true;
}

bool set_timeout(void * socket, int option, int timeout_ms, const char * name, std::string & error) {
    if (socket == nullptr) {
        error = std::string(name) + ": socket is not initialized";
        return false;
    }
    if (timeout_ms < -1) {
        error = std::string(name) + ": timeout must be -1 or non-negative milliseconds";
        return false;
    }
    if (zmq_setsockopt(socket, option, &timeout_ms, sizeof(timeout_ms)) != 0) {
        error = zmq_error(name);
        return false;
    }
    return true;
}

bool read_u16(const uint8_t * data, size_t size, size_t & offset, uint16_t & value) {
    if (offset > size || size - offset < 2) {
        return false;
    }
    value = static_cast<uint16_t>(data[offset]) |
            static_cast<uint16_t>(data[offset + 1]) << 8;
    offset += 2;
    return true;
}

bool read_u32(const uint8_t * data, size_t size, size_t & offset, uint32_t & value) {
    if (offset > size || size - offset < 4) {
        return false;
    }
    value = static_cast<uint32_t>(data[offset]) |
            static_cast<uint32_t>(data[offset + 1]) << 8 |
            static_cast<uint32_t>(data[offset + 2]) << 16 |
            static_cast<uint32_t>(data[offset + 3]) << 24;
    offset += 4;
    return true;
}

bool read_u64(const uint8_t * data, size_t size, size_t & offset, uint64_t & value) {
    if (offset > size || size - offset < 8) {
        return false;
    }
    value = 0;
    for (size_t i = 0; i < 8; ++i) {
        value |= static_cast<uint64_t>(data[offset + i]) << (8 * i);
    }
    offset += 8;
    return true;
}

void append_u16(std::vector<uint8_t> & output, uint16_t value) {
    output.push_back(static_cast<uint8_t>(value));
    output.push_back(static_cast<uint8_t>(value >> 8));
}

void append_u32(std::vector<uint8_t> & output, uint32_t value) {
    for (size_t i = 0; i < 4; ++i) {
        output.push_back(static_cast<uint8_t>(value >> (8 * i)));
    }
}

void append_u64(std::vector<uint8_t> & output, uint64_t value) {
    for (size_t i = 0; i < 8; ++i) {
        output.push_back(static_cast<uint8_t>(value >> (8 * i)));
    }
}

bool valid_message_type(uint16_t type) {
    switch (static_cast<message_type>(type)) {
        case message_type::hello:
        case message_type::batch_meta:
        case message_type::hidden_state:
        case message_type::token:
        case message_type::reset:
        case message_type::node_config:
        case message_type::ready:
        case message_type::profile_result:
        case message_type::batch_decode:
        case message_type::batch_result:
        case message_type::slot_config:
        case message_type::heartbeat:
        case message_type::heartbeat_ack:
        case message_type::error:
            return true;
    }
    return false;
}

bool valid_data_type(uint16_t type) {
    switch (static_cast<data_type>(type)) {
        case data_type::none:
        case data_type::f32:
        case data_type::f16:
        case data_type::i32:
            return true;
    }
    return false;
}

bool encode_metadata(const message & input, std::vector<uint8_t> & metadata, std::string & error) {
    const uint16_t type = static_cast<uint16_t>(input.type);
    const uint16_t dtype = static_cast<uint16_t>(input.dtype);
    if (!valid_message_type(type)) {
        error = "send: unsupported message type " + std::to_string(type);
        return false;
    }
    if (!valid_data_type(dtype)) {
        error = "send: unsupported data type " + std::to_string(dtype);
        return false;
    }
    if (input.shape.size() > ring_max_shape_dims) {
        error = "send: shape has too many dimensions: " + std::to_string(input.shape.size());
        return false;
    }
    if (input.payload.size() > max_payload_bytes) {
        error = "send: payload exceeds limit: " + std::to_string(input.payload.size());
        return false;
    }

    const size_t shape_bytes = input.shape.size() * sizeof(uint64_t);
    if (metadata_fixed_bytes > ring_max_metadata_bytes ||
        shape_bytes > ring_max_metadata_bytes - metadata_fixed_bytes) {
        error = "send: message metadata exceeds limit";
        return false;
    }

    metadata.clear();
    metadata.reserve(metadata_fixed_bytes + shape_bytes);
    metadata.push_back(ring_message_version);
    append_u16(metadata, type);
    append_u32(metadata, input.flags);
    append_u32(metadata, input.rank);
    append_u64(metadata, input.sequence);
    append_u16(metadata, dtype);
    append_u16(metadata, static_cast<uint16_t>(input.shape.size()));
    append_u64(metadata, static_cast<uint64_t>(input.payload.size()));
    for (const uint64_t dimension : input.shape) {
        append_u64(metadata, dimension);
    }
    return true;
}

bool decode_metadata(const uint8_t * data, size_t size, size_t payload_size,
                     message & output, std::string & error) {
    if (size < metadata_fixed_bytes) {
        error = "receive: metadata frame is truncated";
        return false;
    }
    if (size > ring_max_metadata_bytes) {
        error = "receive: metadata frame exceeds limit: " + std::to_string(size);
        return false;
    }

    size_t offset = 0;
    const uint8_t version = data[offset++];
    if (version != ring_message_version) {
        error = "receive: unsupported metadata version " + std::to_string(version);
        return false;
    }

    uint16_t type = 0;
    uint32_t flags = 0;
    uint32_t rank = 0;
    uint64_t sequence = 0;
    uint16_t dtype = 0;
    uint16_t dimensions = 0;
    uint64_t declared_payload_size = 0;
    if (!read_u16(data, size, offset, type) ||
        !read_u32(data, size, offset, flags) ||
        !read_u32(data, size, offset, rank) ||
        !read_u64(data, size, offset, sequence) ||
        !read_u16(data, size, offset, dtype) ||
        !read_u16(data, size, offset, dimensions) ||
        !read_u64(data, size, offset, declared_payload_size)) {
        error = "receive: metadata frame is truncated";
        return false;
    }
    if (!valid_message_type(type)) {
        error = "receive: unsupported message type " + std::to_string(type);
        return false;
    }
    if (!valid_data_type(dtype)) {
        error = "receive: unsupported data type " + std::to_string(dtype);
        return false;
    }
    if (dimensions > ring_max_shape_dims) {
        error = "receive: metadata shape has too many dimensions: " +
                std::to_string(dimensions);
        return false;
    }
    if (declared_payload_size > max_payload_bytes) {
        error = "receive: declared payload exceeds limit: " +
                std::to_string(declared_payload_size);
        return false;
    }
    if (declared_payload_size != payload_size) {
        error = "receive: payload size mismatch: metadata=" +
                std::to_string(declared_payload_size) + ", frame=" +
                std::to_string(payload_size);
        return false;
    }

    const size_t shape_bytes = static_cast<size_t>(dimensions) * sizeof(uint64_t);
    if (shape_bytes > ring_max_metadata_bytes - metadata_fixed_bytes ||
        size != metadata_fixed_bytes + shape_bytes) {
        error = "receive: metadata shape length is invalid";
        return false;
    }

    message decoded;
    decoded.type = static_cast<message_type>(type);
    decoded.flags = flags;
    decoded.rank = rank;
    decoded.sequence = sequence;
    decoded.dtype = static_cast<data_type>(dtype);
    decoded.shape.reserve(dimensions);
    for (uint16_t i = 0; i < dimensions; ++i) {
        uint64_t dimension = 0;
        if (!read_u64(data, size, offset, dimension)) {
            error = "receive: metadata frame is truncated";
            return false;
        }
        decoded.shape.push_back(dimension);
    }
    if (offset != size) {
        error = "receive: metadata frame has trailing bytes";
        return false;
    }
    output = std::move(decoded);
    return true;
}

bool send_frame(void * socket, const void * data, size_t size, int flags,
                const char * name, std::string & error) {
    const int sent = zmq_send(socket, data, size, flags);
    if (sent < 0) {
        error = zmq_error(name);
        return false;
    }
    if (static_cast<size_t>(sent) != size) {
        error = std::string(name) + ": sent an incomplete frame";
        return false;
    }
    return true;
}

bool discard_remaining_frames(void * socket, std::string & error) {
    while (true) {
        zmq_msg_t extra;
        zmq_msg_init(&extra);
        if (zmq_msg_recv(&extra, socket, 0) < 0) {
            error = zmq_error("receive extra frame");
            zmq_msg_close(&extra);
            return false;
        }
        const bool more = zmq_msg_more(&extra) != 0;
        zmq_msg_close(&extra);
        if (!more) {
            return true;
        }
    }
}

std::string bound_endpoint(void * socket, std::string & error) {
    std::array<char, 256> endpoint = {};
    size_t size = endpoint.size();
    if (zmq_getsockopt(socket, ZMQ_LAST_ENDPOINT, endpoint.data(), &size) != 0) {
        error = zmq_error("read bound ZeroMQ endpoint");
        return {};
    }
    if (size > 0 && endpoint[size - 1] == '\0') {
        --size;
    }
    if (size == 0) {
        error = "bind: ZeroMQ returned an empty endpoint";
        return {};
    }
    return std::string(endpoint.data(), size);
}

} // namespace

ring_receiver::ring_receiver(void * context, void * socket, std::string endpoint)
    : context_(context), socket_(socket), endpoint_(std::move(endpoint)) {}

ring_receiver::~ring_receiver() {
    close_handles(context_, socket_);
}

ring_receiver::ring_receiver(ring_receiver && other) noexcept
    : context_(other.context_), socket_(other.socket_), endpoint_(std::move(other.endpoint_)) {
    other.context_ = nullptr;
    other.socket_ = nullptr;
    other.endpoint_.clear();
}

ring_receiver & ring_receiver::operator=(ring_receiver && other) noexcept {
    if (this != &other) {
        close_handles(context_, socket_);
        context_ = other.context_;
        socket_ = other.socket_;
        endpoint_ = std::move(other.endpoint_);
        other.context_ = nullptr;
        other.socket_ = nullptr;
        other.endpoint_.clear();
    }
    return *this;
}

ring_receiver ring_receiver::bind(const std::string & endpoint, std::string & error) {
    error.clear();
    if (endpoint.empty()) {
        error = "bind: endpoint is empty";
        return {};
    }

    void * context = nullptr;
    void * socket = nullptr;
    if (!create_socket(ZMQ_PULL, context, socket, error)) {
        return {};
    }
    if (zmq_bind(socket, endpoint.c_str()) != 0) {
        error = zmq_error((std::string("bind ZeroMQ PULL socket to ") + endpoint).c_str());
        close_handles(context, socket);
        return {};
    }

    std::string actual_endpoint = bound_endpoint(socket, error);
    if (actual_endpoint.empty()) {
        close_handles(context, socket);
        return {};
    }
    return ring_receiver(context, socket, std::move(actual_endpoint));
}

bool ring_receiver::valid() const noexcept {
    return context_ != nullptr && socket_ != nullptr;
}

const std::string & ring_receiver::endpoint() const noexcept {
    return endpoint_;
}

bool ring_receiver::set_receive_timeout(int timeout_ms, std::string & error) {
    error.clear();
    return set_timeout(socket_, ZMQ_RCVTIMEO, timeout_ms, "set receive timeout", error);
}

bool ring_receiver::receive(message & output, std::string & error) {
    error.clear();
    if (!valid()) {
        error = "receive: PULL socket is not initialized";
        return false;
    }

    zmq_msg_t metadata;
    zmq_msg_init(&metadata);
    if (zmq_msg_recv(&metadata, socket_, 0) < 0) {
        error = zmq_error("receive metadata frame");
        zmq_msg_close(&metadata);
        return false;
    }
    if (zmq_msg_more(&metadata) == 0) {
        error = "receive: expected metadata and payload frames";
        zmq_msg_close(&metadata);
        return false;
    }

    zmq_msg_t payload;
    zmq_msg_init(&payload);
    if (zmq_msg_recv(&payload, socket_, 0) < 0) {
        error = zmq_error("receive payload frame");
        zmq_msg_close(&payload);
        zmq_msg_close(&metadata);
        return false;
    }
    const bool extra_frames = zmq_msg_more(&payload) != 0;
    if (extra_frames) {
        std::string discard_error;
        const bool discarded = discard_remaining_frames(socket_, discard_error);
        error = discarded ? "receive: expected exactly two frames" : discard_error;
        zmq_msg_close(&payload);
        zmq_msg_close(&metadata);
        return false;
    }

    const size_t metadata_size = zmq_msg_size(&metadata);
    const size_t payload_size = zmq_msg_size(&payload);
    message decoded;
    if (!decode_metadata(static_cast<const uint8_t *>(zmq_msg_data(&metadata)), metadata_size,
                         payload_size, decoded, error)) {
        zmq_msg_close(&payload);
        zmq_msg_close(&metadata);
        return false;
    }
    if (payload_size != 0) {
        const uint8_t * payload_data = static_cast<const uint8_t *>(zmq_msg_data(&payload));
        decoded.payload.assign(payload_data, payload_data + payload_size);
    }
    output = std::move(decoded);
    zmq_msg_close(&payload);
    zmq_msg_close(&metadata);
    return true;
}

ring_sender::ring_sender(void * context, void * socket, std::string endpoint)
    : context_(context), socket_(socket), endpoint_(std::move(endpoint)) {}

ring_sender::~ring_sender() {
    close_handles(context_, socket_);
}

ring_sender::ring_sender(ring_sender && other) noexcept
    : context_(other.context_), socket_(other.socket_), endpoint_(std::move(other.endpoint_)) {
    other.context_ = nullptr;
    other.socket_ = nullptr;
    other.endpoint_.clear();
}

ring_sender & ring_sender::operator=(ring_sender && other) noexcept {
    if (this != &other) {
        close_handles(context_, socket_);
        context_ = other.context_;
        socket_ = other.socket_;
        endpoint_ = std::move(other.endpoint_);
        other.context_ = nullptr;
        other.socket_ = nullptr;
        other.endpoint_.clear();
    }
    return *this;
}

ring_sender ring_sender::connect(const std::string & endpoint, std::string & error) {
    error.clear();
    if (endpoint.empty()) {
        error = "connect: endpoint is empty";
        return {};
    }

    void * context = nullptr;
    void * socket = nullptr;
    if (!create_socket(ZMQ_PUSH, context, socket, error)) {
        return {};
    }
    const int immediate = 1;
    if (zmq_setsockopt(socket, ZMQ_IMMEDIATE, &immediate, sizeof(immediate)) != 0) {
        error = zmq_error("configure ZeroMQ immediate connect");
        close_handles(context, socket);
        return {};
    }
    if (zmq_connect(socket, endpoint.c_str()) != 0) {
        error = zmq_error((std::string("connect ZeroMQ PUSH socket to ") + endpoint).c_str());
        close_handles(context, socket);
        return {};
    }
    return ring_sender(context, socket, endpoint);
}

bool ring_sender::valid() const noexcept {
    return context_ != nullptr && socket_ != nullptr;
}

const std::string & ring_sender::endpoint() const noexcept {
    return endpoint_;
}

bool ring_sender::set_send_timeout(int timeout_ms, std::string & error) {
    error.clear();
    return set_timeout(socket_, ZMQ_SNDTIMEO, timeout_ms, "set send timeout", error);
}

bool ring_sender::send(const message & input, std::string & error) {
    error.clear();
    if (!valid()) {
        error = "send: PUSH socket is not initialized";
        return false;
    }

    std::vector<uint8_t> metadata;
    if (!encode_metadata(input, metadata, error)) {
        return false;
    }
    if (!send_frame(socket_, metadata.data(), metadata.size(), ZMQ_SNDMORE,
                    "send metadata frame", error)) {
        return false;
    }
    const void * payload = input.payload.empty() ? nullptr : input.payload.data();
    return send_frame(socket_, payload, input.payload.size(), 0, "send payload frame", error);
}
ring_peer ring_peer::bind(const std::string & local_endpoint, std::string & error) {
    error.clear();
    ring_peer result;
    result.receiver_ = ring_receiver::bind(local_endpoint, error);
    if (!result.receiver_.valid()) {
        return {};
    }
    return result;
}

ring_peer ring_peer::bind(const std::string & local_endpoint,
                          const std::string & next_endpoint,
                          std::string & error) {
    ring_peer result = bind(local_endpoint, error);
    if (!result.receiver_.valid()) {
        return {};
    }
    if (!result.connect_next(next_endpoint, error)) {
        return {};
    }
    return result;
}

bool ring_peer::connect_next(const std::string & next_endpoint, std::string & error) {
    error.clear();
    ring_sender sender = ring_sender::connect(next_endpoint, error);
    if (!sender.valid()) {
        return false;
    }
    sender_ = std::move(sender);
    next_endpoint_ = next_endpoint;
    return true;
}

bool ring_peer::valid() const noexcept {
    return receive_valid() && send_valid();
}

bool ring_peer::receive_valid() const noexcept {
    return receiver_.valid();
}

bool ring_peer::send_valid() const noexcept {
    return sender_.valid();
}

const std::string & ring_peer::endpoint() const noexcept {
    return receiver_.endpoint();
}

const std::string & ring_peer::next_endpoint() const noexcept {
    return next_endpoint_;
}

bool ring_peer::set_timeouts(int receive_timeout_ms, int send_timeout_ms, std::string & error) {
    error.clear();
    if (!receiver_.set_receive_timeout(receive_timeout_ms, error)) {
        return false;
    }
    if (!sender_.set_send_timeout(send_timeout_ms, error)) {
        return false;
    }
    return true;
}

bool ring_peer::receive(message & output, std::string & error) {
    return receiver_.receive(output, error);
}

bool ring_peer::send(const message & input, std::string & error) {
    return sender_.send(input, error);
}

ring_receiver & ring_peer::receiver() noexcept {
    return receiver_;
}

const ring_receiver & ring_peer::receiver() const noexcept {
    return receiver_;
}

ring_sender & ring_peer::sender() noexcept {
    return sender_;
}

const ring_sender & ring_peer::sender() const noexcept {
    return sender_;
}

} // namespace potluck

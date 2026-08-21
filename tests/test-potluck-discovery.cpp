#include "../src/potluck-discovery.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>
#include <atomic>

#define CHECK(cond)                                                                       \
    do {                                                                                  \
        if (!(cond)) {                                                                    \
            std::fprintf(stderr, "CHECK failed: %s (%s:%d)\n", #cond, __FILE__, __LINE__); \
            std::abort();                                                                 \
        }                                                                                 \
    } while (0)

namespace {

bool daemon_unavailable(const std::string & error) {
    return error.find("-65563") != std::string::npos;
}

bool lower_hex_id(const std::string & id) {
    if (id.size() != 32) {
        return false;
    }
    for (const unsigned char byte : id) {
        if (!((byte >= '0' && byte <= '9') || (byte >= 'a' && byte <= 'f'))) {
            return false;
        }
    }
    return true;
}

} // namespace

int main() {
    const std::string id = "0123456789abcdef0123456789abcdef";
    const std::string instance = "potluck-test-" + std::to_string(static_cast<unsigned long long>(getpid()));

    std::mutex mutex;
    std::condition_variable condition;
    std::atomic<bool> stop{false};
    bool started = false;
    bool advertisement_valid = false;
    std::string advertisement_error;
    std::string worker_error;
    potluck::node_advertisement advertisement;

    std::thread publisher([&] {
        std::string error;
        potluck::node_advertisement local = potluck::node_advertisement::start(
            id, instance, "potluck-test", 22, 40001, error);
        {
            std::lock_guard<std::mutex> lock(mutex);
            advertisement = std::move(local);
            advertisement_error = std::move(error);
            advertisement_valid = advertisement.valid();
            started = true;
        }
        condition.notify_one();

        while (!stop.load(std::memory_order_relaxed) && advertisement.valid()) {
            std::string process_error;
            if (!advertisement.process(100, process_error)) {
                std::lock_guard<std::mutex> lock(mutex);
                worker_error = std::move(process_error);
                break;
            }
        }
    });

    {
        std::unique_lock<std::mutex> lock(mutex);
        if (!condition.wait_for(lock, std::chrono::seconds(2), [&] { return started; })) {
            stop.store(true, std::memory_order_relaxed);
            lock.unlock();
            publisher.join();
            CHECK(false && "DNS-SD advertisement did not initialize");
        }
    }

    if (!advertisement_valid) {
        const bool skip = daemon_unavailable(advertisement_error);
        stop.store(true, std::memory_order_relaxed);
        publisher.join();
        if (skip) {
            std::fprintf(stderr, "SKIP: DNS-SD daemon unavailable: %s\n", advertisement_error.c_str());
            return 0;
        }
        std::fprintf(stderr, "DNS-SD advertisement failed: %s\n", advertisement_error.c_str());
        return 1;
    }

    std::string discovery_error;
    std::vector<potluck::discovered_node> nodes = potluck::discover_nodes(3000, discovery_error);
    stop.store(true, std::memory_order_relaxed);
    publisher.join();

    if (!discovery_error.empty() && daemon_unavailable(discovery_error)) {
        std::fprintf(stderr, "SKIP: DNS-SD daemon unavailable: %s\n", discovery_error.c_str());
        return 0;
    }
    CHECK(discovery_error.empty());
    CHECK(worker_error.empty());

    const auto found = std::find_if(nodes.begin(), nodes.end(), [&](const potluck::discovered_node & node) {
        return node.id == id;
    });
    CHECK(found != nodes.end());
    CHECK(lower_hex_id(found->id));
    CHECK(found->instance == instance || found->instance.find(instance + "-") == 0);
    CHECK(!found->host.empty());
    CHECK(found->ssh_user == "potluck-test");
    CHECK(found->ssh_port == 22);
    CHECK(found->ring_port == 40001);
    CHECK(found->available);
    CHECK(found->protocol_version == potluck::discovery_protocol_version);

    std::string invalid_error;
    const potluck::node_advertisement invalid = potluck::node_advertisement::start(
        "", instance, "potluck-test", 22, 40001, invalid_error);
    CHECK(!invalid.valid());
    CHECK(!invalid_error.empty());
    return 0;
}

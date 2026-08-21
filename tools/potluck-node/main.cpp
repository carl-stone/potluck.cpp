#include "potluck-discovery.h"

#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <pwd.h>
#include <signal.h>
#include <string>
#include <system_error>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>

namespace {

namespace fs = std::filesystem;

constexpr char program_name[] = "potluck-node";
constexpr uint16_t ssh_port = 22;
constexpr uint16_t ring_port = 40001;
constexpr size_t node_id_bytes = 16;
constexpr size_t node_id_text_size = node_id_bytes * 2;
constexpr size_t password_buffer_limit = 1u << 20;

volatile sig_atomic_t stop_requested = 0;

void handle_stop_signal(int) noexcept {
    stop_requested = 1;
}

int open_flags(int flags) {
#ifdef O_CLOEXEC
    flags |= O_CLOEXEC;
#endif
#ifdef O_NOFOLLOW
    flags |= O_NOFOLLOW;
#endif
    return flags;
}

void set_errno_error(std::string & error, const char * operation, const std::string & target) {
    const int saved_errno = errno;
    error = operation;
    if (!target.empty()) {
        error += " '";
        error += target;
        error += "'";
    }
    error += ": ";
    error += std::strerror(saved_errno);
}

bool is_node_id(const std::string & id) {
    if (id.size() != node_id_text_size) {
        return false;
    }
    for (const char value : id) {
        const bool digit = value >= '0' && value <= '9';
        const bool letter = value >= 'a' && value <= 'f';
        if (!digit && !letter) {
            return false;
        }
    }
    return true;
}

bool config_node_id_path(fs::path & path, std::string & error) {
    const char * config_home = std::getenv("XDG_CONFIG_HOME");
    fs::path config_root;
    if (config_home != nullptr && config_home[0] != '\0') {
        config_root = fs::path(config_home);
        if (!config_root.is_absolute()) {
            error = "XDG_CONFIG_HOME must be an absolute path";
            return false;
        }
    } else {
        const char * home = std::getenv("HOME");
        if (home == nullptr || home[0] == '\0') {
            error = "HOME is not set and XDG_CONFIG_HOME is unavailable";
            return false;
        }
        config_root = fs::path(home) / ".config";
        if (!config_root.is_absolute()) {
            error = "HOME must be an absolute path";
            return false;
        }
    }

    path = config_root / "potluck" / "node-id";
    return true;
}

bool ensure_private_directory(const fs::path & directory, std::string & error) {
    std::error_code status_error;
    fs::file_status status = fs::symlink_status(directory, status_error);
    if (status_error && status_error != std::errc::no_such_file_or_directory) {
        error = "cannot inspect config directory '" + directory.string() + "': " + status_error.message();
        return false;
    }

    if (!status_error && status.type() != fs::file_type::not_found) {
        if (fs::is_symlink(status) || !fs::is_directory(status)) {
            error = "config path is not a directory: '" + directory.string() + "'";
            return false;
        }
    } else {
        std::error_code create_error;
        fs::create_directories(directory, create_error);
        if (create_error) {
            error = "cannot create config directory '" + directory.string() + "': " + create_error.message();
            return false;
        }
        status_error.clear();
        status = fs::symlink_status(directory, status_error);
        if (status_error || fs::is_symlink(status) || !fs::is_directory(status)) {
            error = "config path is not a directory: '" + directory.string() + "'";
            return false;
        }
    }

    if (::chmod(directory.c_str(), 0700) != 0) {
        set_errno_error(error, "cannot set permissions on", directory.string());
        return false;
    }
    return true;
}

bool load_node_id(const fs::path & path, std::string & id, bool & exists, std::string & error) {
    exists = false;
    int fd = ::open(path.c_str(), open_flags(O_RDONLY));
    if (fd < 0) {
        if (errno == ENOENT) {
            return true;
        }
        set_errno_error(error, "cannot open node id", path.string());
        return false;
    }
    exists = true;

    struct stat file_status = {};
    if (::fstat(fd, &file_status) != 0) {
        set_errno_error(error, "cannot inspect node id", path.string());
        ::close(fd);
        return false;
    }
    if (!S_ISREG(file_status.st_mode)) {
        error = "node id is not a regular file: '" + path.string() + "'";
        ::close(fd);
        return false;
    }

    char text[node_id_text_size];
    size_t offset = 0;
    while (offset < sizeof(text)) {
        const ssize_t count = ::read(fd, text + offset, sizeof(text) - offset);
        if (count > 0) {
            offset += static_cast<size_t>(count);
            continue;
        }
        if (count < 0 && errno == EINTR) {
            continue;
        }
        if (count < 0) {
            set_errno_error(error, "cannot read node id", path.string());
        } else {
            error = "node id must contain exactly 32 lowercase hexadecimal characters: '" + path.string() + "'";
        }
        ::close(fd);
        return false;
    }

    char extra = '\0';
    ssize_t extra_count;
    do {
        extra_count = ::read(fd, &extra, sizeof(extra));
    } while (extra_count < 0 && errno == EINTR);
    if (extra_count < 0) {
        set_errno_error(error, "cannot validate node id", path.string());
        ::close(fd);
        return false;
    }
    if (extra_count != 0) {
        error = "node id must contain exactly 32 lowercase hexadecimal characters: '" + path.string() + "'";
        ::close(fd);
        return false;
    }

    id.assign(text, sizeof(text));
    if (!is_node_id(id)) {
        error = "node id must contain exactly 32 lowercase hexadecimal characters: '" + path.string() + "'";
        ::close(fd);
        return false;
    }

    if ((file_status.st_mode & 07777) != 0600 && ::fchmod(fd, 0600) != 0) {
        set_errno_error(error, "cannot set node id permissions", path.string());
        ::close(fd);
        return false;
    }
    if (::close(fd) != 0) {
        set_errno_error(error, "cannot close node id", path.string());
        return false;
    }
    return true;
}

bool random_node_id(std::string & id, std::string & error) {
    int fd = ::open("/dev/urandom", open_flags(O_RDONLY));
    if (fd < 0) {
        set_errno_error(error, "cannot open random source", "/dev/urandom");
        return false;
    }

    std::array<unsigned char, node_id_bytes> bytes = {};
    size_t offset = 0;
    while (offset < bytes.size()) {
        const ssize_t count = ::read(fd, bytes.data() + offset, bytes.size() - offset);
        if (count > 0) {
            offset += static_cast<size_t>(count);
            continue;
        }
        if (count < 0 && errno == EINTR) {
            continue;
        }
        if (count < 0) {
            set_errno_error(error, "cannot read random source", "/dev/urandom");
        } else {
            error = "random source ended before generating a node id";
        }
        ::close(fd);
        return false;
    }
    if (::close(fd) != 0) {
        set_errno_error(error, "cannot close random source", "/dev/urandom");
        return false;
    }

    static constexpr char hex[] = "0123456789abcdef";
    id.assign(node_id_text_size, '0');
    for (size_t i = 0; i < bytes.size(); ++i) {
        id[i * 2] = hex[bytes[i] >> 4];
        id[i * 2 + 1] = hex[bytes[i] & 0x0f];
    }
    return true;
}

bool atomic_write_node_id(const fs::path & path, const std::string & id, std::string & error) {
    const std::string suffix = std::to_string(static_cast<long long>(::getpid()));
    for (unsigned int attempt = 0; attempt < 16; ++attempt) {
        fs::path temporary = path;
        temporary += ".tmp." + suffix + "." + std::to_string(attempt);
        int fd = ::open(temporary.c_str(), open_flags(O_WRONLY | O_CREAT | O_EXCL), 0600);
        if (fd < 0) {
            if (errno == EEXIST) {
                continue;
            }
            set_errno_error(error, "cannot create node id temporary file", temporary.string());
            return false;
        }

        size_t offset = 0;
        bool failed = false;
        while (offset < id.size()) {
            const ssize_t count = ::write(fd, id.data() + offset, id.size() - offset);
            if (count > 0) {
                offset += static_cast<size_t>(count);
                continue;
            }
            if (count < 0 && errno == EINTR) {
                continue;
            }
            if (count < 0) {
                set_errno_error(error, "cannot write node id", temporary.string());
            } else {
                error = "cannot write node id: write returned zero bytes";
            }
            failed = true;
            break;
        }
        if (!failed && ::fchmod(fd, 0600) != 0) {
            set_errno_error(error, "cannot set node id permissions", temporary.string());
            failed = true;
        }
        if (!failed && ::fsync(fd) != 0) {
            set_errno_error(error, "cannot flush node id", temporary.string());
            failed = true;
        }
        if (::close(fd) != 0 && !failed) {
            set_errno_error(error, "cannot close node id", temporary.string());
            failed = true;
        }
        if (failed) {
            ::unlink(temporary.c_str());
            return false;
        }
        if (::rename(temporary.c_str(), path.c_str()) != 0) {
            set_errno_error(error, "cannot install node id", path.string());
            ::unlink(temporary.c_str());
            return false;
        }
        return true;
    }

    error = "cannot create a unique node id temporary file beside '" + path.string() + "'";
    return false;
}

bool load_or_create_node_id(std::string & id, std::string & error) {
    fs::path path;
    if (!config_node_id_path(path, error)) {
        return false;
    }
    if (!ensure_private_directory(path.parent_path(), error)) {
        return false;
    }

    bool exists = false;
    if (!load_node_id(path, id, exists, error)) {
        return false;
    }
    if (exists) {
        return true;
    }
    if (!random_node_id(id, error)) {
        return false;
    }
    return atomic_write_node_id(path, id, error);
}

bool current_username(std::string & username, std::string & error) {
    long configured_size = ::sysconf(_SC_GETPW_R_SIZE_MAX);
    if (configured_size <= 0 || static_cast<unsigned long>(configured_size) > password_buffer_limit) {
        configured_size = 16384;
    }
    std::vector<char> buffer(static_cast<size_t>(configured_size));

    for (;;) {
        struct passwd entry = {};
        struct passwd * result = nullptr;
        const int status = ::getpwuid_r(::getuid(), &entry, buffer.data(), buffer.size(), &result);
        if (status == 0 && result != nullptr && result->pw_name != nullptr && result->pw_name[0] != '\0') {
            username = result->pw_name;
            return true;
        }
        if (status == ERANGE && buffer.size() < password_buffer_limit) {
            buffer.resize(buffer.size() * 2);
            continue;
        }
        if (status != 0) {
            error = "cannot determine current username: ";
            error += std::strerror(status);
        } else {
            error = "cannot determine current username from passwd database";
        }
        return false;
    }
}

bool current_hostname(std::string & hostname, std::string & error) {
    std::array<char, 256> buffer = {};
    if (::gethostname(buffer.data(), buffer.size()) != 0) {
        set_errno_error(error, "cannot determine hostname", "");
        return false;
    }
    if (buffer.back() != '\0') {
        error = "cannot determine hostname: name is too long";
        return false;
    }
    if (buffer[0] == '\0') {
        error = "cannot determine hostname: gethostname returned an empty name";
        return false;
    }
    hostname = buffer.data();
    return true;
}

bool install_signal_handlers(std::string & error) {
    struct sigaction action = {};
    action.sa_handler = handle_stop_signal;
    if (sigemptyset(&action.sa_mask) != 0) {
        set_errno_error(error, "cannot initialize signal handling", "");
        return false;
    }
    if (::sigaction(SIGINT, &action, nullptr) != 0) {
        set_errno_error(error, "cannot install SIGINT handler", "");
        return false;
    }
    if (::sigaction(SIGTERM, &action, nullptr) != 0) {
        set_errno_error(error, "cannot install SIGTERM handler", "");
        return false;
    }
    return true;
}

void report_error(const std::string & error) {
    std::fprintf(stderr, "%s: %s\n", program_name, error.empty() ? "operation failed" : error.c_str());
}

} // namespace

int main(int argc, char **) {
    if (argc != 1) {
        std::fprintf(stderr, "%s: no arguments are accepted; run it without arguments\n", program_name);
        return EXIT_FAILURE;
    }

    std::string error;
    std::string id;
    if (!load_or_create_node_id(id, error)) {
        report_error(error);
        return EXIT_FAILURE;
    }

    std::string username;
    if (!current_username(username, error)) {
        report_error(error);
        return EXIT_FAILURE;
    }

    std::string hostname;
    if (!current_hostname(hostname, error)) {
        report_error(error);
        return EXIT_FAILURE;
    }

    if (!install_signal_handlers(error)) {
        report_error(error);
        return EXIT_FAILURE;
    }

    potluck::node_advertisement advertisement = potluck::node_advertisement::start(
        id, hostname, username, ssh_port, ring_port, error);
    if (!advertisement.valid()) {
        if (error.empty()) {
            error = "DNS-SD advertisement could not be started";
        }
        report_error(error);
        return EXIT_FAILURE;
    }

    std::printf("%s: advertising %s (ssh %u, ring %u)\n", program_name, hostname.c_str(),
                static_cast<unsigned int>(ssh_port), static_cast<unsigned int>(ring_port));
    std::fflush(stdout);
    while (!stop_requested) {
        error.clear();
        if (!advertisement.process(1000, error)) {
            if (stop_requested) {
                break;
            }
            if (error.empty()) {
                error = "DNS-SD processing failed";
            }
            report_error(error);
            return EXIT_FAILURE;
        }
    }

    std::printf("%s: stopped\n", program_name);
    return EXIT_SUCCESS;
}

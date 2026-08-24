#include "../tools/potluck-server/internal.h"

#include <csignal>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
 

#define CHECK(cond)                                                                       \
    do {                                                                                  \
        if (!(cond)) {                                                                    \
            std::fprintf(stderr, "CHECK failed: %s (%s:%d)\n", #cond, __FILE__, __LINE__); \
            std::abort();                                                                 \
        }                                                                                 \
    } while (0)

namespace {

void write_text(const std::filesystem::path & path, const std::string & value) {
    std::ofstream output(path);
    CHECK(output.good());
    output << value;
    CHECK(output.good());
}

std::string read_text(const std::filesystem::path & path) {
    std::ifstream input(path);
    CHECK(input.good());
    std::string value;
    char buffer[512] = {};
    while (input.read(buffer, sizeof(buffer)) || input.gcount() != 0) {
        value.append(buffer, static_cast<size_t>(input.gcount()));
    }
    return value;
}

void make_executable(const std::filesystem::path & path) {
    std::error_code error;
    std::filesystem::permissions(
        path,
        std::filesystem::perms::owner_read | std::filesystem::perms::owner_write |
            std::filesystem::perms::owner_exec,
        std::filesystem::perm_options::add, error);
    CHECK(!error);
}

std::string worker_script(const std::string & marker) {
    return "#!/usr/bin/env bash\n"
           "set -euo pipefail\n"
           "# marker " + marker + "\n"
           "case \"${1:-}\" in\n"
           "    --serve) while :; do sleep 1; done ;;\n"
           "    --probe)\n"
           "        case \"$0\" in\n"
           "            */.incoming/potluck-worker) test \"${FAKE_STAGED_PROBE:-ok}\" = ok ;;\n"
           "            *) test \"${FAKE_INSTALLED_PROBE:-ok}\" = ok ;;\n"
           "        esac ;;\n"
           "    *) exit 1 ;;\n"
           "esac\n";
}

struct refresh_fixture {
    std::filesystem::path root;
    std::filesystem::path fake_bin;
    std::filesystem::path remote_home;
    std::filesystem::path stage;
    std::filesystem::path rsync_log;
    std::string saved_path;
    bootstrap_node bootstrap;
    pid_t live_worker_pid = -1;

    refresh_fixture() {
        root = std::filesystem::temp_directory_path() /
            ("potluck-refresh-" + std::to_string(static_cast<long long>(getpid())));
        std::filesystem::remove_all(root);
        fake_bin = root / "fake-bin";
        remote_home = root / "remote-home";
        stage = root / "stage";
        rsync_log = root / "rsync.log";
        std::filesystem::create_directories(fake_bin);
        std::filesystem::create_directories(remote_home / "potluck");

        write_text(fake_bin / "ssh", R"SH(#!/usr/bin/env bash
set -euo pipefail
command="${!#}"
if [[ "${command}" == 'uname -sm 2>/dev/null' ]]; then
    printf '%s\n' "${FAKE_SSH_PLATFORM}"
else
    HOME="${FAKE_REMOTE_HOME}" bash -c "${command}"
fi
)SH");
        write_text(fake_bin / "rsync", R"SH(#!/usr/bin/env bash
set -euo pipefail
args=("$@")
last=$(( ${#args[@]} - 1 ))
source="${args[$((last - 1))]}"
destination="${args[last]}"
printf '%s\n' "${destination}" >> "${FAKE_RSYNC_LOG}"
if [[ "${FAKE_RSYNC_FAIL:-0}" == 1 ]]; then
    exit 17
fi
source="${source%/}"
remote_path="${destination#*:}"
remote_path="${remote_path%/}"
destination_dir="${FAKE_REMOTE_HOME}/${remote_path}"
mkdir -p "${destination_dir}"
cp -p "${source}"/* "${destination_dir}/"
if [[ "${FAKE_RSYNC_TAMPER:-0}" == 1 ]]; then
    printf 'tampered in transit\n' >> "${destination_dir}/potluck-worker"
fi
)SH");
        make_executable(fake_bin / "ssh");
        make_executable(fake_bin / "rsync");

        const char * path = std::getenv("PATH");
        saved_path = path == nullptr ? std::string() : path;
        const std::string test_path = fake_bin.string() + ":" + saved_path;
        CHECK(setenv("PATH", test_path.c_str(), 1) == 0);
        CHECK(setenv("FAKE_REMOTE_HOME", remote_home.string().c_str(), 1) == 0);
        CHECK(setenv("FAKE_RSYNC_LOG", rsync_log.string().c_str(), 1) == 0);
        bootstrap.ssh_target = "fake-target";
        bootstrap.ring_host = "fake-host";
    }

    ~refresh_fixture() {
        stop_live_worker();
        CHECK(setenv("PATH", saved_path.c_str(), 1) == 0);
        unsetenv("FAKE_REMOTE_HOME");
        unsetenv("FAKE_RSYNC_LOG");
        unsetenv("FAKE_SSH_PLATFORM");
        unsetenv("FAKE_RSYNC_FAIL");
        unsetenv("FAKE_RSYNC_TAMPER");
        unsetenv("FAKE_STAGED_PROBE");
        unsetenv("FAKE_INSTALLED_PROBE");
        std::filesystem::remove_all(root);
    }

    void stop_live_worker() {
        if (live_worker_pid <= 0) {
            return;
        }
        (void) kill(live_worker_pid, SIGTERM);
        int status = 0;
        while (waitpid(live_worker_pid, &status, 0) < 0 && errno == EINTR) {
        }
        live_worker_pid = -1;
    }

    void start_live_worker() {
        stop_live_worker();
        const std::string path = (remote_payload() / "potluck-worker").string();
        live_worker_pid = fork();
        CHECK(live_worker_pid >= 0);
        if (live_worker_pid == 0) {
            execl(path.c_str(), path.c_str(), "--serve", static_cast<char *>(nullptr));
            _exit(127);
        }
        usleep(10000);
        CHECK(kill(live_worker_pid, 0) == 0);
    }

    void check_live_worker() const {
        CHECK(live_worker_pid > 0);
        CHECK(kill(live_worker_pid, 0) == 0);
    }

    void configure(const std::string & platform = "local-platform",
                   bool rsync_fail = false, bool tamper = false,
                   bool staged_fail = false, bool installed_fail = false) {
        CHECK(setenv("FAKE_SSH_PLATFORM", platform.c_str(), 1) == 0);
        CHECK(setenv("FAKE_RSYNC_FAIL", rsync_fail ? "1" : "0", 1) == 0);
        CHECK(setenv("FAKE_RSYNC_TAMPER", tamper ? "1" : "0", 1) == 0);
        CHECK(setenv("FAKE_STAGED_PROBE", staged_fail ? "fail" : "ok", 1) == 0);
        CHECK(setenv("FAKE_INSTALLED_PROBE", installed_fail ? "fail" : "ok", 1) == 0);
    }

    void prepare(const std::string & remote_build_id, const std::string & remote_marker,
                 const std::string & stage_build_id, const std::string & stage_marker,
                 bool remote_present = true) {
        stop_live_worker();
        const std::filesystem::path remote_payload = remote_home / "potluck";
        std::filesystem::remove_all(remote_payload);
        std::filesystem::create_directories(remote_payload);
        if (remote_present) {
            write_text(remote_payload / "potluck-build-id", remote_build_id);
            write_text(remote_payload / "potluck-worker", worker_script(remote_marker));
            write_text(remote_payload / "potluck-node", remote_marker + " node\n");
            write_text(remote_payload / "libzmq.so.5", remote_marker + " library\n");
            make_executable(remote_payload / "potluck-worker");
            make_executable(remote_payload / "potluck-node");
        }

        std::filesystem::remove_all(stage);
        std::filesystem::create_directories(stage);
        write_text(stage / "potluck-build-id", stage_build_id);
        write_text(stage / "potluck-worker", worker_script(stage_marker));
        write_text(stage / "potluck-node", stage_marker + " node\n");
        write_text(stage / "libzmq.so.5", stage_marker + " library\n");
        make_executable(stage / "potluck-worker");
        make_executable(stage / "potluck-node");
        std::filesystem::remove(rsync_log);
        std::filesystem::remove_all(remote_payload / ".incoming");
        std::filesystem::remove_all(remote_payload / ".previous");
    }


    std::filesystem::path remote_payload() const {
        return remote_home / "potluck";
    }

    size_t rsync_calls() const {
        std::ifstream input(rsync_log);
        size_t count = 0;
        std::string line;
        while (std::getline(input, line)) {
            ++count;
        }
        return count;
    }

    void check_old_live() const {
        CHECK(read_text(remote_payload() / "potluck-build-id") ==
              "platform local-platform\ncommit old\n");
        CHECK(read_text(remote_payload() / "potluck-worker").find("# marker old") !=
              std::string::npos);
        CHECK(read_text(remote_payload() / "potluck-node") == "old node\n");
        CHECK(read_text(remote_payload() / "libzmq.so.5") == "old library\n");
    }

    void check_new_live() const {
        CHECK(read_text(remote_payload() / "potluck-build-id") ==
              "platform local-platform\ncommit new\n");
        CHECK(read_text(remote_payload() / "potluck-worker").find("# marker new") !=
              std::string::npos);
        CHECK(read_text(remote_payload() / "potluck-node") == "new node\n");
        CHECK(read_text(remote_payload() / "libzmq.so.5") == "new library\n");
    }

    void check_temporary_dirs_absent() const {
        CHECK(!std::filesystem::exists(remote_payload() / ".incoming"));
        CHECK(!std::filesystem::exists(remote_payload() / ".previous"));
    }
};

} // namespace

int main() {
    const std::string old_build_id = "platform local-platform\ncommit old\n";
    const std::string new_build_id = "platform local-platform\ncommit new\n";
    refresh_fixture fixture;

    fixture.prepare(old_build_id, "old", new_build_id, "new", false);
    fixture.configure();
    CHECK(refresh_remote_binaries(fixture.bootstrap, fixture.stage, "local-platform"));
    fixture.check_new_live();
    fixture.check_temporary_dirs_absent();

    fixture.prepare(old_build_id, "old", new_build_id, "new");
    fixture.start_live_worker();
    fixture.configure("local-platform", false, true);
    CHECK(!refresh_remote_binaries(fixture.bootstrap, fixture.stage, "local-platform"));
    fixture.check_old_live();
    fixture.check_live_worker();
    fixture.check_temporary_dirs_absent();

    fixture.prepare(old_build_id, "old", new_build_id, "new");
    fixture.start_live_worker();
    fixture.configure("local-platform", false, false, true);
    CHECK(!refresh_remote_binaries(fixture.bootstrap, fixture.stage, "local-platform"));
    fixture.check_old_live();
    fixture.check_live_worker();
    fixture.check_temporary_dirs_absent();

    fixture.prepare(old_build_id, "old", new_build_id, "new");
    fixture.start_live_worker();
    fixture.configure("local-platform", false, false, false, true);
    CHECK(!refresh_remote_binaries(fixture.bootstrap, fixture.stage, "local-platform"));
    fixture.check_old_live();
    fixture.check_live_worker();
    fixture.check_temporary_dirs_absent();

    fixture.prepare(new_build_id, "old", new_build_id, "new");
    fixture.configure("local-platform", true);
    CHECK(refresh_remote_binaries(fixture.bootstrap, fixture.stage, "local-platform"));
    CHECK(fixture.rsync_calls() == 0);
    CHECK(read_text(fixture.remote_payload() / "potluck-build-id") == new_build_id);
    CHECK(read_text(fixture.remote_payload() / "potluck-worker").find("# marker old") !=
          std::string::npos);
    CHECK(read_text(fixture.remote_payload() / "potluck-node") == "old node\n");
    CHECK(read_text(fixture.remote_payload() / "libzmq.so.5") == "old library\n");
    fixture.check_temporary_dirs_absent();

    fixture.prepare(old_build_id, "old", new_build_id, "new");
    fixture.start_live_worker();
    fixture.configure("local-platform", false, false, true);
    ring_session session;
    session.healthy = true;
    planned_worker current_worker;
    current_worker.kind = worker_kind::remote;
    current_worker.device.bootstrap = fixture.bootstrap;
    session.workers.push_back(current_worker);
    ring_startup_options options;
    options.has_staged_payload = true;
    options.stage_dir = fixture.stage;
    options.local_platform = "local-platform";
    std::string rebuild_error;
    CHECK(!rebuild_ring(session, options, false, rebuild_error));
    CHECK(rebuild_error.find("worker binary refresh failed") != std::string::npos);
    fixture.check_old_live();
    fixture.check_live_worker();
    fixture.check_temporary_dirs_absent();

    return 0;
}

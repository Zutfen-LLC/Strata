#pragma once
#include "local_fixture.hpp"
#include <fcntl.h>
#include <charconv>
#include <stdexcept>
namespace {
void pause_ms(int ms) { std::this_thread::sleep_for(std::chrono::milliseconds(ms)); }
struct Binary {
    pid_t pid = -1;
    uint16_t port = 0;
    fs::path log;
    Binary(const Fixture& f, int ttl = 30000, int deadline = 10000, const char* executable = STRATA_WORKER_BIN, bool residency_flag = true,
           bool residency_forever = false) {
        // Create a unique empty log before fork. A prior instance's listening line must never
        // be accepted while the new child is still opening/truncating its output.
        std::string pattern = (f.root / "worker-XXXXXX").string();
        std::vector<char> name(pattern.begin(), pattern.end());
        name.push_back(0);
        const int fd = ::mkstemp(name.data());
        require(fd >= 0, "create unique worker log");
        log = name.data();
        pid = ::fork();
        if (pid < 0) { ::close(fd); throw std::runtime_error("fork"); }
        if (pid == 0) {
            if (::dup2(fd, 2) < 0) _exit(126);
            ::close(fd);
            auto root = f.root.string(), manifest = (f.root / "local.manifest").string();
            auto lifetime = std::to_string(ttl), io = std::to_string(deadline);
            if (residency_forever) {
                ::execl(executable, executable, "--port", "0", "--workers", "2", "--memory-budget-mib", "20",
                        "--local-root", root.c_str(), "--local-manifest", manifest.c_str(),
                        "--residency-forever", "--deadline-ms", io.c_str(), nullptr);
                _exit(127);
            }
            if (!residency_flag) {
                ::execl(executable, executable, "--port", "0", "--workers", "2", "--memory-budget-mib", "20",
                        "--local-root", root.c_str(), "--local-manifest", manifest.c_str(),
                        "--deadline-ms", io.c_str(), nullptr);
                _exit(127);
            }
            ::execl(executable, executable, "--port", "0", "--workers", "2", "--memory-budget-mib", "20",
                    "--local-root", root.c_str(), "--local-manifest", manifest.c_str(),
                    "--residency-ms", lifetime.c_str(), "--deadline-ms", io.c_str(), nullptr);
            _exit(127);
        }
        ::close(fd);
        try {
            for (int i = 0; i < 500 && !port; ++i) {
                std::ifstream in(log);
                std::string line;
                while (std::getline(in, line)) {
                    // A writer can expose a partial final line. Wait for its newline first.
                    if (in.eof()) break;
                    port = listening_port(line);
                    if (port) break;
                }
                if (!port) pause_ms(10);
            }
            require(port, "binary listening");
        } catch (...) {
            signal_child(SIGKILL);
            reap();
            print_log();
            throw; // a throwing constructor does not run ~Binary()
        }
    }
    Binary(const Binary&) = delete;
    Binary& operator=(const Binary&) = delete;
    static uint16_t listening_port(const std::string& line) {
        const std::string prefix = "listening on 127.0.0.1:";
        const auto at = line.find(prefix);
        if (at == std::string::npos) return 0;
        const char* first = line.data() + at + prefix.size();
        const char* end = line.data() + line.size();
        unsigned value = 0;
        const auto parsed = std::from_chars(first, end, value);
        if (parsed.ec != std::errc{} || value == 0 || value > 65535 || parsed.ptr == end || *parsed.ptr != ' ')
            return 0;
        return (uint16_t)value;
    }
    // Never signal a process group or all user processes. Only the unreaped positive child
    // PID belongs to this harness; wait_exit() clears it even for a failed/signal exit.
    void signal_child(int sig) const {
        if (pid > 0) ::kill(pid, sig);
    }
    void reap() {
        if (pid <= 0) return;
        while (::waitpid(pid, nullptr, 0) < 0 && errno == EINTR) {}
        pid = -1;
    }
    void print_log() const {
        std::ifstream in(log);
        std::fprintf(stderr, "WORKER_LOG %s\n", log.c_str());
        std::string line;
        while (std::getline(in, line)) std::fprintf(stderr, "%s\n", line.c_str());
    }
    bool wait_exit() {
        if (pid <= 0) return false;
        int status = 0;
        for (int i = 0; i < 200; ++i) {
            const auto result = ::waitpid(pid, &status, WNOHANG);
            if (result == pid) {
                pid = -1;
                const bool ok = WIFEXITED(status) && WEXITSTATUS(status) == 0;
                if (!ok) print_log();
                return ok;
            }
            if (result < 0 && errno == ECHILD) { pid = -1; return false; }
            pause_ms(10);
        }
        return false;
    }
    ~Binary() {
        signal_child(SIGTERM);
        if (pid > 0) wait_exit();
        if (pid > 0) {
            signal_child(SIGKILL);
            reap();
        }
    }
};
} // namespace

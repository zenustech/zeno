#include "NvrtcWorker.h"

#include "NvrtcCompileProtocol.h"

#include <sutil/sutil.h>

#include <nvrtc.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#if defined(_WIN32)
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN 1
#    endif
#    include <windows.h>
#else
#    include <cerrno>
#    include <signal.h>
#    include <sys/wait.h>
#    include <unistd.h>
#endif

#define STRINGIFY(x) STRINGIFY2(x)
#define STRINGIFY2(x) #x
#define LINE_STR STRINGIFY(__LINE__)

#define NVRTC_CHECK_ERROR(func)                                                                                           \
    do                                                                                                                    \
    {                                                                                                                     \
        nvrtcResult code = func;                                                                                          \
        if (code != NVRTC_SUCCESS)                                                                                        \
            throw std::runtime_error("ERROR: " __FILE__ "(" LINE_STR "): " + std::string(nvrtcGetErrorString(code)));    \
    } while (0)

namespace zeno::nvrtc_worker {
namespace {

void printNvrtcCompileFailure(const char* source, const char* log)
{
    std::string mod_cu_source = "1 ";
    int line = 1;
    if (source != nullptr) {
        mod_cu_source.reserve(std::strlen(source));
        for (const auto ch : std::string(source)) {
            mod_cu_source += ch;
            if (ch == '\n') {
                mod_cu_source += std::to_string(++line) + ' ';
            }
        }
    }

    std::cout << "NVRTC Compilation failed.\n===================BEGIN============\n";
    std::cout << mod_cu_source << "\n=============END==============\n";
    if (log != nullptr) {
        std::cout << log << std::endl;
    }
}

bool envFlagEnabled(const char* name, bool fallback)
{
    const char* env = std::getenv(name);
    if (!env) {
        return fallback;
    }
    return std::strcmp(env, "0") != 0 && std::strcmp(env, "false") != 0 && std::strcmp(env, "FALSE") != 0;
}

uint32_t envU32(const char* name, uint32_t fallback)
{
    const char* env = std::getenv(name);
    if (!env) {
        return fallback;
    }
    char* end = nullptr;
    const unsigned long value = std::strtoul(env, &end, 10);
    if (end == env || *end != '\0') {
        return fallback;
    }
    return static_cast<uint32_t>(value);
}

std::filesystem::path helperExecutablePath()
{
    if (const char* env = std::getenv("ZENO_NVRTC_HELPER")) {
        std::filesystem::path path(env);
        if (std::filesystem::exists(path)) {
            return path;
        }
    }

#if defined(_WIN32)
    char exe_path[MAX_PATH] = {};
    const DWORD len = GetModuleFileNameA(nullptr, exe_path, MAX_PATH);
    if (len > 0 && len < MAX_PATH) {
        auto path = std::filesystem::path(exe_path).parent_path() / "zeno_nvrtc_helper.exe";
        if (std::filesystem::exists(path)) {
            return path;
        }
    }
#else
    char exe_path[4096] = {};
    const ssize_t len = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
    if (len > 0) {
        exe_path[len] = '\0';
        auto path = std::filesystem::path(exe_path).parent_path() / "zeno_nvrtc_helper";
        if (std::filesystem::exists(path)) {
            return path;
        }
    }
#endif

    auto cwd_path = std::filesystem::current_path() /
#if defined(_WIN32)
        "zeno_nvrtc_helper.exe";
#else
        "zeno_nvrtc_helper";
#endif
    if (std::filesystem::exists(cwd_path)) {
        return cwd_path;
    }

    return {};
}

#if defined(_WIN32)
std::string quoteProcessArg(const std::filesystem::path& path)
{
    std::string s = path.string();
    std::string quoted = "\"";
    for (char c : s) {
        if (c == '"') {
            quoted += "\\\"";
        } else {
            quoted += c;
        }
    }
    quoted += "\"";
    return quoted;
}
#endif

bool runNvrtcHelperProcess(
    const std::filesystem::path& helper,
    const zeno_nvrtc_ipc::Request& request,
    zeno_nvrtc_ipc::Response& response)
{
    std::ostringstream request_stream(std::ios::out | std::ios::binary);
    zeno_nvrtc_ipc::writeRequest(request_stream, request);
    const std::string request_bytes = request_stream.str();

#if defined(_WIN32)
    const DWORD helper_timeout_ms = envU32("ZENO_NVRTC_HELPER_TIMEOUT_MS", 30000);
    static std::mutex helper_process_spawn_mutex;
    std::unique_lock<std::mutex> spawn_lock(helper_process_spawn_mutex);

    auto writeAllToHandle = [](HANDLE handle, const std::string& data) {
        const char* ptr = data.data();
        size_t remaining = data.size();
        while (remaining > 0) {
            const DWORD chunk = static_cast<DWORD>(std::min<size_t>(remaining, 1u << 20));
            DWORD written = 0;
            if (!WriteFile(handle, ptr, chunk, &written, nullptr) || written == 0) {
                return false;
            }
            ptr += written;
            remaining -= written;
        }
        return true;
    };

    auto readAllFromHandle = [](HANDLE handle, std::string& data) {
        char buffer[64 * 1024];
        for (;;) {
            DWORD bytes_read = 0;
            if (!ReadFile(handle, buffer, static_cast<DWORD>(sizeof(buffer)), &bytes_read, nullptr)) {
                return GetLastError() == ERROR_BROKEN_PIPE;
            }
            if (bytes_read == 0) {
                return true;
            }
            data.append(buffer, bytes_read);
        }
    };

    SECURITY_ATTRIBUTES security_attributes = {};
    security_attributes.nLength = sizeof(security_attributes);
    security_attributes.bInheritHandle = TRUE;

    HANDLE child_stdin_read = nullptr;
    HANDLE child_stdin_write = nullptr;
    HANDLE child_stdout_read = nullptr;
    HANDLE child_stdout_write = nullptr;
    HANDLE child_stderr_write = nullptr;

    if (!CreatePipe(&child_stdin_read, &child_stdin_write, &security_attributes, 0)) {
        return false;
    }
    if (!CreatePipe(&child_stdout_read, &child_stdout_write, &security_attributes, 0)) {
        CloseHandle(child_stdin_read);
        CloseHandle(child_stdin_write);
        return false;
    }
    child_stderr_write = CreateFileA(
        "NUL",
        GENERIC_WRITE,
        FILE_SHARE_READ | FILE_SHARE_WRITE,
        &security_attributes,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        nullptr);
    if (child_stderr_write == INVALID_HANDLE_VALUE) {
        CloseHandle(child_stdin_read);
        CloseHandle(child_stdin_write);
        CloseHandle(child_stdout_read);
        CloseHandle(child_stdout_write);
        return false;
    }

    SetHandleInformation(child_stdin_write, HANDLE_FLAG_INHERIT, 0);
    SetHandleInformation(child_stdout_read, HANDLE_FLAG_INHERIT, 0);

    std::string command = quoteProcessArg(helper);
    STARTUPINFOEXA startup_info = {};
    startup_info.StartupInfo.cb = sizeof(startup_info);
    startup_info.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
    startup_info.StartupInfo.hStdInput = child_stdin_read;
    startup_info.StartupInfo.hStdOutput = child_stdout_write;
    startup_info.StartupInfo.hStdError = child_stderr_write;

    SIZE_T attribute_list_size = 0;
    InitializeProcThreadAttributeList(nullptr, 1, 0, &attribute_list_size);
    std::vector<char> attribute_list_storage(attribute_list_size);
    startup_info.lpAttributeList = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(attribute_list_storage.data());
    if (!InitializeProcThreadAttributeList(startup_info.lpAttributeList, 1, 0, &attribute_list_size)) {
        CloseHandle(child_stdin_read);
        CloseHandle(child_stdin_write);
        CloseHandle(child_stdout_read);
        CloseHandle(child_stdout_write);
        CloseHandle(child_stderr_write);
        return false;
    }

    HANDLE inherited_handles[] = {child_stdin_read, child_stdout_write, child_stderr_write};
    if (!UpdateProcThreadAttribute(
            startup_info.lpAttributeList,
            0,
            PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
            inherited_handles,
            sizeof(inherited_handles),
            nullptr,
            nullptr)) {
        DeleteProcThreadAttributeList(startup_info.lpAttributeList);
        CloseHandle(child_stdin_read);
        CloseHandle(child_stdin_write);
        CloseHandle(child_stdout_read);
        CloseHandle(child_stdout_write);
        CloseHandle(child_stderr_write);
        return false;
    }

    PROCESS_INFORMATION process_info = {};
    const BOOL ok = CreateProcessA(
        nullptr,
        command.data(),
        nullptr,
        nullptr,
        TRUE,
        CREATE_NO_WINDOW | EXTENDED_STARTUPINFO_PRESENT,
        nullptr,
        nullptr,
        &startup_info.StartupInfo,
        &process_info);

    DeleteProcThreadAttributeList(startup_info.lpAttributeList);
    CloseHandle(child_stdin_read);
    CloseHandle(child_stdout_write);
    CloseHandle(child_stderr_write);

    if (!ok) {
        CloseHandle(child_stdin_write);
        CloseHandle(child_stdout_read);
        return false;
    }
    spawn_lock.unlock();

    std::string response_bytes;
    bool read_ok = false;
    bool write_ok = false;
    std::thread writer_thread([&]() {
        write_ok = writeAllToHandle(child_stdin_write, request_bytes);
        CloseHandle(child_stdin_write);
    });
    std::thread reader_thread([&]() {
        read_ok = readAllFromHandle(child_stdout_read, response_bytes);
        CloseHandle(child_stdout_read);
    });

    const DWORD wait_result = WaitForSingleObject(process_info.hProcess, helper_timeout_ms);
    if (wait_result == WAIT_TIMEOUT) {
        TerminateProcess(process_info.hProcess, 1);
        WaitForSingleObject(process_info.hProcess, INFINITE);
    }
    writer_thread.join();
    reader_thread.join();
    CloseHandle(process_info.hThread);
    CloseHandle(process_info.hProcess);

    if (wait_result != WAIT_OBJECT_0 || !write_ok || !read_ok || response_bytes.empty()) {
        return false;
    }

    try {
        std::istringstream response_stream(response_bytes, std::ios::in | std::ios::binary);
        response = zeno_nvrtc_ipc::readResponse(response_stream);
        return true;
    } catch (...) {
        return false;
    }
#else
    const uint32_t helper_timeout_ms = envU32("ZENO_NVRTC_HELPER_TIMEOUT_MS", 30000);
    auto writeAllToFd = [](int fd, const std::string& data) {
        const char* ptr = data.data();
        size_t remaining = data.size();
        while (remaining > 0) {
            const ssize_t written = write(fd, ptr, remaining);
            if (written < 0) {
                if (errno == EINTR) {
                    continue;
                }
                return false;
            }
            if (written == 0) {
                return false;
            }
            ptr += written;
            remaining -= static_cast<size_t>(written);
        }
        return true;
    };

    auto readAllFromFd = [](int fd, std::string& data) {
        char buffer[64 * 1024];
        for (;;) {
            const ssize_t bytes_read = read(fd, buffer, sizeof(buffer));
            if (bytes_read < 0) {
                if (errno == EINTR) {
                    continue;
                }
                return false;
            }
            if (bytes_read == 0) {
                return true;
            }
            data.append(buffer, static_cast<size_t>(bytes_read));
        }
    };

    int stdin_pipe[2] = {-1, -1};
    int stdout_pipe[2] = {-1, -1};
    if (pipe(stdin_pipe) != 0) {
        return false;
    }
    if (pipe(stdout_pipe) != 0) {
        close(stdin_pipe[0]);
        close(stdin_pipe[1]);
        return false;
    }

    const pid_t pid = fork();
    if (pid < 0) {
        close(stdin_pipe[0]);
        close(stdin_pipe[1]);
        close(stdout_pipe[0]);
        close(stdout_pipe[1]);
        return false;
    }

    if (pid == 0) {
        dup2(stdin_pipe[0], STDIN_FILENO);
        dup2(stdout_pipe[1], STDOUT_FILENO);
        close(stdin_pipe[0]);
        close(stdin_pipe[1]);
        close(stdout_pipe[0]);
        close(stdout_pipe[1]);

        const std::string helper_string = helper.string();
        execl(helper_string.c_str(), helper.filename().string().c_str(), static_cast<char*>(nullptr));
        _exit(127);
    }

    close(stdin_pipe[0]);
    close(stdout_pipe[1]);

    std::string response_bytes;
    bool read_ok = false;
    bool write_ok = false;
    std::thread writer_thread([&]() {
        write_ok = writeAllToFd(stdin_pipe[1], request_bytes);
        close(stdin_pipe[1]);
    });
    std::thread reader_thread([&]() {
        read_ok = readAllFromFd(stdout_pipe[0], response_bytes);
        close(stdout_pipe[0]);
    });

    int status = 0;
    bool timed_out = false;
    const auto wait_begin = std::chrono::steady_clock::now();
    for (;;) {
        const pid_t wait_result = waitpid(pid, &status, WNOHANG);
        if (wait_result == pid) {
            break;
        }
        if (wait_result < 0 && errno != EINTR) {
            break;
        }
        const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - wait_begin).count();
        if (elapsed_ms >= helper_timeout_ms) {
            timed_out = true;
            kill(pid, SIGKILL);
            while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {
            }
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    writer_thread.join();
    reader_thread.join();

    if (timed_out || !write_ok || !read_ok || response_bytes.empty()) {
        return false;
    }

    try {
        std::istringstream response_stream(response_bytes, std::ios::in | std::ios::binary);
        response = zeno_nvrtc_ipc::readResponse(response_stream);
        return true;
    } catch (...) {
        return false;
    }
#endif
}

bool compileShaderCudaInHelper(
    std::string& compiled,
    const char* source,
    const char* name,
    const char** log,
    const std::vector<const char*>& options,
    bool& success,
    bool allow_helper)
{
    if (!allow_helper) {
        return false;
    }

    if (!envFlagEnabled("ZENO_NVRTC_MULTIPROCESS", true)) {
        return false;
    }

    const auto helper = helperExecutablePath();
    if (helper.empty()) {
        return false;
    }

    try {
        zeno_nvrtc_ipc::Request request;
        request.source = source != nullptr ? source : "";
        request.name = name != nullptr ? name : "";
        request.options.reserve(options.size());
        for (const char* option : options) {
            request.options.emplace_back(option != nullptr ? option : "");
        }

        zeno_nvrtc_ipc::Response response;
        if (!runNvrtcHelperProcess(helper, request, response)) {
            return false;
        }

        compiled = std::move(response.data);
        static thread_local std::string helper_log;
        helper_log = std::move(response.log);
        if (log) {
            *log = helper_log.empty() ? nullptr : helper_log.c_str();
        }
        success = response.success;

        return true;
    } catch (...) {
        return false;
    }
}

bool compileShaderCuda(
    std::string& compiled,
    const char* source,
    const char* name,
    const char** log,
    const std::vector<const char*>& options,
    bool allow_helper)
{
    bool helper_success = false;
    if (compileShaderCudaInHelper(compiled, source, name, log, options, helper_success, allow_helper)) {
        if (!helper_success) {
            printNvrtcCompileFailure(source, log != nullptr ? *log : nullptr);
        }
        return helper_success;
    }

    nvrtcProgram prog;
    NVRTC_CHECK_ERROR(nvrtcCreateProgram(&prog, source, name, sutil::getIncFileTab().size(), sutil::getIncFileTab().data(), sutil::getIncPathTab().data()));

    const nvrtcResult compileRes = nvrtcCompileProgram(prog, static_cast<int>(options.size()), options.data());

    std::string nvrtc_log;
    size_t log_size = 0;
    NVRTC_CHECK_ERROR(nvrtcGetProgramLogSize(prog, &log_size));

    if (log_size > 1) {
        nvrtc_log.resize(log_size);
        NVRTC_CHECK_ERROR(nvrtcGetProgramLog(prog, nvrtc_log.data()));
    }

    if (compileRes != NVRTC_SUCCESS) {
        printNvrtcCompileFailure(source, nvrtc_log.c_str());
        NVRTC_CHECK_ERROR(nvrtcDestroyProgram(&prog));
        std::cout << "mabimabi not compiled!!!!!" << std::endl;
        return false;
    }

    size_t code_size = 0;
    bool using_ir = false;
    for (auto opt : options) {
        if (opt != nullptr && std::strcmp(opt, "--optix-ir") == 0) {
            using_ir = true;
            break;
        }
    }

    if (using_ir) {
        NVRTC_CHECK_ERROR(nvrtcGetOptiXIRSize(prog, &code_size));
        compiled.resize(code_size);
        NVRTC_CHECK_ERROR(nvrtcGetOptiXIR(prog, compiled.data()));
    } else {
        NVRTC_CHECK_ERROR(nvrtcGetPTXSize(prog, &code_size));
        compiled.resize(code_size);
        NVRTC_CHECK_ERROR(nvrtcGetPTX(prog, compiled.data()));
    }

    NVRTC_CHECK_ERROR(nvrtcDestroyProgram(&prog));
    return true;
}

} // namespace

CompileResult compile(
    const char* source,
    const char* macro,
    const char* name,
    const std::vector<const char*>& compilerOptions)
{
    (void)macro;

    bool allow_helper = true;
    std::vector<const char*> effectiveCompilerOptions;
    effectiveCompilerOptions.reserve(compilerOptions.size());
    for (const char* option : compilerOptions) {
        if (option != nullptr && std::strcmp(option, "--zeno-nvrtc-worker=0") == 0) {
            allow_helper = false;
            continue;
        }
        if (option != nullptr && std::strcmp(option, "--zeno-nvrtc-worker=1") == 0) {
            allow_helper = true;
            continue;
        }
        effectiveCompilerOptions.push_back(option);
    }

    CompileResult result;
    const char* log = nullptr;
    result.success = compileShaderCuda(
        result.data,
        source,
        name,
        &log,
        effectiveCompilerOptions,
        allow_helper);
    if (log != nullptr) {
        result.log = log;
    }
    if (!result.success) {
        result.data.clear();
    }
    return result;
}

} // namespace zeno::nvrtc_worker

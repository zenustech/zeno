#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <string>
#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#include <direct.h>
#elif defined(__linux__) || defined(__APPLE__) || defined(__unix__)
#include <unistd.h>
#endif

static const char logfile[] = "zeno_output.txt";

static std::string my_dirname(std::string const &str)
{
    return str.substr(0, str.find_last_of("/\\"));
}

static void report() {
    printf("Platform: %s\n",
#if defined(_WIN64)
    "Windows (x64)"
#elif defined(_WIN32)
    "Windows"
#elif defined(__linux__)
    "Linux"
#elif defined(__APPLE__)
    "Mac OS X"
#elif defined(__unix__)
    "Other Unix"
#elif defined(_AIX)
    "AIX"
#elif defined(__sun__)
    "Solaris"
#else
    "Unknown"
#endif
    );
    printf("Build: %s %s\n", __DATE__, __TIME__);
    printf("Compiler: %s %d, C++ %ld\n",
#if defined(_MSC_VER)
    "MSVC", _MSC_VER
#elif defined(__clang__)
    "Clang", __clang__
#elif defined(__GNUC__)
    "GCC", __GNUC__
#elif defined(__DPCPP__)
    "DPC++", __DPCPP__
#else
    "Unknown", 0
#endif
#if defined(__cplusplus)
    , __cplusplus
#else
    , 0l
#endif
    );
    printf("CPU cores: %d\n", std::thread::hardware_concurrency());
#if defined(_WIN32) || defined(_WIN64)
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    GlobalMemoryStatusEx(&status);
    printf("Memory: %ld MiB\n", status.ullTotalPhys / 1024 / 1024);
#elif defined(__linux__) || defined(__APPLE__) || defined(__unix__)
    long memsize = sysconf(_SC_PHYS_PAGES) * sysconf(_SC_PAGE_SIZE);
    printf("Memory: %ld MiB\n", memsize / 1024 / 1024);
#endif
}

static void failure(int stat) {
    fprintf(stderr, "================================================================\n");
    fprintf(stderr, "Sorry, the program crashed! Would you mind report this by either:\n\n");
    fprintf(stderr, "  1. Opening a GitHub issue on: https://github.com/zenustech/zeno/issues\n");
    fprintf(stderr, "  2. Send the report to maintainers via WeChat: tanh233 or shinshinzhang\n\n");
    fprintf(stderr, "Also please consider attach this file:\n\n  %s\n\n", logfile);
    fprintf(stderr, "so that we could locate the problem easier? Thank for your help!\n");
    fprintf(stderr, "Would also be helpful if you'd like to attach details about the\n");
    fprintf(stderr, "current editing graph (if any), the graphics card you use, and\n");
    fprintf(stderr, "other possible information might related to the problem, thanks!\n");
    fprintf(stderr, "================================================================\n");
#if defined(_WIN32)
    system("pause");
#endif
    exit(stat);
}

#if defined(_WIN32)
#define popen(x, y) _popen(x, y)
#define pclose(x) _pclose(x)
#define setenv(x, y, z) SetEnvironmentVariable(x, y)
#endif

static void start(const char *path) {
    char *buf = (char *)alloca(strlen(path) + 64);
    sprintf(buf, "'%s' 2>&1", path);
    setenv("ZEN_LOGLEVEL", "debug", 0);
    fprintf(stderr, ">>> launching ZENO now <<<\n");
    printf("launching command: %s\n", buf);
    FILE *pipe = popen(buf, "r");
    if (!pipe) {
        perror(buf);
        printf("failed to launch\n");
        failure(1);
    }
    char c;
    printf(">>> begin of log <<<\n");
    while ((c = fgetc(pipe)) != EOF) {
        fputc(c, stdout);
        fputc(c, stderr);
    }
    printf(">>> end of log <<<\n");
    int stat = pclose(pipe);
    printf("exit code: %d (0x%x)\n", stat, stat);
    if (stat) {
        failure(stat);
    }
}

int main(int argc, char **argv) {
    auto path = my_dirname(argv[0]);
    fprintf(stderr, "==> release date: %s\n", __DATE__);
    fprintf(stderr, "==> working directory: %s\n", path.c_str());
#if defined(_WIN32)
    SetCurrentDirectory(path.c_str());
#else
    chdir(path.c_str());
#endif
    if (argv[1]) {
        setenv("ZEN_OPEN", argv[1], 1);
    }
    freopen(logfile, "w", stdout);
    report();
    start(
#if defined(_WIN32)
            ".\\zenqte.exe"
#else
            "./zenqte"
#endif
            );
    return 0;
}

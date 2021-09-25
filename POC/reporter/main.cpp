#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#elif defined(__linux__) || defined(__APPLE__) || defined(__unix__)
#include <unistd.h>
#endif

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
#elif defined(__GNUC__)
    "GCC", __GNUC__
#elif defined(__clang__)
    "Clang", __clang__
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

static void start(const char *path) {
    char *buf = (char *)alloca(strlen(path) + 64);
    sprintf(buf, "'%s' 2>&1", path);
    FILE *pipe = popen(buf, "r");
    if (!pipe) {
        perror("popen");
        abort();
    }
    char c;
    printf("=== begin of output\n");
    while ((c = fgetc(pipe)) != EOF) {
        fputc(c, stdout);
    }
    printf("=== end of output\n");
    pclose(pipe);
}

int main() {
    report();
    start("ls");
    return 0;
}

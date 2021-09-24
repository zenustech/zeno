#include <cstdio>

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
#else
    "Unknown", 0
#endif
#if defined(__cplusplus)
    , __cplusplus
#else
    , 0l
#endif
    );
}

int main() {
    report();
    return 0;
}

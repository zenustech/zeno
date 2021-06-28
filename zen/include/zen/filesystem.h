#pragma once

#if __has_include(<filesystem>)
#include <filesystem>
namespace zen::fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
namespace zen::fs = std::experimental::filesystem;
#else
#error "missing <filesystem> header."
#endif

#endif

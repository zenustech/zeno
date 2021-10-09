#pragma once

#if __has_include(<filesystem>)
#include <filesystem>
namespace zeno {
namespace fs = std::filesystem;
}
#elif __has_include(<experimental/filesystem>)
#include <experimental/filesystem>
namespace zeno {
namespace fs = std::experimental::filesystem;
}
#else
#error "missing <filesystem> header."
#endif

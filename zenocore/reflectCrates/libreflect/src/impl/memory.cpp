#include "utils/memory"

using namespace zeno::reflect;

void* memmove(void *dest, const void *src, size_t n)
{
    auto d = static_cast<uint8_t*>(dest);
    auto s = static_cast<const uint8_t*>(src);

    if (d < s) {
        for (size_t i = 0; i < n; ++i) {
            d[i] = s[i];
        }
    } else {
        for (size_t i = n; i != 0; --i) {
            d[i - 1] = s[i - 1];
        }
    }

    return dest;
}
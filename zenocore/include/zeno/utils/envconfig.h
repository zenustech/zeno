#pragma once

#include <string>
#include <cstdlib>
#include <algorithm>

namespace zeno::envconfig {

static char *get(const char *key) {
    return std::getenv((std::string("ZENO_") + key).c_str());
}

static bool has(const char *key) {
    if (auto p = get(key); p && *p)
        return true;
    else
        return false;
}

static int getInt(const char *key, int defl = 0) {
    if (auto p = get(key); p && *p)
        return std::stoi(p);
    else
        return defl;
}

static uint64_t getUint64(const char *key, uint64_t defl = 0) {
    if (auto p = get(key); p && *p)
        return std::stoull(p);
    else
        return defl;
}

static char *getCStr(const char *key, const char *defl = nullptr) {
    if (auto p = get(key); p && *p)
        return p;
    else
        return (char *)defl;
}

static std::string getStr(const char *key, std::string defl = {}) {
    if (auto p = get(key); p && *p)
        return p;
    else
        return defl;
}

static bool getBool(const char *key, bool defl = false) {
    if (auto p = get(key); p && *p)
        return p != std::string("0");
    else
        return defl;
}

template <class Enum = std::size_t, std::size_t N>
static Enum getEnum(const char *key, const char (&arr)[N], Enum defl = Enum()) {
    std::size_t i(std::find(arr, arr + N, getStr(key)) - arr);
    if (i == N) return defl;
    return Enum(i);
}

}

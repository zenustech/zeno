#pragma once

#include <zeno/utils/scope_exit.h>

#if defined(_MSC_VER)
# if defined(ZENO_DLLEXPORT)
#  define ZENO_API __declspec(dllexport)
# elif defined(ZENO_DLLIMPORT)
#  define ZENO_API __declspec(dllimport)
# else
#  define ZENO_API
# endif
#else
# define ZENO_API
#endif

#define ZENO_VERSION_2

#define CALLBACK_REGIST(api_name, ret_type, ...) \
    ZENO_API std::string register_##api_name(std::function<ret_type(__VA_ARGS__)> cb_func) {\
        const std::string& uuid = zeno::generateUUID();\
        m_callback_##api_name.insert(std::make_pair(uuid, cb_func));\
        return uuid;\
    }\
    \
    ZENO_API bool unregister_##api_name(const std::string& uuid) {\
        if (m_callback_##api_name.find(uuid) == m_callback_##api_name.end())\
            return false;\
        m_callback_##api_name.erase(uuid);\
        return true;\
    }\
    std::unordered_map<std::string, std::function<ret_type(__VA_ARGS__)>> m_callback_##api_name;

#define CALLBACK_NOTIFY(api_name, ...) \
    for (auto& [key, func] : m_callback_##api_name) { \
        func(__VA_ARGS__);\
    }

#define CORE_API_BATCH \
    getSession().beginApiCall();\
    zeno::scope_exit([](){ getSession().endApiCall(); });
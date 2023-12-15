#pragma once

#ifdef _MSC_VER
#define ZENO_MUSTLINK_IMPORT __declspec(dllimport)
#define ZENO_MUSTLINK_EXPORT __declspec(dllexport)
#else
#define ZENO_MUSTLINK_IMPORT
#define ZENO_MUSTLINK_EXPORT
#endif

#define ZENO_REQUIRE(name) \
    namespace zeno::mustlink { \
        ZENO_MUSTLINK_IMPORT int require_##name(); \
        static int _define_##name = require_##name(); \
    }

#define ZENO_PACKAGE(name) \
    namespace zeno::mustlink { \
        ZENO_MUSTLINK_EXPORT int require_##name() { return 42; } \
    }

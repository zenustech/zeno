#pragma once

namespace zeno {
namespace ansiclr {
    enum color_t {
        black = 0,
        red = 1,
        green = 2,
        yellow = 3,
        blue = 4,
        magenta = 5,
        cyan = 6,
        white = 7,
        light = 8,
    };

#ifdef __unix__
    static constexpr const char *reset = "\033[0m";
    static constexpr const char *fg[16] = {
        "\033[30m", "\033[31m", "\033[32m", "\033[33m",
        "\033[34m", "\033[35m", "\033[36m", "\033[37m",
        "\033[30;1m", "\033[31;1m", "\033[32;1m", "\033[33;1m",
        "\033[34;1m", "\033[35;1m", "\033[36;1m", "\033[37;1m",
    };
    static constexpr const char *bg[16] = {
        "\033[40m", "\033[41m", "\033[42m", "\033[43m",
        "\033[44m", "\033[45m", "\033[46m", "\033[47m",
        "\033[40;1m", "\033[41;1m", "\033[42;1m", "\033[43;1m",
        "\033[44;1m", "\033[45;1m", "\033[46;1m", "\033[47;1m",
    };
#else
    static constexpr const char *reset = "";
    static constexpr const char *fg[16] = {
        "", "", "", "",
        "", "", "", "",
        "", "", "", "",
        "", "", "", "",
    };
    static constexpr const char *bg[16] = {
        "", "", "", "",
        "", "", "", "",
        "", "", "", "",
        "", "", "", "",
    };
#endif
}
}

#pragma once

namespace zeno {
    
struct disable_copy {
    disable_copy() = default;
    disable_copy(disable_copy const &) = delete;
    disable_copy(disable_copy &&) = delete;
    disable_copy &operator=(disable_copy const &) = delete;
    disable_copy &operator=(disable_copy &&) = delete;
};

struct disable_copy_allow_move {
    disable_copy_allow_move() = default;
    disable_copy_allow_move(disable_copy_allow_move const &) = delete;
    disable_copy_allow_move(disable_copy_allow_move &&) = default;
    disable_copy_allow_move &operator=(disable_copy_allow_move const &) = delete;
    disable_copy_allow_move &operator=(disable_copy_allow_move &&) = default;
};

struct explicit_default_ctor {
    explicit explicit_default_ctor() = default;
};

}

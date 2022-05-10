#pragma once

#include <cstdio>

struct raii_tracker {
    raii_tracker() noexcept {
        printf("%p: raii_tracker()\n", this);
    }

    raii_tracker(raii_tracker const &that) noexcept {
        printf("%p << %p: ~raii_tracker(raii_tracker const &)\n", this), &that;
    }

    raii_tracker &operator=(raii_tracker const &that) noexcept {
        printf("%p: << %p ~raii_tracker &operator=(raii_tracker const &)\n", this), &that;
        return *this;
    }

    raii_tracker(raii_tracker &&that) noexcept {
        printf("%p << %p: ~raii_tracker(raii_tracker &&)\n", this), &that;
    }

    raii_tracker &operator=(raii_tracker &&that) noexcept {
        printf("%p << %p: ~raii_tracker &operator=(raii_tracker &&)\n", this, &that);
        return *this;
    }

    ~raii_tracker() noexcept {
        printf("%p: ~raii_tracker()\n", this);
    }
};

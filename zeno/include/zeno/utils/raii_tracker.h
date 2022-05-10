#pragma once

#include <cstdio>

struct raii_tracker {
    raii_tracker() noexcept {
        printf("raii_tracker()\n");
    }

    raii_tracker(raii_tracker const &) noexcept {
        printf("~raii_tracker(raii_tracker const &)\n");
    }

    raii_tracker &operator=(raii_tracker const &) noexcept {
        printf("~raii_tracker &operator=(raii_tracker const &)\n");
    }

    raii_tracker(raii_tracker &&) noexcept {
        printf("~raii_tracker(raii_tracker &&)\n");
    }

    raii_tracker &operator=(raii_tracker &&) noexcept {
        printf("~raii_tracker &operator=(raii_tracker &&)\n");
    }

    ~raii_tracker() noexcept {
        printf("~raii_tracker()\n");
    }
};

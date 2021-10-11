#pragma once

#include <memory>

#define ZS_ZTD_PIMPL_DECL(Class, ...) \
    struct Class {
        std::shared_ptr<Self> self;
        Class() = default;
        Class(Class const &) = default;
        Class &operator=(Class const &) = default;
        Class(Class &&) = default;
        Class &operator=(Class &&) = default;
        Class(std::shared_ptr<Self> const &self) : self(self) {}
        __VA_ARGS__
    };

#define ZS_ZTD_PIMPL_IMPL(Class, ...) \
Class::Class(std::shared_ptr<Self> const &self) : self(self) {}

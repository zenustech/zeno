#pragma once

#include <exception>
#include <zeno/dop/Node.h>
#include <zeno/zmt/format.h>

ZENO_NAMESPACE_BEGIN
namespace dop {

class Exception : public std::exception {
    Node *node{};
    std::exception_ptr ep;
public:
    Exception(Node *node, std::exception_ptr ep) noexcept
        : node(node), ep(std::move(ep)) {}

    inline Node *getNode() const {
        return node;
    }

    virtual char const *what() const noexcept {
        try {
            if (ep) {
                std::rethrow_exception(ep);
            }
        } catch (std::exception const &e) {
            return e.what();
        } catch (...) {
            return "unknown exception";
        }
        return "no exception";
    }

    ~Exception() noexcept = default;
};

}
ZENO_NAMESPACE_END

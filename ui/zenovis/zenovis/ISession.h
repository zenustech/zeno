#pragma once

#include <memory>

namespace zenovis {

struct ISession {
    virtual ~ISession() = default;
};

std::unique_ptr<ISession> makeSession();

}

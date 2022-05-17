#pragma once

#include <string>

struct ZSerializable {
    virtual std::string z_serialize() const = 0;
    virtual void z_deserialize(std::string const &s) = 0;
};

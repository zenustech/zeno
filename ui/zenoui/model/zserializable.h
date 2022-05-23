#pragma once

#include <string>

// todo luzh: seems qt will crash on multi-inherience
struct ZSerializable {
    virtual std::string z_serialize() const = 0;
    virtual void z_deserialize(std::string_view s) = 0;
};

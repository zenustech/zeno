#pragma once

#include "vec.h"


namespace fdb {


enum class Access {
    read, write, read_write, discard_write, discard_read_write, atomic,
};


static constexpr struct HostHandler {} HOST;


}

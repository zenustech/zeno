#pragma once

#include <zeno/core/IObject.h>

namespace zeno {

struct DummyObject : IObjectClone<DummyObject> {
    /* nothing, just a empty object for fake static view object stubs */
};

}

#pragma once

#include <zeno/utils/api.h>
#include <zeno/core/IObject.h>
#include <vector>
#include <string>
#include <memory>

namespace zeno {

ZENO_API zany decodeObject(const char *buf, size_t len);
ZENO_API bool encodeObject(IObject *object, std::vector<char> &buf);

}

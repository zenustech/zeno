#pragma once

#include <zeno/core/IObject.h>
#include <vector>
#include <string>
#include <memory>

namespace zeno {

ZENO_API std::shared_ptr<IObject> decodeObject(const char *buf, size_t len);
ZENO_API bool encodeObject(IObject const *object, std::vector<char> &buf);

}

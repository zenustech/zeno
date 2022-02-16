#pragma once

#include <zeno/core/IObject.h>
#include <vector>
#include <string>
#include <memory>

namespace zeno {

std::shared_ptr<IObject> decodeObject(const char *buf, size_t len);
bool encodeObject(IObject const *object, std::vector<char> &buf);

}

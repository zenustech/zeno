#include <zeno/types/StringObject.h>
#include <zeno/funcs/ObjectCodec.h>
#include <zeno/utils/log.h>
#include <algorithm>
#include <cstring>
namespace zeno {

namespace _implObjectCodec {

std::shared_ptr<StringObject> decodeStringObject(const char *it);
std::shared_ptr<StringObject> decodeStringObject(const char *it) {
    auto obj = std::make_shared<StringObject>();
    size_t size = *(int *)it;
    it += sizeof(size);
    obj->value.assign(it, size);
    return obj;
}

bool encodeStringObject(StringObject const *obj, std::back_insert_iterator<std::vector<char>> it);
bool encodeStringObject(StringObject const *obj, std::back_insert_iterator<std::vector<char>> it) {
    size_t size = obj->value.size();
    char const *data = obj->value.data();
    it = std::copy_n((char const *)&size, sizeof(size), it);
    it = std::copy_n(data, size, it);
    return true;
}

}

}

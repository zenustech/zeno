#include <zeno/funcs/ObjectCodec.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/utils/cppdemangle.h>
#include <zeno/utils/log.h>
#include <algorithm>
#include <cstring>

namespace zeno {

namespace {

enum class ObjectType {
    PrimitiveObject,
};

struct ObjectHeader {
    ObjectType type;
};

}

namespace _implObjectCodec {

std::shared_ptr<PrimitiveObject> decodePrimitiveObject(const char *it);
bool encodePrimitiveObject(PrimitiveObject const *obj, std::back_insert_iterator<std::vector<char>> it);

}

using namespace _implObjectCodec;

std::shared_ptr<IObject> decodeObject(const char *buf, size_t len) {
    if (len < sizeof(ObjectHeader)) {
        log_warn("data too short, giving up");
        return nullptr;
    }
    auto &header = *(ObjectHeader *)buf;
    auto it = buf + sizeof(ObjectHeader);

    if (header.type == ObjectType::PrimitiveObject) {
        return decodePrimitiveObject(it);

    } else {
        log_warn("invalid object type {}", header.type);
        return nullptr;
    }
}

bool encodeObject(IObject const *object, std::vector<char> &buf) {
    buf.resize(sizeof(ObjectHeader));
    auto &header = *(ObjectHeader *)buf.data();
    auto it = std::back_inserter(buf);

    if (auto obj = dynamic_cast<PrimitiveObject const *>(object)) {
        header.type = ObjectType::PrimitiveObject;
        return encodePrimitiveObject(obj, it);

    } else {
        log_warn("invalid object type `{}`", cppdemangle(typeid(*object)));
        return false;
    }
}

}

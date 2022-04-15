#include <zeno/funcs/ObjectCodec.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/utils/cppdemangle.h>
#include <zeno/types/UserData.h>
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
    size_t numUserData;
    size_t beginUserData;
};

}

namespace _implObjectCodec {

std::shared_ptr<PrimitiveObject> decodePrimitiveObject(const char *it);
bool encodePrimitiveObject(PrimitiveObject const *obj, std::back_insert_iterator<std::vector<char>> it);
std::shared_ptr<PrimitiveObject> decodeNumericObject(const char *it);
bool encodeNumericObject(PrimitiveObject const *obj, std::back_insert_iterator<std::vector<char>> it);

}

using namespace _implObjectCodec;

std::shared_ptr<IObject> _decodeObjectImpl(const char *buf, size_t len) {
    if (len < sizeof(ObjectHeader)) {
        log_error("data too short, giving up");
        return nullptr;
    }
    auto &header = *(ObjectHeader *)buf;
    auto it = buf + sizeof(ObjectHeader);

    if (header.type == ObjectType::PrimitiveObject) {
        return decodePrimitiveObject(it);

    } else {
        log_error("invalid object type {}", header.type);
        return nullptr;
    }
}

std::shared_ptr<IObject> decodeObject(const char *buf, size_t len) {
    auto object = _decodeObjectImpl(buf, len);

    auto &header = *(ObjectHeader *)buf;
    auto ptr = buf + header.beginUserData;
    for (int i = 0; i < header.numUserData; i++) {
        size_t valbufsize = *(size_t *)ptr;
        ptr += sizeof(valbufsize);
        auto nextptr = ptr + valbufsize;

        size_t keysize = *(size_t *)ptr;
        ptr += sizeof(keysize);
        std::string key{ptr, keysize};
        ptr += keysize;

        auto decolen = valbufsize > keysize ? valbufsize - keysize : 0;
        auto val = decodeObject(ptr, decolen);
        object->userData().set(key, std::move(val));

        ptr = nextptr;
    }

    return object;
}

static bool _encodeObjectImpl(IObject const *object, std::vector<char> &buf) {
    auto it = std::back_inserter(buf);
    it = std::fill_n(it, sizeof(ObjectHeader), 0);
    auto &header = *(ObjectHeader *)buf.data();

    if (auto obj = dynamic_cast<PrimitiveObject const *>(object)) {
        header.type = ObjectType::PrimitiveObject;
        return encodePrimitiveObject(obj, it);

    } else {
        log_error("invalid object type `{}`", cppdemangle(typeid(*object)));
        return false;
    }
}

bool encodeObject(IObject const *object, std::vector<char> &buf) {
    bool ret = _encodeObjectImpl(object, buf);
    ObjectHeader &header = *(ObjectHeader *)buf.data();

    std::vector<std::vector<char>> valbufs;
    for (auto const &[key, val]: object->userData()) {
        std::vector<char> valbuf;
        size_t keysize = key.size();
        valbuf.insert(valbuf.end(), (char *)&keysize, (char *)(&keysize + 1));
        valbuf.insert(valbuf.end(), key.begin(), key.end());
        if (encodeObject(val.get(), valbuf))
            valbufs.push_back(std::move(valbuf));
    }
    header.numUserData = valbufs.size();
    header.beginUserData = buf.size();
    for (auto const &valbuf: valbufs) {
        size_t valbufsize = valbuf.size();
        buf.insert(buf.end(), (char *)&valbufsize, (char *)(&valbufsize + 1));
        buf.insert(buf.end(), valbuf.begin(), valbuf.end());
    }
    return ret;
}

}

#include <zeno/types/ObjectCodec.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/utils/cppdemangle.h>
#include <zeno/utils/variantswitch.h>
#include <zeno/utils/log.h>
#include <cstring>

namespace zeno {

namespace {

enum class ObjectType {
    PrimitiveObject,
};

struct ObjectHeader {
    ObjectType type;
};

enum class AttributeType {
    Vec3f,
    Float,
};

using AttrTypesTuple = std::tuple<vec3f, float>;

struct AttributeHeader {
    AttributeType type;
    size_t size;
    size_t namelen;
    char name[128];
};

struct AttrVectorHeader {
    size_t size;
    size_t nattrs;
};

template <class T0, class It>
bool decodeAttrVector(AttrVector<T0> &arr, It it) {
    AttrVectorHeader header;
    std::copy_n(it, sizeof(header), (char *)&header);
    it += sizeof(header);
    arr.reserve(header.size);
    std::copy_n(it, header.size, std::back_inserter(arr));
    it += header.size;

    for (int a = 0; a < header.nattrs; a++) {
        AttributeHeader header;
        std::copy_n(it, sizeof(header), (char *)&header);
        it += sizeof(header);
        std::string key{header.name, header.namelen};
        index_switch<std::tuple_size_v<AttrTypesTuple>>((size_t)header.type, [&] (auto type) {
            using T = std::tuple_element_t<type.value, AttrTypesTuple>;
            auto &attr = arr.template add_attr<T>();
            attr.reserve(header.size);
            std::copy_n(it, header.size, std::back_inserter(attr));
            it += header.size;
        });
    }
}

template <class T0, class It>
bool encodeAttrVector(AttrVector<T0> const &arr, It it) {
    AttrVectorHeader header;
    header.size = arr.size();
    header.nattrs = arr.num_attrs();
    it = std::copy_n((char *)&header, sizeof(header), it);
    it = std::copy_n(arr.data(), arr.size(), it);

    arr.foreach_attr([&] (auto const &key, auto const &attr) {
        AttributeHeader header;
        using T = std::decay_t<decltype(attr[0])>;
        if constexpr (std::is_same_v<T, float>) {
            header.type = AttributeType::Float;
        } else if constexpr (std::is_same_v<T, vec3f>) {
            header.type = AttributeType::Vec3f;
        } else {
            static_assert(std::is_void_v<std::is_void<T>>);
        }
        header.size = attr.size();
        header.namelen = key.size();
        std::strncpy(header.name, key.c_str(), sizeof(header.name));
        it = std::copy_n((char *)&header, sizeof(header), it);
        std::copy_n(attr.data(), attr.size(), it);
    });
}

template <class It>
std::shared_ptr<PrimitiveObject> decodePrimitiveObject(It it) {
    auto obj = std::make_shared<PrimitiveObject>();
    decodeAttrVector(obj->verts, it);
    decodeAttrVector(obj->points, it);
    decodeAttrVector(obj->lines, it);
    decodeAttrVector(obj->tris, it);
    decodeAttrVector(obj->quads, it);
    decodeAttrVector(obj->loops, it);
    decodeAttrVector(obj->polys, it);
    return obj;
}

template <class It>
bool encodePrimitiveObject(PrimitiveObject const *obj, It it) {
    encodeAttrVector(obj->verts, it);
    encodeAttrVector(obj->points, it);
    encodeAttrVector(obj->lines, it);
    encodeAttrVector(obj->tris, it);
    encodeAttrVector(obj->quads, it);
    encodeAttrVector(obj->loops, it);
    encodeAttrVector(obj->polys, it);
    return true;
}

}

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

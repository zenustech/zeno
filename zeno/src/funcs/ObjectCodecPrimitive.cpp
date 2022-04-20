#include <zeno/funcs/ObjectCodec.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/MaterialObject.h>
#include <zeno/utils/variantswitch.h>
#include <zeno/utils/log.h>
#include <algorithm>
#include <cstring>
namespace zeno {

namespace _implObjectCodec {

namespace {

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
void decodeAttrVector(AttrVector<T0> &arr, It &it) {
    AttrVectorHeader header;
    std::copy_n(it, sizeof(header), (char *)&header);
    it += sizeof(header);
    arr.values.reserve(header.size);
    std::copy_n((T0 const *)it, header.size, std::back_inserter(arr.values));
    it += sizeof(T0) * header.size;

    for (int a = 0; a < header.nattrs; a++) {
        AttributeHeader header;
        std::copy_n(it, sizeof(header), (char *)&header);
        it += sizeof(header);
        std::string key{header.name, header.namelen};
        index_switch<std::tuple_size_v<AttrTypesTuple>>((size_t)header.type, [&] (auto type) {
            using T = std::tuple_element_t<type.value, AttrTypesTuple>;
            auto &attr = arr.template add_attr<T>(key);
            attr.clear();
            attr.reserve(header.size);
            std::copy_n((T const *)it, header.size, std::back_inserter(attr));
            it += sizeof(T) * header.size;
        });
    }
    arr.update();
}

template <class T0, class It>
void encodeAttrVector(AttrVector<T0> const &arr, It &it) {
    AttrVectorHeader header;
    header.size = arr.size();
    header.nattrs = arr.num_attrs();
    it = std::copy_n((char *)&header, sizeof(header), it);
    it = std::copy_n((char *)arr.data(), sizeof(T0) * arr.size(), it);

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
        it = std::copy_n((char *)attr.data(), sizeof(T) * attr.size(), it);
    });
}

}

std::shared_ptr<PrimitiveObject> decodePrimitiveObject(const char *it);
std::shared_ptr<PrimitiveObject> decodePrimitiveObject(const char *it) {
    auto obj = std::make_shared<PrimitiveObject>();
    decodeAttrVector(obj->verts, it);
    decodeAttrVector(obj->points, it);
    decodeAttrVector(obj->lines, it);
    decodeAttrVector(obj->tris, it);
    decodeAttrVector(obj->quads, it);
    decodeAttrVector(obj->loops, it);
    decodeAttrVector(obj->polys, it);
    if (*it++ == '1') {
        obj->mtl = std::make_shared<MaterialObject>();
        obj->mtl->deserialize(it);
    }
    return obj;
}

bool encodePrimitiveObject(PrimitiveObject const *obj, std::back_insert_iterator<std::vector<char>> it);
bool encodePrimitiveObject(PrimitiveObject const *obj, std::back_insert_iterator<std::vector<char>> it) {
    encodeAttrVector(obj->verts, it);
    encodeAttrVector(obj->points, it);
    encodeAttrVector(obj->lines, it);
    encodeAttrVector(obj->tris, it);
    encodeAttrVector(obj->quads, it);
    encodeAttrVector(obj->loops, it);
    encodeAttrVector(obj->polys, it);
    if (obj->mtl) {
        *it++ = '1';
        for (char c: obj->mtl->serialize())
            *it++ = c;
    } else {
        *it++ = '0';
    }
    return true;
}

}

}

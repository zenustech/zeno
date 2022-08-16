#include <zeno/funcs/ObjectCodec.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/MaterialObject.h>
#include <zeno/utils/variantswitch.h>
#include <zeno/utils/log.h>
//#include <zeno/utils/zeno_p.h>
#include <algorithm>
#include <cstring>
namespace zeno {

namespace _implObjectCodec {

namespace {

struct AttributeHeader {
    size_t type;
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
        AttributeHeader h;
        std::copy_n(it, sizeof(h), (char *)&h);
        it += sizeof(h);
        std::string key{h.name, h.namelen};
        index_switch<std::variant_size_v<AttrAcceptAll>>((size_t)h.type, [&] (auto type) {
            using T = std::variant_alternative_t<type.value, AttrAcceptAll>;
            auto &attr = arr.template add_attr<T>(key);
            attr.clear();
            attr.reserve(h.size);
            std::copy_n((T const *)it, h.size, std::back_inserter(attr));
            it += sizeof(T) * h.size;
        });
    }
    arr.update();
}

template <class T0, class It>
void encodeAttrVector(AttrVector<T0> const &arr, It &it) {
    AttrVectorHeader header;
    header.size = arr.size();
    header.nattrs = arr.template num_attrs<AttrAcceptAll>();
    it = std::copy_n((char const *)&header, sizeof(header), it);
    it = std::copy_n((char const *)arr.data(), sizeof(T0) * arr.size(), it);

    arr.template foreach_attr<AttrAcceptAll>([&] (auto const &key, auto const &attr) {
        AttributeHeader h;
        using T = std::decay_t<decltype(attr[0])>;
        h.type = variant_index<AttrAcceptAll, T>::value;
        h.size = attr.size();
        h.namelen = key.size();
        std::strncpy(h.name, key.c_str(), sizeof(h.name));
        it = std::copy_n((char const *)&h, sizeof(h), it);
        it = std::copy_n((char const *)attr.data(), sizeof(T) * attr.size(), it);
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
    decodeAttrVector(obj->edges, it);
    decodeAttrVector(obj->uvs, it);
    decodeAttrVector(obj->loop_uvs, it);
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
    encodeAttrVector(obj->edges, it);
    encodeAttrVector(obj->uvs, it);
    encodeAttrVector(obj->loop_uvs, it);
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

#include <zeno/types/MaterialObject.h>
#include <zeno/types/DummyObject.h>
#include <zeno/funcs/ObjectCodec.h>
#include <zeno/utils/log.h>
#include <algorithm>
#include <cstring>

namespace zeno {

namespace _implObjectCodec {

std::shared_ptr<MaterialObject> decodeMaterialObject(const char *it);
std::shared_ptr<MaterialObject> decodeMaterialObject(const char *it) {
    auto mtl = std::make_shared<MaterialObject>();
    mtl->deserialize(it);
    return mtl;
}

bool encodeMaterialObject(MaterialObject const *obj, std::back_insert_iterator<std::vector<char>> it);
bool encodeMaterialObject(MaterialObject const *obj, std::back_insert_iterator<std::vector<char>> it) {
    auto v = obj->serialize();
    std::copy(v.begin(), v.end(), it);
    return true;
}

std::shared_ptr<DummyObject> decodeDummyObject(const char *it);
std::shared_ptr<DummyObject> decodeDummyObject(const char *it) {
    return std::make_shared<DummyObject>();
}

bool encodeDummyObject(DummyObject const *obj, std::back_insert_iterator<std::vector<char>> it);
bool encodeDummyObject(DummyObject const *obj, std::back_insert_iterator<std::vector<char>> it) {
    return true;
}

}

}

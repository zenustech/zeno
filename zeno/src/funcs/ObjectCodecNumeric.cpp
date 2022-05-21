#include <zeno/types/NumericObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/funcs/ObjectCodec.h>
#include <zeno/utils/variantswitch.h>
#include <zeno/utils/log.h>
#include <algorithm>
#include <cstring>
namespace zeno {

namespace _implObjectCodec {

std::shared_ptr<NumericObject> decodeNumericObject(const char *it);
std::shared_ptr<NumericObject> decodeNumericObject(const char *it) {
    auto obj = std::make_shared<NumericObject>();
    size_t index = *(size_t *)it;
    it += sizeof(index);
    auto succ = index_switch<std::variant_size_v<NumericValue>, true>(index, [&] (auto idx) {
        if constexpr (std::is_same_v<decltype(idx), index_variant_monostate>) {
            return false;
        } else {
            using T = std::variant_alternative_t<idx.value, NumericValue>;
            obj->value = *(T *)it;
            it += sizeof(T);
            return true;
        }
    });
    return succ ? obj : nullptr;
}

bool encodeNumericObject(NumericObject const *obj, std::back_insert_iterator<std::vector<char>> it);
bool encodeNumericObject(NumericObject const *obj, std::back_insert_iterator<std::vector<char>> it) {
    size_t index = obj->value.index();
    it = std::copy_n((char const *)&index, sizeof(index), it);
    std::visit([&] (auto const &val) {
        using T = std::decay_t<decltype(val)>;
        it = std::copy_n((char const *)&val, sizeof(val), it);
    }, obj->value);
    return true;
}

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

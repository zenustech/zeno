#include <zeno/types/ListObject.h>
#include <zeno/funcs/ObjectCodec.h>
#include <zeno/utils/log.h>
#include <algorithm>
#include <cstring>

namespace zeno {

namespace _implObjectCodec {

std::shared_ptr<ListObject> decodeListObject(const char *it);
std::shared_ptr<ListObject> decodeListObject(const char *it) {
    auto obj = std::make_shared<ListObject>();

    size_t size = *(int *)it;
    it += sizeof(size);

    std::vector<size_t> tab(size * 2);
    std::memcpy(tab.data(), it, sizeof(size_t) * tab.size()); 
    it += sizeof(size_t) * tab.size();

    obj->arr.resize(size);
    for (size_t i = 0; i < size; i++) {
        auto elm = decodeObject(it + tab[i * 2], tab[i * 2 + 1]);
        if (!elm) return nullptr;
        obj->arr[i] = std::move(elm);
    }

    return obj;
}

bool encodeListObject(ListObject const *obj, std::back_insert_iterator<std::vector<char>> it);
bool encodeListObject(ListObject const *obj, std::back_insert_iterator<std::vector<char>> it) {
    size_t size = obj->arr.size();
    std::copy_n((char const *)&size, sizeof(size), it);

    std::vector<char> buf;
    std::vector<char> fin;
    std::vector<size_t> tab(size * 2);
    size_t base = 0;
    for (size_t i = 0; i < size; i++) {
        auto const *elm = obj->arr[i].get();
        if (!encodeObject(elm, buf))
            return false;
        size_t len = buf.size();
        fin.insert(fin.end(), buf.begin(), buf.end());
        buf.clear();
        tab[i * 2] = base;
        tab[i * 2 + 1] = len;
        base += len;
    }
    std::copy_n(tab.data(), tab.size() * sizeof(size_t), it);
    std::copy(fin.begin(), fin.end(), it);

    return true;
}

// TODO: support DictObject

}

}

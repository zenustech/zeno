#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/utils/string.h>
#include <zeno/utils/fileio.h>
#include <zeno/utils/vec.h>
#include <string_view>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <cassert>
#include <cstdio>
#include <fstream>

namespace zeno {
namespace {

template <std::size_t ...Is>
static bool match_helper(char const *&it, char const *arr, std::index_sequence<Is...>) {
    if (((it[Is] == arr[Is]) && ...)) {
        it += sizeof...(Is);
        return true;
    } else {
        return false;
    }
}

template <std::size_t N>
static bool match(char const *&it, char const (&arr)[N]) {
    return match_helper(it, arr, std::make_index_sequence<N>{});
}

void parse_obj(std::vector<char> &&bin) {
    bin.resize(bin.size() + 8, '\n');

    char const *it = bin.data();
    char const *eit = bin.data() + bin.size();

    if (match(it, "o ")) {
        it += 2;
        std::string_view o_name(it, std::find(it, eit, '\n') - it);
    }
}

struct PrimitiveReadObj : INode {
    virtual void apply() override {
        auto path = get_input<StringObject>("path")->get();
        auto binary = zeno::file_get_binary(path);
        parse_obj(std::move(binary));
    }
};

}
}

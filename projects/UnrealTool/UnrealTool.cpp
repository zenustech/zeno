#include "zeno/unreal/UnrealTool.h"
#include "zeno/unreal/HashMap.h"
#include "zeno/unreal/Pair.h"
#include "zeno/unreal/SimpleList.h"
#include <cstdarg>
#include <map>
#include <memory.h>
#include <zeno/core/Graph.h>

template<typename K, typename V>
zeno::HashMap<K, V> CreateHashMapFromStd(const std::map<K, V>& InMap) {
    zeno::HashMap<K, V> Map;
    for (const std::pair<K, V>& pair : InMap) {
        Map.put(pair.first, pair.second);
    }
    return Map;
}

template<typename K, typename V>
std::map<K, V> ConvertHashMapToStd(const zeno::HashMap<K, V>& Map, const zeno::SimpleList<K>& KeyList) {
    std::map<K, V> result;
    for (const K& key : KeyList) {
        V value;
        Map.get(key, value);
        result.insert(std::make_pair(key, value));
    }
    return result;
}

std::string zeno::CopyToString(const char* const data) {
    return std::string { data };
}

char* zeno::CopyToChar(const std::string& str) {
    char* data = static_cast<char*>(malloc(str.size()));
    memcpy(data, str.c_str(), str.size());
    return data;
}

// TODO [darc] : fix memory leak caused by char buffer :
zeno::SimpleList<std::pair<zeno::SimpleCharBuffer, zeno::zany>> zeno::CallTempNode(const std::shared_ptr<Graph>& graph, const char *id, size_t argc, ...) {
    typedef std::pair<const char*, zeno::zany> ZPair ;

    std::map<std::string, zeno::zany> inputs;

    va_list vl;
    va_start(vl, argc);
    for (size_t i = 0; i < argc; ++i) {
        const ZPair& pair = va_arg(vl, ZPair);
        inputs.insert(std::make_pair(std::string(pair.first), pair.second));
    }
    va_end(vl);


    std::map<std::string, std::shared_ptr<zeno::IObject>> output = graph->callTempNode(id, inputs);
    zeno::SimpleList<std::pair<zeno::SimpleCharBuffer, zeno::zany>> result(output.size());
    for (const auto& pair : output) {
        result.add(std::make_pair(zeno::SimpleCharBuffer{ pair.first.c_str(), pair.first.size() }, pair.second));
    }

    return result;
}

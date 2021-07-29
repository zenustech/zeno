#ifdef ZENO_EMBED
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <zeno/zeno.h>

namespace zeno {

using namespace rapidjson;

static zeno::IValue generic_get(Value const &x) {
    if (x.IsString()) {
        return x.GetString();
    } else if (x.IsInt()) {
        return x.GetInt();
    } else if (x.IsFloat()) {
        return x.GetFloat();
    } else {
        return 0;
    }
}

void loadSceneFromList(const char *json) {
    Document d;
    d.Parse(json);

    for (int i = 0; i < d.Size(); i++) {
        Value const &di = d[i];
        std::string cmd = di[0].GetString();
        if (0) {
        } else if (cmd == "addNode") {
            addNode(di[1].GetString(), di[2].GetString());
        } else if (cmd == "completeNode") {
            completeNode(di[1].GetString());
        } else if (cmd == "setNodeParam") {
            setNodeParam(di[1].GetString(), di[2].GetString(), generic_get(di[3]));
        } else if (cmd == "setNodeOption") {
            setNodeOption(di[1].GetString(), di[2].GetString());
        } else if (cmd == "bindNodeInput") {
            bindNodeInput(di[1].GetString(), di[2].GetString(), di[3].GetString(), di[4].GetString());
        } else if (cmd == "switchGraph") {
            switchGraph(di[1].GetString());
        } else if (cmd == "clearAllState") {
            clearAllState();
        }
    }
}

}
#endif

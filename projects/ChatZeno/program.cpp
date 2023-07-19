#include <zeno/zeno.h>
#include <zeno/types/DictObject.h>
#include <zeno/extra/assetDir.h>
#include <zeno/extra/EventCallbacks.h>
#include <zeno/utils/log.h>
#include <zeno/types/GenericObject.h>
#include <zeno/types/FunctionObject.h>
#include <zeno/types/UserData.h>
#include <zeno/core/Graph.h>
#include <zeno/utils/zeno_p.h>
#include <zeno/utils/string.h>
#include <zeno/utils/scope_exit.h>
#include <iostream>

namespace zeno {

static int subprogram_dumpdesc_main(int argc, char **argv) {
    std::string str = zeno::getSession().dumpDescriptorsJSON();
    std::cout << str << std::endl;
    return 0;
}

static int defChatZenoInit = getSession().eventCallbacks->hookEvent("init", [] {
    getSession().userData().set("subprogram_dumpdesc", std::make_shared<GenericObject<int(*)(int, char **)>>(subprogram_dumpdesc_main));
});

}

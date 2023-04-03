#include <zeno/core/Session.h>
#include <zeno/extra/EventCallbacks.h>
#include "unrealtcpserver.h"
#include "unrealudpserver.h"

namespace zeno {

static int defUnrealBridgeInit = getSession().eventCallbacks->hookEvent("init", [] {
    zeno::startUnrealTcpServer();
    zeno::startUnrealUdpServer();
});

}

#include "unrealtcpserver.h"
#include "unrealudpserver.h"
#include "ubipcclient.h"
#include <QTcpServer>
#include <zeno/core/Session.h>
#include <zeno/extra/EventCallbacks.h>

namespace zeno {

static int defUnrealBridgeInit = getSession().eventCallbacks->hookEvent("init", [] {
    {
        QTcpServer tmpServer;
        if (!tmpServer.listen(QHostAddress{"127.0.0.1"}, 23343)) {
            // already has a server
            // let ipc client connect
            ::UnrealBridge::IPCClient::getStatic().setup();
        } else {
            tmpServer.close();
            zeno::startUnrealTcpServer();
            zeno::startUnrealUdpServer();
            ::UnrealBridge::IPCServer::getStatic().setup();
        }
    }
});

}

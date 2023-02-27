#ifndef ZENO_UNREALMISC_H
#define ZENO_UNREALMISC_H

#include "../model/packethandler.h"

REG_PACKET_HANDLER(SendAuthToken, ZBTControlPacketType::SendAuthToken, {
    bHasRespond = false;
});

#endif //ZENO_UNREALMISC_H

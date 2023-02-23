#ifndef ZENO_UNREALMISC_H
#define ZENO_UNREALMISC_H

#include "../model/packethandler.h"

REG_PACKET_HANDLER(AuthRequire, ZBTControlPacketType::AuthRequire, {
      outSize = 1;
      outBuffer = { 0x11 };
      outPacketType = ZBTControlPacketType::AuthRequire;
      bHasRespond = true;
});

#endif //ZENO_UNREALMISC_H

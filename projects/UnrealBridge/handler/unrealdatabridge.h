#ifndef ZENO_UNREALDATABRIDGE_H
#define ZENO_UNREALDATABRIDGE_H

#include "../model/packethandler.h"

REG_PACKET_HANDLER(SendHeightField, ZBTControlPacketType::SendHeightField, {
   // TODO: darc receive height field from unreal
});

#endif //ZENO_UNREALDATABRIDGE_H

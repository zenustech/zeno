#ifndef ZENO_UNREALMISC_H
#define ZENO_UNREALMISC_H

#include "../model/packethandler.h"
#include "../unrealregistry.h"
#include "include/msgpack.h"
#include "unrealudpserver.h"
#include <QNetworkDatagram>

#ifndef ZENO_BRIDGE_TOKEN
#define ZENO_BRIDGE_TOKEN "123456"
#endif // ZENO_BRIDGE_TOKEN

REG_PACKET_HANDLER(SendAuthToken, ZBTControlPacketType::SendAuthToken, {
    bHasRespond = true;

    std::error_code err;
    const auto data = msgpack::unpack<ZPKSendToken>(reinterpret_cast<const uint8_t*>(inData) + sizeof(ZBTPacketHeader), inSize, err);

    if (!err && data.token == ZENO_BRIDGE_TOKEN) {
        std::string newSessionName = UnrealSessionRegistry::getStatic().newSession();
        outPacketType = ZBTControlPacketType::RegisterSession;

        ZPKRegisterSession sessionPacket { std::move(newSessionName) };
        outBuffer = msgpack::pack(sessionPacket);
        outSize = outBuffer->size();
    } else {
        outPacketType = ZBTControlPacketType::AuthFailed;
        outSize = 0;
    }
});

REG_PACKET_HANDLER(BindUdpToSession, ZBTControlPacketType::BindUdpToSession, {
    std::error_code err;
    const auto data = msgpack::unpack<ZPKBindUdpToSession>(reinterpret_cast<const uint8_t*>(inData) + sizeof(ZBTPacketHeader), inSize, err);

    if (!err) {
        UnrealSessionRegistry::getStatic().updateSession(data.sessionName, {data.address, data.port});
        // send subjects
        UnrealSubjectRegistry::getStatic().markDirty(true);
    }
});

REG_PACKET_HANDLER(RemoveSession, ZBTControlPacketType::RemoveSession, {
    std::error_code err;
    const auto data = msgpack::unpack<ZPKRegisterSession>(reinterpret_cast<const uint8_t*>(inData) + sizeof(ZBTPacketHeader), inSize, err);

    if (!err) {
        UnrealSessionRegistry::getStatic().removeSession(data.sessionName);
    }
});

#endif //ZENO_UNREALMISC_H

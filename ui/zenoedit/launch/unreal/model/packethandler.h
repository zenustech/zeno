#ifndef ZENO_PACKETHANDLER_H
#define ZENO_PACKETHANDLER_H

#include <optional>
#include <map>
#include <vector>
#include "networktypes.h"

typedef std::optional<std::vector<uint8_t>> OutPacketBufferType;
typedef void(*PacketHandler)(void*, bool&, ZBTControlPacketType&, OutPacketBufferType&, uint16_t&);

template <
    ZBTControlPacketType InPacketType
>
struct PacketHandlerAnnotation {
    static PacketHandler handler;
};

struct PacketHandlerMap {
    static PacketHandlerMap& get() {
        static PacketHandlerMap sMap;
        return sMap;
    }

    void addHandler(const ZBTControlPacketType value, PacketHandler handler) {
        handlerMap.insert(std::make_pair(value, handler));
    }

    void tryCall(
        const ZBTControlPacketType value,
        void* inPacket,
        bool& outHasRespond,
        ZBTControlPacketType& outPacketType,
        OutPacketBufferType& outRespondData,
        uint16_t& outDataSize
    ) {
        auto it = handlerMap.find(value);
        if (it != handlerMap.end()) {
            PacketHandler handler = it->second;
            handler(inPacket, outHasRespond, outPacketType, outRespondData, outDataSize);
        }
    }

private:
    std::map<ZBTControlPacketType, PacketHandler> handlerMap;
};


#define REG_PACKET_HANDLER(Name, PacketType, Block) \
    template<> PacketHandler PacketHandlerAnnotation<PacketType>::handler = [] (void* inData, bool& bHasRespond, ZBTControlPacketType& outPacketType, OutPacketBufferType& outBuffer, uint16_t& outSize) Block;\
    static struct THIS_IS_NOT_START_WITH_StaticInitFor##Name {     \
        THIS_IS_NOT_START_WITH_StaticInitFor##Name() {             \
            PacketHandlerMap::get().addHandler(PacketType, PacketHandlerAnnotation<PacketType>::handler); \
        }                                    \
    } THIS_IS_NOT_START_WITH__StaticInitFor##Name;

#endif //ZENO_PACKETHANDLER_H

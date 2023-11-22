#pragma once

#include "rpc/pch.h"
#include <event/event_bus.pb.h>
#include <event/event_bus.grpc.pb.h>

#include <memory>

#include "zeno/types/PrimitiveObject.h"


class EventBusService final : public zeno::event::EventBus::Service {
public:
    ::grpc::Status TriggerEvent(::grpc::ServerContext *context, const ::zeno::event::TriggerEventQuery *request, ::zeno::event::TriggerEventResponse *response) override;
    ::grpc::Status PushPrimitiveNotify(::grpc::ServerContext *context, const ::zeno::event::PutPrimitiveObjectQuery *request, ::zeno::event::PutPrimitiveObjectResponse *response) override;
};


std::shared_ptr<zeno::PrimitiveObject> FromProtobuf(const zeno::event::PrimitiveObject* Object);

struct NamedPrimitiveObject {
    std::string Channel;
    std::string Name;
    std::shared_ptr<zeno::PrimitiveObject> Primitive;
};

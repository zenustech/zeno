#pragma once

#include "pch.h"
#include <event/event_bus.pb.h>
#include <event/event_bus.grpc.pb.h>

class EventBusService final : public zeno::event::EventBus::Service {
public:
    ::grpc::Status TriggerEvent(::grpc::ServerContext *context, const ::zeno::event::TriggerEventQuery *request, ::zeno::event::TriggerEventResponse *response) override;
};


#include "event.h"
#include <zeno/zeno.h>
#include <zeno/extra/EventCallbacks.h>

::grpc::Status EventBusService::TriggerEvent(::grpc::ServerContext *context, const ::zeno::event::TriggerEventQuery *request, ::zeno::event::TriggerEventResponse *response) {
    std::string requestEventName = zeno::event::EventType_Name(request->eventtype());

    zeno::getSession().eventCallbacks->triggerEvent(requestEventName);

    response->set_status(zeno::common::SUCCESS);

    return grpc::Status::OK;
}

::grpc::Status EventBusService::PutPrimitiveToMemoryCache(::grpc::ServerContext *context, const ::zeno::event::PutPrimitiveObjectQuery *request, ::zeno::event::PutPrimitiveObjectResponse *response) {
    // TODO [darc] : impl :
    return grpc::Status::CANCELLED;
}

namespace {
    [[maybe_unused]] StaticServiceRegister<EventBusService> AutoRegisterForEventBusService {};
}

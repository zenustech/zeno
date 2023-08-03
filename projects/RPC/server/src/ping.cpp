#include "ping.h"

#include <iostream>
#include "pch.h"

::grpc::Status PingPongService::Ping(::grpc::ServerContext *context, const ::zeno::common::PingQuery *request, ::zeno::common::PongResponse *response) {
    std::cout << "Ping!" << request->hashtoken() << std::endl;

    response->set_status(zeno::common::StatusCode::SUCCESS);

    return grpc::Status::OK;
}

namespace {
    [[maybe_unused]] StaticServiceRegister<PingPongService> AutoRegisterForPingPongService {};
}

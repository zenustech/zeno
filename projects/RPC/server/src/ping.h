#pragma once

#include "common/ping.pb.h"
#include "common/ping.grpc.pb.h"

class PingPongService final : public zeno::common::HealthCheck::Service {
public:
    virtual ::grpc::Status Ping(::grpc::ServerContext *context, const ::zeno::common::PingQuery *request, ::zeno::common::PongResponse *response) override;
};

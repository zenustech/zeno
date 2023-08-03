
#include <common/ping.pb.h>
#include <common/ping.grpc.pb.h>

#include <grpcpp/grpcpp.h>
#include <grpcpp/create_channel.h>

int main(int Argc, char* Argv[]) {
    zeno::common::PingQuery Query;
    zeno::common::PongResponse Response;
    Query.set_hashtoken("123");

    auto Channel = grpc::CreateChannel("localhost:25561", grpc::InsecureChannelCredentials());
    std::unique_ptr<zeno::common::HealthCheck::Stub> Stub = zeno::common::HealthCheck::NewStub(Channel);
    grpc::ClientContext Context;
    grpc::Status Status = Stub->Ping(&Context, Query, &Response);

    if (!Status.ok()) {
        std::cout << "Failed to make rpc call." << std::endl;
        std::cout << Status.error_message() << std::endl;
        return 1;
    }

    std::cout << "Status: " << Response.status() << std::endl;

    return 0;
}

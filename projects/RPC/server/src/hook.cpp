
#include <zeno/zeno.h>
#include <zeno/extra/EventCallbacks.h>
#include <grpcpp/server_builder.h>
#include <thread>
#include <optional>
#include <chrono>

namespace {
    std::optional<std::thread> ServerThreadObj;

    std::unique_ptr<grpc::Server> RPCServer;

    void StartRPCServer() {
        grpc::ServerBuilder Builder;
        Builder.AddListeningPort("0.0.0.0:25561", grpc::InsecureServerCredentials());

        RPCServer = Builder.BuildAndStart();
        RPCServer->Wait();
    }

    [[maybe_unused]] int defRPCInit =
        zeno::getSession().eventCallbacks->hookEvent("init", [] {
            ServerThreadObj = std::thread { StartRPCServer };
        });

    [[maybe_unused]] int defRPCRunnerInit =
        zeno::getSession().eventCallbacks->hookEvent("preRunnerStart", [] {
        });

    [[maybe_unused]] int defRPCDestroy =
        zeno::getSession().eventCallbacks->hookEvent("beginDestroy", [] {
            RPCServer->Shutdown();
            // Waiting for server shutdown
            ServerThreadObj->join();
        });
}

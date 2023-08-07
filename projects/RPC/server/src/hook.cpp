
#include "rpc/pch.h"
#include <chrono>
#include <grpcpp/server_builder.h>
#include <optional>
#include <thread>
#include <zeno/extra/EventCallbacks.h>
#include <zeno/zeno.h>

namespace {
    std::optional<std::thread> ServerThreadObj;

    std::unique_ptr<grpc::Server> RPCServer;

    void StartRPCServer(std::vector<std::shared_ptr<grpc::Service>>* Services) {
        grpc::ServerBuilder Builder;
        Builder.AddListeningPort("0.0.0.0:25561", grpc::InsecureServerCredentials());
        // Builder.SetSyncServerOption(grpc::ServerBuilder::SyncServerOption::NUM_CQS, 4);

        if (Services->empty()) {
            std::cout << "[RPC] No service was found." << std::endl;
            return;
        }

        for (const std::shared_ptr<grpc::Service>& Service : *Services) {
            if (Service) {
                Builder.RegisterService(Service.get());
            }
        }

        // std::unique_ptr<grpc::ServerCompletionQueue> CompletionQueue = Builder.AddCompletionQueue(true);

        RPCServer = Builder.BuildAndStart();
        RPCServer->Wait();
        // CompletionQueue->Shutdown();
    }

    [[maybe_unused]] int defRPCInit =
        zeno::getSession().eventCallbacks->hookEvent("editorConstructed", [] (auto _) {
            std::cout << "Starting RPC Server..." << std::endl;
            ServerThreadObj = std::thread { StartRPCServer, &GetRPCServiceList() };
        });

    [[maybe_unused]] int defRPCRunnerInit =
        zeno::getSession().eventCallbacks->hookEvent("preRunnerStart", [] (auto _) {
        });

    [[maybe_unused]] int defRPCDestroy =
        zeno::getSession().eventCallbacks->hookEvent("beginDestroy", [] (auto _) {
            RPCServer->Shutdown();
            // Waiting for server shutdown
            ServerThreadObj->join();
        });
}

std::vector<std::shared_ptr<grpc::Service>>& GetRPCServiceList() {
    static std::vector<std::shared_ptr<grpc::Service>> RPCServices {};
    return RPCServices;
}

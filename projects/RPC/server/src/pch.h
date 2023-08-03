#pragma once

#include <vector>
#include <memory>
#include <grpcpp/grpcpp.h>

extern std::vector<std::shared_ptr<grpc::Service>> RPCServices;

/**
 * Example:
 *
 * ```
 * static StaticRegister<FooService> _Reg(...Construct args...);
 * ```
 * @tparam ServiceType
 * @tparam Args
 */
template <typename ServiceType, typename... Args>
struct StaticServiceRegister {
    explicit StaticServiceRegister(Args... args) {
        static_assert(std::is_base_of_v<grpc::Service, ServiceType>);

        auto PTR = std::make_shared<ServiceType>(std::forward<Args>(args)...);
        RPCServices.push_back(PTR);
    }
};

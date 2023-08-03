#pragma once

#include <vector>
#include <memory>
#include <grpcpp/grpcpp.h>

std::vector<std::shared_ptr<grpc::Service>>& GetRPCServiceList();

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
    explicit StaticServiceRegister(Args... args);
};

template<typename ServiceType, typename... Args>
inline StaticServiceRegister<ServiceType, Args...>::StaticServiceRegister(Args... args) {
    static_assert(std::is_base_of_v<grpc::Service, ServiceType>);

    std::cout << "Registering service: " << typeid(ServiceType).name() << std::endl;

    auto PTR = std::make_shared<ServiceType>(std::forward<Args>(args)...);
    GetRPCServiceList().push_back(PTR);
}

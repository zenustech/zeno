
# Requirements

1. Ensure you have gRPC and Protobuf installed. If not, you can install them on Windows using Vcpkg as follows:

```bash
vcpkg install grpc:x64-windows protobuf:x64-windows
```

Make sure your Vcpkg is integrated correctly with your environment. After you installed the libraries, all the files should be ready for using within your CLion.

# How to use

First of all, cmake flag `ZENO_WITH_RPC` must be **ON** (by adding `-DZENO_WITH_RPC=ON`) to use RPC framework.
Check this in `CMakeLists.txt` if your want to use RPC framework.

```cmake
if (NOT ZENO_WITH_RPC)
    message(FATAL_ERROR "You must enable RPC to use my code.")
endif ()
```

To add a library dependency to your target you should modify your CMakeLists.txt in your project folder. 
Suppose your target is named `YourTargetName`,

```cmake
target_link_libraries(YourTargetName RPCProto)
```

You can add custom `proto` file by adding:

```cmake
target_sources(RPCProto ${YOUR_PROTO_FILES})
```

Then add a custom service to server using:

```c++
#include "rpc/pch.h"

// Must be public inherited class
class YourService final : public your::proto::generated::ServiceClass {
public:
    // Fill your implementation 
};

// Using anonymous namespace to prevent pollute the global namespace
namespace {
    [[maybe_unused]] StaticServiceRegister<YourService> AutoRegisterForYourService {};
}
```

**You must register your service before `editorConstructed` event (after ZenoMainWindow been constructed.)**

# Example

See `projects/Roads`.

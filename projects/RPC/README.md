
# Requirements

1. Ensure you have gRPC and Protobuf installed. If not, you can install them on Windows using Vcpkg as follows:

```bash
vcpkg install grpc:x64-windows protobuf:x64-windows
```

Make sure your Vcpkg is integrated correctly with your environment. After you installed the libraries, all the files should be ready for using within your CLion.

# How to use

To add a library dependency to your target you should modify your CMakeLists.txt in your project folder. 
Suppose your target is named `YourTargetName`,

```cmake
target_link_libraries(YourTargetName RPCProto)
```

You can add custom `proto` file by adding:

```cmake
target_sources(RPCProto ${YOUR_PROTO_FILES})
```

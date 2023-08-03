
# Requirements

gRPC and Protobuf.

For Windows:
```bash
vcpkg install grpc:x64-windows protobuf:x64-windows
```

# How to use

## Programming

1. Add library dependency named `RPCProto` to your target.
2. Build `RPCProto` to generate required header files.
3. Everything is ready to go.

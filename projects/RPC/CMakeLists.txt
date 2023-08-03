CMAKE_MINIMUM_REQUIRED(VERSION 3.14)

# Fix for windows.h and winsock2.h conflict
target_compile_definitions(zeno PUBLIC -DWIN32_LEAN_AND_MEAN)

# Protobuf definition
ADD_SUBDIRECTORY(proto)
# gRPC service server
ADD_SUBDIRECTORY(server)
# gRPC client wrapper
ADD_SUBDIRECTORY(client)

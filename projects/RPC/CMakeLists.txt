CMAKE_MINIMUM_REQUIRED(VERSION 3.14)

# Protobuf definition
ADD_SUBDIRECTORY(proto)
# gRPC service server
ADD_SUBDIRECTORY(server)
# gRPC client wrapper
ADD_SUBDIRECTORY(client)

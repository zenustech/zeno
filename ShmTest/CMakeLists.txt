cmake_minimum_required(VERSION 3.18)
set(CMAKE_CXX_STANDARD 20)
project(shmtest)

include_directories(../include)
add_executable(client client.cpp)
add_executable(server server.cpp)

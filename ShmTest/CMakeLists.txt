cmake_minimum_required(VERSION 3.18)
set(CMAKE_CXX_STANDARD 20)
project(shmtest)

add_executable(client client.cpp)
add_executable(server server.cpp)

cmake_minimum_required(VERSION 3.10)
project(ZFX LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)

add_executable(ZFX main.cpp Tokenizer.cpp)
target_include_directories(ZFX PRIVATE .)

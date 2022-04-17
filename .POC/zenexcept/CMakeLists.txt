cmake_minimum_required(VERSION 3.10)
project(zenexcept LANGUAGES C CXX)

file(GLOB PROJECT_SOURCE *.cpp *.h)
add_executable(zenexcept ${PROJECT_SOURCE})

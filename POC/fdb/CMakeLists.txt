set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

add_executable(adflip adgrid.cpp timer.cpp)

find_package(OpenMP REQUIRED)
target_link_libraries(adflip OpenMP::OpenMP_CXX)

add_library(zen SHARED libzen.cpp include/zen/zen.h)
target_include_directories(zen PRIVATE ../include)
target_include_directories(zen PUBLIC include)

add_library(zenpy SHARED python/libzenpy.cpp)
target_link_libraries(zenpy PRIVATE zen)

include(../cmake/ZenPybind11.cmake)
do_python_bindings()

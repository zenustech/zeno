file(GLOB PROJECT_SOURCE include/*/*.h *.cpp view/*.cpp mesh/*.cpp pars/*.cpp prim/*.cpp) 

add_library(zen SHARED ${PROJECT_SOURCE})
target_include_directories(zen PRIVATE ../include)
target_include_directories(zen PUBLIC include)

find_package(OpenMP)
if (TARGET OpenMP::OpenMP_CXX)
    message("found package: OpenMP::OpenMP_CXX")
    target_link_libraries(zen PRIVATE OpenMP::OpenMP_CXX)
endif()

add_library(zenpy SHARED python/libzenpy.cpp)
target_link_libraries(zenpy PRIVATE zen)

include(../cmake/ZenPybind11.cmake)
do_python_bindings()

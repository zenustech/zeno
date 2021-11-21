
add_library(SyclSTL policies.cpp)

# This is only supported in CMake 1.1 and above
set_property(TARGET SyclSTL PROPERTY CXX_STANDARD "11")
set_property(TARGET SyclSTL PROPERTY CXX_STANDARD_REQUIRED True)

set_property(TARGET SyclSTL APPEND PROPERTY COMPILE_FLAGS "-std=c++11")

target_link_libraries(SyclSTL ${SYCL_LIBRARY} ${OPENCL_LIBRARY})


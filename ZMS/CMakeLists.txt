file(GLOB ZMSLIB_SOURCE *.cpp *.h)

add_library(ZMSlib SHARED ${ZMSLIB_SOURCE})
target_compile_options(ZMSlib PRIVATE $<$<COMPILE_LANGUAGE:CXX>: -march=native -Wno-ignored-attributes -fopenmp>)
target_include_directories(ZMSlib PRIVATE .)

find_package(OpenMP REQUIRED)
if (TARGET OpenMP::OpenMP_CXX)
    target_link_libraries(zen PRIVATE OpenMP::OpenMP_CXX)
endif()

target_link_libraries(ZMSlib PRIVATE zen)

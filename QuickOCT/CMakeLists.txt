file(GLOB OCTLIB_SOURCE *.cpp *.h)

add_library(OCTlib SHARED ${OCTLIB_SOURCE})
target_compile_options(OCTlib PRIVATE $<$<COMPILE_LANGUAGE:CXX>: -march=native -Wno-ignored-attributes -fopenmp>)
target_include_directories(OCTlib PRIVATE .)

find_package(OpenMP REQUIRED)
if (TARGET OpenMP::OpenMP_CXX)
    target_link_libraries(zen PRIVATE OpenMP::OpenMP_CXX)
endif()

target_link_libraries(OCTlib PRIVATE zen)

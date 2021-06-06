file(GLOB OCTLIB_SOURCE *.cpp *.h)

add_library(OCTlib SHARED ${OCTLIB_SOURCE})
target_compile_options(OCTlib PRIVATE $<$<COMPILE_LANGUAGE:CXX>: -march=native -Wno-ignored-attributes>)
target_include_directories(OCTlib PRIVATE .)
set_property(TARGET OCTlib PROPERTY LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/python/zen/mods)

find_package(OpenMP)
if (TARGET OpenMP::OpenMP_CXX)
    message("found package: OpenMP::OpenMP_CXX")
    target_link_libraries(OCTlib PRIVATE OpenMP::OpenMP_CXX)
endif()

target_link_libraries(OCTlib PRIVATE zen zenbase)

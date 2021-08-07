file(GLOB ZMSLIB_SOURCE *.cpp *.h)

add_library(zeno_ZMS ${ZMSLIB_SOURCE})
#target_compile_options(zeno_ZMS PRIVATE $<$<COMPILE_LANGUAGE:CXX>: -march=native -Wno-ignored-attributes>)
target_include_directories(zeno_ZMS PRIVATE .)
target_include_directories(zeno_ZMS PRIVATE ../oldzenbase/include)

find_package(OpenMP)
if (TARGET OpenMP::OpenMP_CXX)
    message("found package: OpenMP::OpenMP_CXX")
    target_link_libraries(zeno_ZMS PRIVATE OpenMP::OpenMP_CXX)
endif()

target_link_libraries(zeno_ZMS PUBLIC zeno)

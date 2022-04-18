file(GLOB OldZMSLIB_SOURCE *.cpp *.h)

target_sources(zeno PRIVATEE ${OldZMSLIB_SOURCE})
#target_compile_options(zeno PRIVATE $<$<COMPILE_LANGUAGE:CXX>: -march=native -Wno-ignored-attributes>)
target_include_directories(zeno PRIVATE .)
target_include_directories(zeno PRIVATE ../oldzenbase/include)

find_package(OpenMP)
if (TARGET OpenMP::OpenMP_CXX)
    message("found package: OpenMP::OpenMP_CXX")
    target_link_libraries(zeno PRIVATE OpenMP::OpenMP_CXX)
endif()

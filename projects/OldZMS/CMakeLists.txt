file(GLOB OldZMSLIB_SOURCE *.cpp *.h)

add_library(zeno_OldZMS OBJECT ${OldZMSLIB_SOURCE})
#target_compile_options(zeno_OldZMS PRIVATE $<$<COMPILE_LANGUAGE:CXX>: -march=native -Wno-ignored-attributes>)
target_include_directories(zeno_OldZMS PRIVATE .)
target_include_directories(zeno_OldZMS PRIVATE ../oldzenbase/include)

find_package(OpenMP)
if (TARGET OpenMP::OpenMP_CXX)
    message("found package: OpenMP::OpenMP_CXX")
    target_link_libraries(zeno_OldZMS PRIVATE OpenMP::OpenMP_CXX)
endif()

target_link_libraries(zeno_OldZMS PUBLIC zenocore)

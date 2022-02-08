file(GLOB_RECURSE ZENO_SOURCE CONFIGURE_DEPENDS *.h *.cpp)

add_library(zeno SHARED ${ZENO_SOURCE})

if (ZENO_ENABLE_OPENMP)
    find_package(OpenMP)
    if (TARGET OpenMP::OpenMP_CXX)
        message(STATUS "Found OpenMP::OpenMP_CXX")
        target_link_libraries(zeno PRIVATE OpenMP::OpenMP_CXX)
        #target_compile_options(zeno # MSVC, GNU, Clang, Intel
        #PUBLIC   $<$<COMPILE_LANGUAGE:CXX>: $<IF:$<CXX_COMPILER_ID:MSVC>, /openmp:llvm /openmp:experimental, -fopenmp $<IF:$<CXX_COMPILER_ID:Clang>, , > >>
        #)
    else()
        message(WARNING "Not found OpenMP")
    endif()
endif()
if (ZENO_BENCHMARKING)
    target_compile_definitions(zeno PRIVATE -DZENO_BENCHMARKING)
endif()
if (ZENO_FAIL_SILENTLY)
    target_compile_definitions(zeno PRIVATE -DZENO_FAIL_SILENTLY)
endif()
if (ZENO_FAULTHANDLER)
    target_compile_definitions(zeno PRIVATE -DZENO_FAULTHANDLER)
    add_backward(zeno)
endif()

# we are not using shared libraries now, so no need for this:
#target_compile_definitions(zeno PRIVATE -DZENO_DLLEXPORT INTERFACE -DZENO_DLLIMPORT)
target_include_directories(zeno PUBLIC ..)

target_link_libraries(zeno PUBLIC spdlog zeno_external_include)

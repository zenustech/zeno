file(GLOB_RECURSE ZENO_SOURCE CONFIGURE_DEPENDS *.h *.cpp)

add_library(zeno OBJECT ${ZENO_SOURCE})

if (ZENO_ENABLE_OPENMP)
    find_package(OpenMP)
    if (TARGET OpenMP::OpenMP_CXX)
        message(STATUS "Found OpenMP::OpenMP_CXX")
        target_link_libraries(zeno PRIVATE OpenMP::OpenMP_CXX)
    else()
        message(WARNING "Not found OpenMP")
    endif()
endif()

if (ZENO_BENCHMARKING)
    target_compile_definitions(zeno PRIVATE -DZENO_BENCHMARKING)
endif()

if (ZENO_FAULTHANDLER)
    target_compile_definitions(zeno PRIVATE -DZENO_FAULTHANDLER)
    add_backward(zeno)
endif()

# we are not using shared libraries now, so no need for this:
target_compile_definitions(zeno PRIVATE -DZENO_DLLEXPORT INTERFACE -DZENO_DLLIMPORT)
target_include_directories(zeno PUBLIC ..)

target_link_libraries(zeno PUBLIC zenoinclude)
if (ZENO_ENABLE_SPDLOG)
    target_link_libraries(zeno PUBLIC spdlog)
    target_compile_definitions(zeno PUBLIC -DZENO_ENABLE_SPDLOG)
endif()
if (ZENO_ENABLE_MAGICENUM)
    target_compile_definitions(zeno PUBLIC -DZENO_ENABLE_MAGICENUM)
endif()

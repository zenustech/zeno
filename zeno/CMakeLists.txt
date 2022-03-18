file(GLOB_RECURSE ZENO_SOURCE CONFIGURE_DEPENDS *.h *.cpp)

add_library(zenocore OBJECT ${ZENO_SOURCE})

if (ZENO_ENABLE_OPENMP)
    find_package(OpenMP)
    if (TARGET OpenMP::OpenMP_CXX)
        message(STATUS "Found OpenMP::OpenMP_CXX")
        target_link_libraries(zenocore PRIVATE OpenMP::OpenMP_CXX)
    else()
        message(WARNING "Not found OpenMP")
    endif()
endif()

if (ZENO_BENCHMARKING)
    target_compile_definitions(zenocore PRIVATE -DZENO_BENCHMARKING)
endif()

if (ZENO_FAULTHANDLER)
    target_compile_definitions(zenocore PRIVATE -DZENO_FAULTHANDLER)
    add_backward(zenocore)
endif()

# we are not using shared libraries now, so no need for this:
target_compile_definitions(zenocore PRIVATE -DZENO_DLLEXPORT INTERFACE -DZENO_DLLIMPORT)
target_include_directories(zenocore PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${INSTALL_CMAKE_DIR}>)

target_link_libraries(zenocore PUBLIC zenoinclude)
if (ZENO_ENABLE_SPDLOG)
    target_link_libraries(zenocore PUBLIC spdlog)
    target_compile_definitions(zenocore PUBLIC -DZENO_ENABLE_SPDLOG)
endif()
if (ZENO_ENABLE_MAGICENUM)
    target_compile_definitions(zenocore PUBLIC -DZENO_ENABLE_MAGICENUM)
endif()

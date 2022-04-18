file(GLOB_RECURSE source CONFIGURE_DEPENDS include/*.h src/*.cpp)

add_library(zeno OBJECT ${source})

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

if (ZENO_PARALLEL_STL)
    if (MSVC)
        find_package(TBB)
        if (TBB_FOUND)
            target_link_libraries(zeno PRIVATE TBB::tbb)
        else()
            message(WARNING "TBB not found, disabling parallel STL")
            set(ZENO_PARALLEL_STL OFF)
        endif()
    endif()
    if (ZENO_PARALLEL_STL)
        target_compile_definitions(zeno PRIVATE -DZENO_PARALLEL_STL)
    endif()
endif()

if (ZENO_FAULTHANDLER)
    target_compile_definitions(zeno PRIVATE -DZENO_FAULTHANDLER)
    add_backward(zeno)
endif()

# we are not using shared libraries now, so no need for this:
#target_compile_definitions(zeno PRIVATE -DZENO_DLLEXPORT INTERFACE -DZENO_DLLIMPORT)
target_include_directories(zeno PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include/Zeno/zeno>)

if (ZENO_INSTALL_TARGET)
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include DESTINATION include/Zeno/zeno)
    install(TARGETS zeno EXPORT ZenoTargets)
endif()

target_link_libraries(zeno PUBLIC zenoinclude)

if (ZENO_ENABLE_SPDLOG)
    target_link_libraries(zeno PUBLIC spdlog)
    target_compile_definitions(zeno PUBLIC -DZENO_ENABLE_SPDLOG)
endif()

if (ZENO_ENABLE_MAGICENUM)
    target_compile_definitions(zeno PUBLIC -DZENO_ENABLE_MAGICENUM)
endif()

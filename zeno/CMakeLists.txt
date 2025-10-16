option(ZENO_BENCHMARKING "Enable ZENO benchmarking timer" ON)
option(ZENO_PARALLEL_STL "Enable parallel STL in ZENO" OFF)
option(ZENO_ENABLE_OPENMP "Enable OpenMP in ZENO for parallelism" ON)
option(ZENO_ENABLE_MAGICENUM "Enable magicenum in ZENO for enum reflection" OFF)
option(ZENO_ENABLE_BACKWARD "Enable ZENO fault handler for traceback" OFF)

file(GLOB_RECURSE source CONFIGURE_DEPENDS include/*.h src/*.cpp)

if (ZENO_BUILD_SHARED)
    add_library(zenobase SHARED ${source})
else()
    add_library(zenobase OBJECT ${source})
endif()

target_link_libraries(zenobase PUBLIC zenocore)

if(MSVC)
    target_compile_options(zenobase PRIVATE /MP)
endif()

if (ZENO_ENABLE_OPENMP)
    find_package(OpenMP)
    if (TARGET OpenMP::OpenMP_CXX)
        message(STATUS "Found OpenMP::OpenMP_CXX")
        target_link_libraries(zenobase PRIVATE OpenMP::OpenMP_CXX)
        if (MSVC)
            include(CheckCXXCompilerFlag)
            check_cxx_compiler_flag("/openmp:experimental" MSVC_SUPPORT_OPENMP_EXPERIMENTAL)
            if (MSVC_SUPPORT_OPENMP_EXPERIMENTAL)
                target_compile_options(zenobase PRIVATE $<IF:$<COMPILE_LANGUAGE:CXX>,/openmp:experimental,>)
            endif()
        endif()
    else()
        message(WARNING "Not found OpenMP")
    endif()
endif()

if (ZENO_BENCHMARKING)
    target_compile_definitions(zenobase PUBLIC -DZENO_BENCHMARKING)
endif()

# only work without CUDA option.
#if (ZENO_DEBUG_MSVC)
#    zeno_dbg_msvc(zenobase)
#endif()

if (ZENO_WITH_PYTHON)
    message(STATUS "python3 libraries for zenoedit: ${Python3_LIBRARIES}")
    message(STATUS "python3 include directories for zenoedit: ${Python3_INCLUDE_DIRS}")
    target_link_libraries(zenobase PRIVATE ${Python3_LIBRARIES})
    target_include_directories(zenobase PRIVATE ${Python3_INCLUDE_DIRS})
endif()

find_package(TBB)
message(STATUS "Found zenobase TBB: ${TBB_DIR}")
target_link_libraries(zenobase PRIVATE TBB::tbb)

if (ZENO_ENABLE_BACKWARD)
    add_subdirectory(tpls/backward-cpp)
    target_compile_definitions(zenobase PUBLIC -DZENO_ENABLE_BACKWARD)
    target_sources(zenobase PRIVATE ${BACKWARD_ENABLE})
    add_backward(zenobase)
endif()

if (ZENO_BUILD_SHARED)
    target_compile_definitions(zenobase PRIVATE -DZENO_DLLEXPORT INTERFACE -DZENO_DLLIMPORT)
endif()

target_include_directories(zenobase PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include/Zeno/zeno/include>)

if (ZENO_INSTALL_TARGET AND ZENO_BUILD_SHARED)
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include DESTINATION include/Zeno/zeno)
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tpls/include DESTINATION include/Zeno/zeno)
    install(TARGETS zenobase EXPORT ZenoTargets)
endif()

if (ZENO_ENABLE_MAGICENUM)
    target_compile_definitions(zenobase PUBLIC -DZENO_ENABLE_MAGICENUM)
endif()

if (ZENO_WITH_PYTHON)
    find_package(Python3 COMPONENTS Development REQUIRED)

    message(STATUS "python3 libraries for zenqt: ${Python3_LIBRARIES}")
    message(STATUS "python3 include directories for zenqt: ${Python3_INCLUDE_DIRS}")
    target_link_libraries(zenobase PRIVATE ${Python3_LIBRARIES} zenpybind)
    target_include_directories(zenobase PRIVATE ${Python3_INCLUDE_DIRS})

    target_compile_definitions(zenobase PRIVATE -DZENO_WITH_PYTHON)
endif()

#if (ZENO_NO_WARNING)
    #if (CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        #target_compile_options(zenobase PUBLIC $<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:CXX>:-Wno-all -Wno-cpp -Wno-deprecated-declarations -Wno-enum-compare -Wno-ignored-attributes -Wno-extra -Wreturn-type -Wmissing-declarations -Wnon-virtual-dtor -Wsuggest-override -Wconversion-null>>)
    #endif()
    #if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        #target_compile_options(zenobase PUBLIC $<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:C,CXX>:-w>>)
    #endif()
#endif()

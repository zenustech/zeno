option(ZENO_BENCHMARKING "Enable ZENO benchmarking timer" ON)
option(ZENO_PARALLEL_STL "Enable parallel STL in ZENO" OFF)
option(ZENO_ENABLE_OPENMP "Enable OpenMP in ZENO for parallelism" ON)
option(ZENO_ENABLE_MAGICENUM "Enable magicenum in ZENO for enum reflection" OFF)
option(ZENO_ENABLE_BACKWARD "Enable ZENO fault handler for traceback" OFF)

file(GLOB_RECURSE source CONFIGURE_DEPENDS include/*.h src/*.cpp)

if (ZENO_BUILD_SHARED)
    add_library(zeno SHARED ${source})
else()
    add_library(zeno OBJECT ${source})
endif()

target_link_libraries(zeno PUBLIC zenocore)

if (ZENO_ENABLE_OPENMP)
    find_package(OpenMP)
    if (TARGET OpenMP::OpenMP_CXX)
        message(STATUS "Found OpenMP::OpenMP_CXX")
        target_link_libraries(zeno PRIVATE OpenMP::OpenMP_CXX)
        if (MSVC)
            include(CheckCXXCompilerFlag)
            check_cxx_compiler_flag("/openmp:experimental" MSVC_SUPPORT_OPENMP_EXPERIMENTAL)
            if (MSVC_SUPPORT_OPENMP_EXPERIMENTAL)
                target_compile_options(zeno PRIVATE $<IF:$<COMPILE_LANGUAGE:CXX>,/openmp:experimental,>)
            endif()
        endif()
    else()
        message(WARNING "Not found OpenMP")
    endif()
endif()

if (ZENO_BENCHMARKING)
    target_compile_definitions(zeno PUBLIC -DZENO_BENCHMARKING)
endif()

# only work without CUDA option.
#if (ZENO_DEBUG_MSVC)
#    zeno_dbg_msvc(zeno)
#endif()

if (ZENO_WITH_PYTHON)
    message(STATUS "python3 libraries for zenoedit: ${Python3_LIBRARIES}")
    message(STATUS "python3 include directories for zenoedit: ${Python3_INCLUDE_DIRS}")
    target_link_libraries(zeno PRIVATE ${Python3_LIBRARIES})
    target_include_directories(zeno PRIVATE ${Python3_INCLUDE_DIRS})
endif()

if (ZENO_PARALLEL_STL)
    find_package(Threads REQUIRED)
    target_link_libraries(zeno PRIVATE Threads::Threads)
    if (NOT MSVC)
        find_package(TBB)
        if (TBB_FOUND)
            message(STATUS "Found TBB: ${TBB_DIR}")
            target_link_libraries(zeno PRIVATE TBB::tbb)
        else()
            message(WARNING "TBB not found, disabling parallel STL")
            set(ZENO_PARALLEL_STL OFF)
        endif()
    endif()
    if (ZENO_PARALLEL_STL)
        target_compile_definitions(zeno PUBLIC -DZENO_PARALLEL_STL)
    endif()
endif()

if (ZENO_ENABLE_BACKWARD)
    add_subdirectory(tpls/backward-cpp)
    target_compile_definitions(zeno PUBLIC -DZENO_ENABLE_BACKWARD)
    target_sources(zeno PRIVATE ${BACKWARD_ENABLE})
    add_backward(zeno)
endif()

if (ZENO_BUILD_SHARED)
    target_compile_definitions(zeno PRIVATE -DZENO_DLLEXPORT INTERFACE -DZENO_DLLIMPORT)
endif()

target_include_directories(zeno PUBLIC
    $<INSTALL_INTERFACE:include/Zeno/zeno/include>)

if (ZENO_INSTALL_TARGET AND ZENO_BUILD_SHARED)
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include DESTINATION include/Zeno/zeno)
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tpls/include DESTINATION include/Zeno/zeno)
    install(TARGETS zeno EXPORT ZenoTargets)
endif()

if (ZENO_ENABLE_MAGICENUM)
    target_compile_definitions(zeno PUBLIC -DZENO_ENABLE_MAGICENUM)
endif()

if (ZENO_WITH_PYTHON)
    find_package(Python3 COMPONENTS Development REQUIRED)

    message(STATUS "python3 libraries for zenqt: ${Python3_LIBRARIES}")
    message(STATUS "python3 include directories for zenqt: ${Python3_INCLUDE_DIRS}")
    target_link_libraries(zeno PRIVATE ${Python3_LIBRARIES})
    target_include_directories(zeno PRIVATE ${Python3_INCLUDE_DIRS})

    target_compile_definitions(zeno PRIVATE -DZENO_WITH_PYTHON)
endif()

if (ZENO_WITH_REFLECTOR)
    include(${CMAKE_SOURCE_DIR}/reflect/cmake/ReflectionUtils.cmake)

    set(GENERATE_ON_BUILD_DIR ON)
    set(REFLECT_TEMPLATE_INCLUDE "#include <zeno/types/ObjectDef.h>")
    set(INJA_TEMPLATE_DIR_PATH  ${CMAKE_BINARY_DIR}/intermediate)
    #头文件包含目录
    set(CUSTOM_HEADER_INCLUDE_DIR ${CMAKE_BINARY_DIR}/intermediate)

    #遍历整个目录，并收集所有标有ZRECORD或者ZPRIMITIVE的类的文件，暂时不包含project文件夹，以后可能插件化
    #如果有什么动态Object或者类型，现在这里添加搜索目录（如果没有的话）
    file(GLOB_RECURSE scan_source CONFIGURE_DEPENDS 
        ../projects/FBX/*.h
        ../projects/FBX/*.cpp
    )

    set(REFLECTE_HEADER "dummy")
    foreach(file_path ${scan_source})
        file(READ ${file_path} CURR_FILE_CONTENT)
        string(FIND "${CURR_FILE_CONTENT}" "ZPRIMITIVE" OUT_MATCH_RESULT)
        if (NOT OUT_MATCH_RESULT EQUAL -1)
            list(APPEND REFLECTE_HEADER ${file_path})
        endif()

        string(FIND "${CURR_FILE_CONTENT}" "REFLECT_REGISTER_OBJECT" OUT_MATCH_RESULT)
        if (NOT OUT_MATCH_RESULT EQUAL -1)
            list(APPEND REFLECTE_HEADER ${file_path})
        endif()
    endforeach()
    list(POP_FRONT REFLECTE_HEADER)    #把最开始的移除掉

    #送入Generator并生成
    #zeno_declare_reflection_support(zeno "${REFLECTE_HEADER}" zeno_util)
endif()

#if (ZENO_NO_WARNING)
    #if (CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        #target_compile_options(zeno PUBLIC $<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:CXX>:-Wno-all -Wno-cpp -Wno-deprecated-declarations -Wno-enum-compare -Wno-ignored-attributes -Wno-extra -Wreturn-type -Wmissing-declarations -Wnon-virtual-dtor -Wsuggest-override -Wconversion-null>>)
    #endif()
    #if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        #target_compile_options(zeno PUBLIC $<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:C,CXX>:-w>>)
    #endif()
#endif()

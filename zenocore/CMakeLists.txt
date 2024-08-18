option(ZENO_BENCHMARKING "Enable ZENO benchmarking timer" ON)
option(ZENO_PARALLEL_STL "Enable parallel STL in ZENO" OFF)
option(ZENO_ENABLE_OPENMP "Enable OpenMP in ZENO for parallelism" ON)
option(ZENO_ENABLE_MAGICENUM "Enable magicenum in ZENO for enum reflection" OFF)
option(ZENO_ENABLE_BACKWARD "Enable ZENO fault handler for traceback" OFF)

file(GLOB_RECURSE source CONFIGURE_DEPENDS include/*.h src/*.cpp)

add_library(zenocore SHARED ${source})

if (ZENO_DEBUG_MSVC)
    zeno_dbg_msvc(zenocore)
endif()

if (ZENO_ENABLE_OPENMP)
    find_package(OpenMP)
    if (TARGET OpenMP::OpenMP_CXX)
        message(STATUS "Found OpenMP::OpenMP_CXX")
        target_link_libraries(zenocore PRIVATE OpenMP::OpenMP_CXX)
        if (MSVC)
            include(CheckCXXCompilerFlag)
            check_cxx_compiler_flag("/openmp:experimental" MSVC_SUPPORT_OPENMP_EXPERIMENTAL)
            if (MSVC_SUPPORT_OPENMP_EXPERIMENTAL)
                target_compile_options(zenocore PRIVATE $<IF:$<COMPILE_LANGUAGE:CXX>,/openmp:experimental,>)
            endif()
            check_cxx_compiler_flag("/openmp:llvm" MSVC_SUPPORT_OPENMP_LLVM)
            if (MSVC_SUPPORT_OPENMP_LLVM)
                target_compile_options(zenocore PRIVATE $<IF:$<COMPILE_LANGUAGE:CXX>,/openmp:llvm,>)
            endif()
        endif()
    else()
        message(WARNING "Not found OpenMP")
    endif()
endif()

if (ZENO_BENCHMARKING)
    target_compile_definitions(zenocore PUBLIC -DZENO_BENCHMARKING)
endif()

if (ZENO_PARALLEL_STL)
    find_package(Threads REQUIRED)
    target_link_libraries(zenocore PRIVATE Threads::Threads)
    if (NOT MSVC)
        find_package(TBB)
        if (TBB_FOUND)
            message(STATUS "Found TBB: ${TBB_DIR}")
            target_link_libraries(zenocore PRIVATE TBB::tbb)
        else()
            message(WARNING "TBB not found, disabling parallel STL")
            set(ZENO_PARALLEL_STL OFF)
        endif()
    endif()
    if (ZENO_PARALLEL_STL)
        target_compile_definitions(zenocore PUBLIC -DZENO_PARALLEL_STL)
    endif()
endif()

if (ZENO_WITH_PYTHON3)
    message(STATUS "python3 libraries for zenocore: ${Python3_LIBRARIES}")
    message(STATUS "python3 include directories for zenocore: ${Python3_INCLUDE_DIRS}")
    target_link_libraries(zenocore PRIVATE ${Python3_LIBRARIES})
    target_include_directories(zenocore PRIVATE ${Python3_INCLUDE_DIRS})
    target_compile_definitions(zenocore PRIVATE -DZENO_WITH_PYTHON3)
endif()

if (ZENO_ENABLE_BACKWARD)
    add_subdirectory(tpls/backward-cpp)
    target_compile_definitions(zenocore PUBLIC -DZENO_ENABLE_BACKWARD)
    target_sources(zenocore PRIVATE ${BACKWARD_ENABLE})
    add_backward(zeno)
endif()

if (ZENO_BUILD_SHARED)
    target_compile_definitions(zenocore PRIVATE -DZENO_DLLEXPORT INTERFACE -DZENO_DLLIMPORT)
endif()

target_include_directories(zenocore PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/tpls/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/tpls/deprecated>
    $<INSTALL_INTERFACE:include/Zeno/zeno/include>
    ${PROJECT_SOURCE_DIR}
)

if (ZENO_INSTALL_TARGET AND ZENO_BUILD_SHARED)
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include DESTINATION include/Zeno/zeno)
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tpls/include DESTINATION include/Zeno/zeno)
    install(TARGETS zenocore EXPORT ZenoTargets)
endif()

if (ZENO_ENABLE_MAGICENUM)
    target_compile_definitions(zenocore PUBLIC -DZENO_ENABLE_MAGICENUM)
endif()

if (ZENO_WITH_REFLECTOR)
    #生成反射信息
    target_compile_definitions(zenocore PUBLIC -DZENO_WITH_REFLECTOR)
    target_link_libraries(zenocore PUBLIC libreflect)
    target_include_directories(zenocore PUBLIC
        ${CMAKE_SOURCE_DIR}/reflect/crates/libreflect/include
        ${CMAKE_BINARY_DIR}/intermediate
    )
    include(${CMAKE_SOURCE_DIR}/reflect/cmake/ReflectionUtils.cmake)

    set(GENERATE_ON_BUILD_DIR ON)
    set(REFLECT_TEMPLATE_INCLUDE "#include <zeno/types/ObjectDef.h>")
    set(INJA_TEMPLATE_DIR_PATH  ${CMAKE_BINARY_DIR}/intermediate)

        #头文件包含目录
    set(CUSTOM_HEADER_INCLUDE_DIR ${CMAKE_BINARY_DIR}/intermediate)

    #由于这个template的字符数过大导致MSVC编译期截断，需要分发template，以读写的方式加载
    file(COPY ${PROJECT_SOURCE_DIR}/reflect/src/template/reflected_type_register.inja DESTINATION ${INJA_TEMPLATE_DIR_PATH})

    # 将节点的反射和数据类型的反射分离，有助于编译时间的优化和头文件的管理

    # BEGIN - 基础数据类型和自定义类型的自动扫描
    set(reflecttypes
        ../zeno/src/nodes/reflect/zenoreflecttypes.cpp
    )

    make_absolute_paths(REFLECTION_TYPES ${reflecttypes})
    #遍历整个目录，并收集所有标有ZRECORD或者ZPRIMITIVE的类的文件，暂时不包含project文件夹，以后可能插件化
    #如果有什么动态Object或者类型，现在这里添加搜索目录（如果没有的话）
    file(GLOB_RECURSE scan_source CONFIGURE_DEPENDS 
        include/*.h
        src/*.cpp
        ../zeno/*.h
        ../zeno/*.cpp
    )

    set(REFLECTE_ZENO_TYPES_HEADER "dummy")
    foreach(file_path ${scan_source})
        file(READ ${file_path} CURR_FILE_CONTENT)
        string(FIND "${CURR_FILE_CONTENT}" "ZPRIMITIVE" OUT_MATCH_RESULT)
        if (NOT OUT_MATCH_RESULT EQUAL -1)
            list(APPEND REFLECTE_ZENO_TYPES_HEADER ${file_path})
        endif()

        string(FIND "${CURR_FILE_CONTENT}" "REFLECT_REGISTER_OBJECT" OUT_MATCH_RESULT)
        if (NOT OUT_MATCH_RESULT EQUAL -1)
            list(APPEND REFLECTE_ZENO_TYPES_HEADER ${file_path})
        endif()
    endforeach()
    list(POP_FRONT REFLECTE_ZENO_TYPES_HEADER)    #把最开始的移除掉

    #送入Generator并生成
    zeno_declare_reflection_support(zenocore "${REFLECTE_ZENO_TYPES_HEADER}" zeno_types)
    # END - 基础数据类型和自定义类型的自动扫描


    # BEGIN - 节点定义的自动扫描，在这里我们把带有ZDEFNODE标识的约定为节点的定义

    # 生成节点类的反射信息
    set(REFLECTION_NODES "dummy")
    #make_absolute_paths(REFLECTION_NODES ${reflectnodes})
    foreach(file_path ${scan_source})
        file(READ ${file_path} CURR_FILE_CONTENT)
        string(FIND "${CURR_FILE_CONTENT}" "ZDEFNODE" OUT_MATCH_RESULT)
        if (NOT OUT_MATCH_RESULT EQUAL -1)
            list(APPEND REFLECTION_NODES ${file_path})
        endif()
    endforeach()
    list(POP_FRONT REFLECTION_NODES)
    #送入Generator并生成
    zeno_declare_reflection_support(zenocore "${REFLECTION_NODES}" zeno_nodes)
    # END - 节点定义的自动扫描，在这里我们把带有ZDEFNODE标识的约定为节点的定义
endif()

#if (ZENO_NO_WARNING)
    #if (CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        #target_compile_options(zeno PUBLIC $<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:CXX>:-Wno-all -Wno-cpp -Wno-deprecated-declarations -Wno-enum-compare -Wno-ignored-attributes -Wno-extra -Wreturn-type -Wmissing-declarations -Wnon-virtual-dtor -Wsuggest-override -Wconversion-null>>)
    #endif()
    #if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        #target_compile_options(zeno PUBLIC $<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:C,CXX>:-w>>)
    #endif()
#endif()

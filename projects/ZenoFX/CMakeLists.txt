add_library(ZenoFXlib SHARED)
target_include_directories(ZenoFXlib PRIVATE .)

find_package(zeno REQUIRED)
target_link_libraries(ZenoFXlib PRIVATE zeno ZFX)
target_sources(ZenoFXlib PRIVATE pw.cpp pnw.cpp GottaFuckCpp.cpp)

option(ZENOFX_ENABLE_OPENVDB "Enable Volume Wranglers in ZenoFX" ON)
if (ZENOFX_ENABLE_OPENVDB)
    find_package(TBB CONFIG REQUIRED COMPONENTS tbb tbbmalloc)
    find_package(IlmBase REQUIRED COMPONENTS Half REQUIRED)
    find_package(OpenVDB REQUIRED)
    target_link_libraries(ZenoFXlib PRIVATE
        OpenVDB::openvdb IlmBase::Half TBB::tbbmalloc TBB::tbb)
    target_sources(ZenoFXlib PRIVATE vw.cpp)
    target_include_directories(ZenoFXlib PRIVATE ../zenvdb/include)
endif()

find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set_target_properties(ZenoFXlib PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${zeno_AUTOLOAD_DIR}
    RUNTIME_OUTPUT_DIRECTORY ${zeno_AUTOLOAD_DIR}
    ARCHIVE_OUTPUT_DIRECTORY ${zeno_AUTOLOAD_DIR}
    LIBRARY_OUTPUT_DIRECTORY_DEBUG ${zeno_AUTOLOAD_DIR}
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${zeno_AUTOLOAD_DIR}
    ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${zeno_AUTOLOAD_DIR}
    LIBRARY_OUTPUT_DIRECTORY_RELEASE ${zeno_AUTOLOAD_DIR}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${zeno_AUTOLOAD_DIR}
    ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${zeno_AUTOLOAD_DIR}
    )

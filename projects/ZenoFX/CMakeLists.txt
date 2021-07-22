add_library(ZenoFXlib SHARED)
target_include_directories(ZenoFXlib PRIVATE .)

find_package(zeno REQUIRED)
target_link_libraries(ZenoFXlib PRIVATE zeno ZFX)
target_sources(ZenoFXlib PRIVATE pw.cpp pnw.cpp)

option(ZENOFX_ENABLE_OPENVDB "Enable OpenVDB wrangler support in ZenoFX" ON)
if (ZENOFX_ENABLE_OPENVDB)
    target_sources(ZenoFXlib PRIVATE vw.cpp)
    target_include_directories(ZenoFXlib PRIVATE ../zenvdb/include)
    target_link_libraries(ZenoFXlib PRIVATE openvdb)
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

add_library(ZenoFXlib SHARED)
target_include_directories(ZenoFXlib PRIVATE .)

target_link_libraries(ZenoFXlib PRIVATE zeno ZFX)
target_sources(ZenoFXlib PRIVATE nw.cpp pw.cpp pnw.cpp ppw.cpp)

if (EXTENSION_zenvdb)
    option(ZENOFX_ENABLE_OPENVDB "Enable OpenVDB wrangler in ZenoFX" ON)
else()
    option(ZENOFX_ENABLE_OPENVDB "Enable OpenVDB wrangler in ZenoFX" OFF)
endif()
if (ZENOFX_ENABLE_OPENVDB)
    target_sources(ZenoFXlib PRIVATE vw.cpp)
    target_include_directories(ZenoFXlib PRIVATE ../zenvdb/include)
    target_link_libraries(ZenoFXlib PRIVATE openvdb)
endif()

get_property(zeno_AUTOLOAD_DIR TARGET zeno PROPERTY LIBRARY_OUTPUT_DIRECTORY)
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

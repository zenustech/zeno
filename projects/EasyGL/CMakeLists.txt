#add_library(zeno_EasyGL SHARED)
add_executable(zeno_EasyGL)
target_include_directories(zeno_EasyGL PRIVATE .)

target_link_libraries(zeno_EasyGL PRIVATE zeno)
target_link_libraries(zeno_EasyGL PRIVATE GLESv2)
target_link_libraries(zeno_EasyGL PRIVATE glut)
target_sources(zeno_EasyGL PRIVATE main.cpp)

if (0 AND ZENO_ENABLE_PYTHON)
    get_property(zeno_AUTOLOAD_DIR TARGET zeno PROPERTY LIBRARY_OUTPUT_DIRECTORY)
    set_target_properties(zeno_EasyGL PROPERTIES
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
endif()

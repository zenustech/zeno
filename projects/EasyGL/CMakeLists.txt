add_library(zeno_EasyGL SHARED)
target_include_directories(zeno_EasyGL PRIVATE .)

target_link_libraries(zeno_EasyGL PUBLIC zeno)
target_link_libraries(zeno_EasyGL PUBLIC GLESv2)
target_sources(zeno_EasyGL PRIVATE nodes.cpp passtoy.cpp)

option(EASYGL_ENABLE_GLUT "Enable GLUT nodes in ZENO EasyGL" ON)
if (EASYGL_ENABLE_GLUT)
    target_sources(zeno_EasyGL PRIVATE glut.cpp)
    target_link_libraries(zeno_EasyGL PUBLIC glut)
endif()

if (ZENO_ENABLE_PYTHON)
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

#add_executable(zeno_EasyGL_main main.cpp)
#target_link_libraries(zeno_EasyGL_main PRIVATE zeno)
#target_link_libraries(zeno_EasyGL_main PRIVATE zeno_EasyGL)
#target_link_libraries(zeno_EasyGL_main PRIVATE GLESv2)
#target_link_libraries(zeno_EasyGL_main PRIVATE glut)

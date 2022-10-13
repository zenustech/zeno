if(ZENO_WITH_LIBIGL)
    if (NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/libigl/CMakeLists.txt)
        message(FATAL_ERROR "libigl submodule not found! Please run: git submodule update --init --recursive")
    endif()


    # option(LIBIGL_BUILD_TESTS      "Build libigl unit test"        ${LIBIGL_TOPLEVEL_PROJECT})
    # option(LIBIGL_BUILD_TUTORIALS  "Build libigl tutorial"         ${LIBIGL_TOPLEVEL_PROJECT})
    # option(LIBIGL_EXPORT_TARGETS   "Export libigl CMake targets"   ${LIBIGL_TOPLEVEL_PROJECT})
    
    # USE_STATIC_LIBRARY speeds up the generation of multiple binaries,
    # at the cost of a longer initial compilation time
    # (by default, static build is off since libigl is a header-only library)
    # set(LIBIGL_USE_STATIC_LIBRARY "Use libigl as static library" ON)
    
    # All dependencies that are downloaded as cmake projects and tested on the auto-builds are ON
    # (by default, all build options are off)
    # set(LIBIGL_WITH_COMISO            "Use CoMiso"                   ON  CACHE BOOL "EMBED")
    # set(LIBIGL_WITH_EMBREE            "Use Embree"                   OFF  CACHE BOOL "EMBED")
    # set(LIBIGL_WITH_OPENGL            "Use OpenGL"                   OFF  CACHE BOOL "EMBED")
    # set(LIBIGL_WITH_OPENGL_GLFW       "Use GLFW"                     OFF  CACHE BOOL "EMBED")
    # set(LIBIGL_WITH_OPENGL_GLFW_IMGUI "Use ImGui"                    OFF  CACHE BOOL "EMBED")
    # set(LIBIGL_WITH_PNG               "Use PNG"                      OFF  CACHE BOOL "EMBED")
    # set(LIBIGL_WITH_TETGEN            "Use Tetgen"                   ON  CACHE BOOL "EMBED")
    # set(LIBIGL_WITH_TRIANGLE          "Use Triangle"                 OFF  CACHE BOOL "EMBED")
    # set(LIBIGL_WITH_PREDICATES        "Use exact predicates"         OFF  CACHE BOOL "EMBED")
    # set(LIBIGL_WITH_XML               "Use XML"                      OFF  CACHE BOOL "EMBED")
    # set(LIBIGL_WITH_PYTHON            "Use Python"                   OFF  CACHE BOOL "EMBED")


    # set(LIBIGL_USE_STATIC_LIBRARY     OFF CACHE BOOL "EMBED")
    # set(LIBIGL_WITH_CGAL              ON  CACHE BOOL "EMBED")
    # set(LIBIGL_WITH_TETGEN            ON  CACHE BOOL "EMBED")
    # set(LIBIGL_WITH_COMISO            ON CACHE BOOL "EMBED")
    # set(LIBIGL_WITH_EMBREE            OFF CACHE BOOL "EMBED")
    # set(LIBIGL_WITH_OPENGL            OFF CACHE BOOL "EMBED")
    # set(LIBIGL_WITH_OPENGL_GLFW       OFF CACHE BOOL "EMBED")
    # set(LIBIGL_WITH_OPENGL_GLFW_IMGUI OFF CACHE BOOL "EMBED")
    # set(LIBIGL_WITH_PNG               OFF CACHE BOOL "EMBED")
    # set(LIBIGL_WITH_TETGEN            OFF CACHE BOOL "EMBED")
    # set(LIBIGL_WITH_TRIANGLE          OFF CACHE BOOL "EMBED")
    # set(LIBIGL_WITH_PREDICATES        OFF CACHE BOOL "EMBED")
    # set(LIBIGL_WITH_XML               OFF CACHE BOOL "EMBED")
    # set(LIBIGL_WITH_PYTHON            OFF CACHE BOOL "EMBED")

    add_subdirectory(libigl)    
endif()

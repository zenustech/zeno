message("Finding package ZENO...")

if (PYTHON_EXECUTABLE)
    message("Using ${PYTHON_EXECUTABLE} as python executable.")
else ()
    if (WIN32)
        message("Using 'python' as python interpreter.")
        set(PYTHON_EXECUTABLE python)
    else ()
        message("Using 'python3' as python interpreter.")
        set(PYTHON_EXECUTABLE python3)
    endif()
endif ()
execute_process(COMMAND ${PYTHON_EXECUTABLE} --version)

execute_process(COMMAND ${PYTHON_EXECUTABLE} -c
        "import zenutils"
        RESULT_VARIABLE zeno_IMPORT_RET)
if (zeno_IMPORT_RET)
    # returns zero if success
    message(FATAL_ERROR "Failed to import zenutils. Have you installed it or add it to PYTHONPATH?")
endif ()


execute_process(COMMAND ${PYTHON_EXECUTABLE} -c
        "import zenutils, sys, os; sys.stdout.write(os.path.abspath(zenutils.rel2abs(zenutils.__file__, '..', 'zeno')))"
        OUTPUT_VARIABLE zeno_INSTALL_DIR)

message("Found ZENO at: ${zeno_INSTALL_DIR}")

set(zeno_AUTOLOAD_DIR ${zeno_INSTALL_DIR}/lib)
set(zeno_CMAKE_MODULE_DIR ${zeno_INSTALL_DIR}/share/cmake)
set(zeno_INCLUDE_DIR ${zeno_INSTALL_DIR}/include)
set(zeno_LIBRARY_DIR ${zeno_INSTALL_DIR}/lib)
if (NOT WIN32)
    set(zeno_LIBRARY ${zeno_LIBRARY_DIR}/libzeno.so)
else()
    set(zeno_LIBRARY ${zeno_LIBRARY_DIR}/zeno.lib)
endif()

#message("zeno_INSTALL_DIR=${zeno_INSTALL_DIR}")
#message("zeno_INCLUDE_DIR=${zeno_INCLUDE_DIR}")
#message("zeno_CMAKE_MODULE_DIR=${zeno_CMAKE_MODULE_DIR}")
#message("zeno_AUTOLOAD_DIR=${zeno_AUTOLOAD_DIR}")
#message("zeno_LIBRARY_DIR=${zeno_LIBRARY_DIR}")
#message("zeno_LIBRARY=${zeno_LIBRARY}")
if (NOT TARGET zeno)
    add_library(zeno INTERFACE)
    target_link_libraries(zeno INTERFACE ${zeno_LIBRARY})
    target_include_directories(zeno INTERFACE ${zeno_INCLUDE_DIR})
endif()

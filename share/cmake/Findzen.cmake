message("Finding package zen...")

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
        "import zen"
        RESULT_VARIABLE zen_IMPORT_RET)
if (zen_IMPORT_RET)
    # returns zero if success
    message(FATAL_ERROR "Failed to import zen. Have you installed it or add it to PYTHONPATH? Also try clean autoload folder?")
endif ()


execute_process(COMMAND ${PYTHON_EXECUTABLE} -c
        "import sys; import zen; sys.stdout.write(zen.getInstallDir())"
        OUTPUT_VARIABLE zen_INSTALL_DIR)

message("zen_INSTALL_DIR=${zen_INSTALL_DIR}")

set(zen_AUTOLOAD_DIR ${zen_INSTALL_DIR}/autoload)
set(zen_CMAKE_MODULE_DIR ${zen_INSTALL_DIR}/usr/share/cmake)
set(zen_INCLUDE_DIR ${zen_INSTALL_DIR}/usr/include)
set(zen_LIBRARY_DIR ${zen_INSTALL_DIR}/usr/lib)
if (NOT WIN32)
    set(zen_LIBRARY ${zen_INSTALL_DIR}/usr/lib/libzensession.so)
else()
    set(zen_LIBRARY ${zen_INSTALL_DIR}/usr/lib/zensession.lib)
endif()

message("zen_INCLUDE_DIR=${zen_INCLUDE_DIR}")
message("zen_CMAKE_MODULE_DIR=${zen_CMAKE_MODULE_DIR}")
message("zen_AUTOLOAD_DIR=${zen_AUTOLOAD_DIR}")
message("zen_LIBRARY_DIR=${zen_LIBRARY_DIR}")
message("zen_LIBRARY=${zen_LIBRARY}")
if (NOT TARGET zen)
    add_library(zen INTERFACE)
    target_link_libraries(zen INTERFACE ${zen_LIBRARY})
    target_include_directories(zen INTERFACE ${zen_INCLUDE_DIR})
endif()

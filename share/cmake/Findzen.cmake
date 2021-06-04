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
    message(FATAL_ERROR "Cannot import zen. Have you installed it or add it to PYTHONPATH?")
endif ()


execute_process(COMMAND ${PYTHON_EXECUTABLE} -c
        "import sys; import zen; sys.stdout.write(zen.getInstallDir())"
        OUTPUT_VARIABLE zen_INSTALL_DIR)

message("zen_INSTALL_DIR=${zen_INSTALL_DIR}")

set(zen_AUTOLOAD_DIR ${zen_INSTALL_DIR}/autoload)
set(zen_CMAKE_MODULE_DIR ${zen_INSTALL_DIR}/usr/share/cmake)
set(zen_INCLUDE_DIR ${zen_INSTALL_DIR}/usr/include)
set(zen_LIBRARY_DIR ${zen_INSTALL_DIR}/usr/lib)

add_library(zen INTERFACE)
target_include_directories(zen INTERFACE ${zen_INCLUDE_DIR})
link_directories(${zen_LIBRARY_DIR})
target_link_libraries(zen INTERFACE zensession)

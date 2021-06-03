message("Finding ZENO...")

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
        RESULT_VARIABLE ZENO_IMPORT_RET)
if (ZENO_IMPORT_RET)
    # returns zero if success
    message(FATAL_ERROR "Cannot import zen. Have you installed it or add it to PYTHONPATH?")
endif ()

execute_process(COMMAND ${PYTHON_EXECUTABLE} -c
        "import sys; import zen; sys.stdout.write(zen.getIncludeDir())"
        OUTPUT_VARIABLE ZENO_INCLUDE_DIR)

execute_process(COMMAND ${PYTHON_EXECUTABLE} -c
        "import sys; import zen; sys.stdout.write(zen.getLibraryDir())"
        OUTPUT_VARIABLE ZENO_LIBRARY_DIR)

link_directories(${ZENO_LIBRARY_DIR})
include_directories(${ZENO_INCLUDE_DIR})

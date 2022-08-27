file(GLOB_RECURSE sources CONFIGURE_DEPENDS *.cpp *.h)
target_sources(zeno PRIVATE ${sources})

set(BUILD_LIBPYTHON_SHARED ON CACHE BOOL "ZENOCIHOUPYTHON" FORCE)
if (NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/CPython/CMakeLists.txt)
    message(FATAL_ERROR "CPython submodule not found! Please run: git submodule update --init --recursive")
endif()
add_subdirectory(CPython)

target_link_libraries(zeno PRIVATE libpython3-shared)

set(PYINCDIR
    ${CMAKE_CURRENT_BINARY_DIR}/Python-${PYTHON_VERSION}/Include   # Python.h
    ${CMAKE_CURRENT_BINARY_DIR}/CPython/bin   # pyconfig.h
    )
target_include_directories(zeno PRIVATE ${PYINCDIR})

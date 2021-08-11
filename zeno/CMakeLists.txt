file(GLOB ZENO_SOURCE core/*.cpp utils/*.cpp extra/*.cpp include/*/*.h include/*/*/*.h)
file(GLOB ZENO_NODES_SOURCE nodes/*.cpp nodes/*/*.cpp)

add_library(zeno SHARED ${ZENO_SOURCE} ${ZENO_NODES_SOURCE})

if (ZENO_ENABLE_OPENMP)
    find_package(OpenMP)
    if (OPENMP_FOUND)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    endif()
endif()

if (ZENO_VISUALIZATION)
    target_compile_definitions(zeno PUBLIC -DZENO_VISUALIZATION)
    if (UNIX OR APPLE)
		target_link_libraries(zeno PRIVATE stdc++fs)  # for <experimental/filesystem>
	endif()
endif()
if (ZENO_GLOBALSTATE)
    target_compile_definitions(zeno PUBLIC -DZENO_GLOBALSTATE)
endif()
if (ZENO_BENCHMARKING)
    target_compile_definitions(zeno PRIVATE -DZENO_BENCHMARKING)
endif()
if (ZENO_FAULTHANDLER)
    target_compile_definitions(zeno PRIVATE -DZENO_FAULTHANDLER)
endif()

target_compile_definitions(zeno PRIVATE -DDLL_ZENO)
target_include_directories(zeno PUBLIC include)

target_link_libraries(zeno PRIVATE spdlog)

set(OUTPUT_DIR ${PROJECT_SOURCE_DIR}/zenqt/lib)
set_target_properties(zeno PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIR}
    ARCHIVE_OUTPUT_DIRECTORY ${OUTPUT_DIR}
    LIBRARY_OUTPUT_DIRECTORY_DEBUG ${OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${OUTPUT_DIR}
    ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${OUTPUT_DIR}
    LIBRARY_OUTPUT_DIRECTORY_RELEASE ${OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${OUTPUT_DIR}
    ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${OUTPUT_DIR}
    )
